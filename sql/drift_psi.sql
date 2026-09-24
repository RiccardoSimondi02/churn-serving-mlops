-- Drift: PSI per feature, per rolling window, against the `val` reference.
-- The baseline is the first 20 simulated days: the harness sends only `test`
-- traffic there, so whatever PSI shows up is sampling noise, not drift.

-- 1. Noise floor against signal, for both window widths.
WITH bounds AS (
    SELECT min(day) + INTERVAL '20 days' AS ramp_start FROM drift_metrics
),
labelled AS (
    SELECT d.feature, d.window_days, d.psi,
           CASE WHEN d.day < b.ramp_start THEN 'baseline' ELSE 'ramp' END AS period
    FROM drift_metrics d CROSS JOIN bounds b
),
agg AS (
    SELECT
        feature,
        window_days,
        percentile_disc(0.50) WITHIN GROUP (ORDER BY psi)
            FILTER (WHERE period = 'baseline') AS baseline_p50,
        percentile_disc(0.95) WITHIN GROUP (ORDER BY psi)
            FILTER (WHERE period = 'baseline') AS baseline_p95,
        max(psi) FILTER (WHERE period = 'baseline') AS baseline_max,
        percentile_disc(0.50) WITHIN GROUP (ORDER BY psi)
            FILTER (WHERE period = 'ramp')     AS ramp_p50,
        max(psi) FILTER (WHERE period = 'ramp') AS ramp_max
    FROM labelled
    GROUP BY feature, window_days
)
SELECT
    feature,
    window_days,
    round(baseline_p50::numeric, 4) AS baseline_p50,
    round(baseline_p95::numeric, 4) AS baseline_p95,
    round(baseline_max::numeric, 4) AS baseline_max,
    -- the alert line query 4 uses, shown here so the two stay readable together
    round((baseline_p50 + 3 * (baseline_p95 - baseline_p50))::numeric, 4) AS alert_at,
    round(ramp_p50::numeric, 4)     AS ramp_p50,
    round(ramp_max::numeric, 4)     AS ramp_max,
    round((ramp_p50 / nullif(baseline_max, 0))::numeric, 2) AS separation
FROM agg
ORDER BY separation DESC NULLS LAST, feature;

-- 4. Detection day.
--
--    The alert line is built from the SHAPE of the quiet distribution, not
--    from its maximum: p50 + 3 * (p95 - p50). The maximum is decided by one
--    unlucky window and grows the longer you observe, so a line built on it is
--    fitted to an accident and does not survive into the next period. This one
--    uses the whole distribution and moves little when a single window is odd.
--
--    A single crossing is enough. There is no "two windows in a row" rule,
--    because consecutive 7-day rolling windows share six days out of seven:
--    one bad day appears in seven consecutive windows, so a persistence rule
--    would fire on exactly the case it was meant to filter out. The smoothing
--    is done by the window itself, one odd day is a seventh of the rows.
--
--    `shifted_pct_that_day` is the share of traffic from the held-out
--    tenure >= 60 segment that day: the ground truth this demo has and
--    production does not, and what makes "caught while still at X%" sayable.
WITH bounds AS (
    SELECT min(day) + INTERVAL '20 days' AS ramp_start FROM drift_metrics
),
thresholds AS (
    SELECT
        d.feature,
        d.window_days,
        percentile_disc(0.50) WITHIN GROUP (ORDER BY d.psi)
          + 3 * (percentile_disc(0.95) WITHIN GROUP (ORDER BY d.psi)
               - percentile_disc(0.50) WITHIN GROUP (ORDER BY d.psi)) AS alert_at
    FROM drift_metrics d CROSS JOIN bounds b
    WHERE d.day < b.ramp_start          -- quiet period only
    GROUP BY d.feature, d.window_days
),
daily_mix AS (
    SELECT event_time::date AS day,
           avg(CASE WHEN tenure >= 60 THEN 1.0 ELSE 0.0 END) AS shifted_share
    FROM predictions
    GROUP BY 1
),
crossings AS (
    SELECT
        d.feature,
        d.window_days,
        t.alert_at,
        min(d.day)                                                      AS alert_day,
        -- must be zero: an alert inside the quiet period means the line is too low
        count(*) FILTER (WHERE d.day < (SELECT ramp_start FROM bounds)) AS false_alarms
    FROM drift_metrics d
    JOIN thresholds t
      ON t.feature = d.feature AND t.window_days = d.window_days
    WHERE d.psi >= t.alert_at
    GROUP BY d.feature, d.window_days, t.alert_at
)
SELECT
    c.feature,
    c.window_days,
    round(c.alert_at::numeric, 4)                        AS alert_at,
    c.false_alarms,
    c.alert_day,
    (c.alert_day - (SELECT min(day) FROM drift_metrics)) AS day_index,
    round((100 * m.shifted_share)::numeric, 1)           AS shifted_pct_that_day
FROM crossings c
LEFT JOIN daily_mix m ON m.day = c.alert_day
ORDER BY c.window_days, c.alert_day;
