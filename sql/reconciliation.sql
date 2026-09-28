-- Reconciling predictions with labels that arrive late.

-- 1. One week of predictions, watched as it matures.
WITH target_week AS (
    SELECT
        (SELECT min(event_time)::date FROM predictions) + INTERVAL '7 days'  AS week_start,
        (SELECT min(event_time)::date FROM predictions) + INTERVAL '14 days' AS week_end
),
vantage_points AS (
    -- days after the END of the week being judged
    SELECT unnest(ARRAY[7, 21, 45, 70, 90, 120]) AS days_after
),
scored AS (
    SELECT
        v.days_after,
        (w.week_end + (v.days_after || ' days')::interval)::date AS as_of,
        p.prediction,
        l.label,
        (l.available_at <= w.week_end + (v.days_after || ' days')::interval) AS revealed
    FROM predictions p
    JOIN labels l ON l.request_id = p.request_id
    CROSS JOIN target_week w
    CROSS JOIN vantage_points v
    WHERE p.event_time >= w.week_start AND p.event_time < w.week_end
)
SELECT
    days_after                                        AS days_after_week,
    as_of,
    count(*)                                          AS n_predictions,
    count(*) FILTER (WHERE revealed)                  AS n_labelled,
    round((100.0 * count(*) FILTER (WHERE revealed) / count(*))::numeric, 1) AS coverage_pct,
    -- the column that explains the distortion: the labelled sample is not the population
    round(avg(label::numeric) FILTER (WHERE revealed), 3)                    AS labelled_positive_rate,
    round((count(*) FILTER (WHERE revealed AND prediction = 'churn' AND label = 1)::numeric
         / nullif(count(*) FILTER (WHERE revealed AND prediction = 'churn'), 0)), 3) AS precision,
    round((count(*) FILTER (WHERE revealed AND prediction = 'churn' AND label = 1)::numeric
         / nullif(count(*) FILTER (WHERE revealed AND label = 1), 0)), 3)            AS recall
FROM scored
GROUP BY days_after, as_of
ORDER BY days_after;


-- 2. The same thing across the whole timeline, at one fixed vantage point.
WITH as_of AS (
    SELECT (SELECT max(event_time)::date FROM predictions) + INTERVAL '45 days' AS t
),
scored AS (
    SELECT
        date_trunc('week', p.event_time)::date AS prediction_week,
        p.prediction,
        l.label,
        (l.available_at <= (SELECT t FROM as_of)) AS revealed
    FROM predictions p
    JOIN labels l ON l.request_id = p.request_id
)
SELECT
    prediction_week,
    count(*)                                          AS n_predictions,
    round((100.0 * count(*) FILTER (WHERE revealed) / count(*))::numeric, 1) AS coverage_pct,
    round(avg(label::numeric) FILTER (WHERE revealed), 3)                    AS labelled_positive_rate,
    round((count(*) FILTER (WHERE revealed AND prediction = 'churn' AND label = 1)::numeric
         / nullif(count(*) FILTER (WHERE revealed AND prediction = 'churn'), 0)), 3) AS precision,
    round((count(*) FILTER (WHERE revealed AND prediction = 'churn' AND label = 1)::numeric
         / nullif(count(*) FILTER (WHERE revealed AND label = 1), 0)), 3)            AS recall
FROM scored
GROUP BY prediction_week
ORDER BY prediction_week;


-- 3. The mature truth, for comparison: every label revealed.
WITH bounds AS (
    SELECT (SELECT min(event_time)::date FROM predictions) + INTERVAL '20 days' AS ramp_start
)
SELECT
    CASE WHEN p.event_time < b.ramp_start THEN 'baseline' ELSE 'ramp' END AS period,
    count(*)                                                              AS n,
    round(avg(l.label::numeric), 3)                                       AS prevalence,
    round((count(*) FILTER (WHERE p.prediction = 'churn' AND l.label = 1)::numeric
         / nullif(count(*) FILTER (WHERE p.prediction = 'churn'), 0)), 3) AS precision,
    round((count(*) FILTER (WHERE p.prediction = 'churn' AND l.label = 1)::numeric
         / nullif(count(*) FILTER (WHERE l.label = 1), 0)), 3)            AS recall
FROM predictions p
JOIN labels l ON l.request_id = p.request_id
CROSS JOIN bounds b
GROUP BY period
ORDER BY period;
