-- 1. Total duration percentiles.
SELECT
    count(*)                                                    AS n,
    percentile_disc(0.50) WITHIN GROUP (ORDER BY duration_ms)   AS p50_ms,
    percentile_disc(0.95) WITHIN GROUP (ORDER BY duration_ms)   AS p95_ms,
    percentile_disc(0.99) WITHIN GROUP (ORDER BY duration_ms)   AS p99_ms,
    max(duration_ms)                                            AS max_ms
FROM request_log
WHERE path = '/predict' AND status_code = 200;


-- 2. Per-phase percentiles, one row per phase.
SELECT
    phase,
    percentile_disc(0.50) WITHIN GROUP (ORDER BY ms) AS p50_ms,
    percentile_disc(0.95) WITHIN GROUP (ORDER BY ms) AS p95_ms,
    percentile_disc(0.99) WITHIN GROUP (ORDER BY ms) AS p99_ms
FROM request_log
CROSS JOIN LATERAL (VALUES
    (1, 'routing', routing_ms),
    (2, 'predict', predict_ms),
    (3, 'insert',  insert_ms),
    (4, 'total',   duration_ms)
) AS p(ord, phase, ms)
WHERE path = '/predict' AND status_code = 200
  AND routing_ms IS NOT NULL AND predict_ms IS NOT NULL AND insert_ms IS NOT NULL
GROUP BY ord, phase
ORDER BY ord;


-- 3. Composition of a typical request vs a slow one.
WITH base AS (
    SELECT
        duration_ms, routing_ms, predict_ms, insert_ms,
        duration_ms - (routing_ms + predict_ms + insert_ms) AS other_ms
    FROM request_log
    WHERE path = '/predict' AND status_code = 200
      AND routing_ms IS NOT NULL AND predict_ms IS NOT NULL AND insert_ms IS NOT NULL
),
thresholds AS (
    SELECT
        percentile_disc(0.45) WITHIN GROUP (ORDER BY duration_ms) AS p45,
        percentile_disc(0.55) WITHIN GROUP (ORDER BY duration_ms) AS p55,
        percentile_disc(0.99) WITHIN GROUP (ORDER BY duration_ms) AS p99
    FROM base
),
bucketed AS (
    SELECT
        b.*,
        CASE
            WHEN b.duration_ms BETWEEN t.p45 AND t.p55 THEN 'p50'
            WHEN b.duration_ms >= t.p99                THEN 'p99+'
        END AS bucket
    FROM base b
    CROSS JOIN thresholds t
)
SELECT
    bucket,
    count(*)                                                AS n,
    round(avg(duration_ms)::numeric, 2)                     AS avg_total_ms,
    round(avg(routing_ms)::numeric, 2)                      AS routing_ms,
    round(avg(predict_ms)::numeric, 2)                      AS predict_ms,
    round(avg(insert_ms)::numeric, 2)                       AS insert_ms,
    round(avg(other_ms)::numeric, 2)                        AS other_ms,
    round((100 * avg(routing_ms) / avg(duration_ms))::numeric, 1) AS routing_pct,
    round((100 * avg(predict_ms) / avg(duration_ms))::numeric, 1) AS predict_pct,
    round((100 * avg(insert_ms)  / avg(duration_ms))::numeric, 1) AS insert_pct,
    round((100 * avg(other_ms)   / avg(duration_ms))::numeric, 1) AS other_pct
FROM bucketed
WHERE bucket IS NOT NULL
GROUP BY bucket
ORDER BY bucket;