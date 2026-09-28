-- 1. Distribution of delays by labels
SELECT l.label,
       count(*) AS n,
       round(min(extract(epoch FROM l.available_at - p.event_time) / 86400)::numeric, 1) AS min_days,
       round(avg(extract(epoch FROM l.available_at - p.event_time) / 86400)::numeric, 1) AS avg_days,
       round(max(extract(epoch FROM l.available_at - p.event_time) / 86400)::numeric, 1) AS max_days
FROM labels l JOIN predictions p USING (request_id)
GROUP BY l.label;


-- 2. Labels available at an intirmidiate date
SELECT l.label, count(*) AS available
FROM labels l
WHERE l.available_at <= (SELECT min(event_time) FROM predictions) + INTERVAL '60 days'
GROUP BY l.label;
