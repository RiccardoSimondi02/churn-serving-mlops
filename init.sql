CREATE TABLE predictions (
    request_id TEXT PRIMARY KEY,

    -- Request payload: one column per feature (no JSON blob, so it stays queryable).
    customer_id TEXT NOT NULL,
    gender TEXT NOT NULL,
    "SeniorCitizen" SMALLINT NOT NULL,
    "Partner" TEXT NOT NULL,
    "Dependents" TEXT NOT NULL,
    "PhoneService" TEXT NOT NULL,
    "MultipleLines" TEXT NOT NULL,
    "InternetService" TEXT NOT NULL,
    "OnlineSecurity" TEXT NOT NULL,
    "OnlineBackup" TEXT NOT NULL,
    "DeviceProtection" TEXT NOT NULL,
    "TechSupport" TEXT NOT NULL,
    "StreamingTV" TEXT NOT NULL,
    "StreamingMovies" TEXT NOT NULL,
    "Contract" TEXT NOT NULL,
    "PaperlessBilling" TEXT NOT NULL,
    "PaymentMethod" TEXT NOT NULL,
    tenure INTEGER NOT NULL,
    "MonthlyCharges" DOUBLE PRECISION NOT NULL,
    "TotalCharges" DOUBLE PRECISION,

    -- Response payload, minus request_id/customer_id already stored above.
    prediction TEXT NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    model_version TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL,
    event_time TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_predictions_customer_id ON predictions (customer_id);
CREATE INDEX idx_predictions_created_at ON predictions (created_at);

CREATE INDEX idx_predictions_time_event_ ON predictions (event_time);


CREATE TABLE request_log (
    request_id TEXT NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    duration_ms DOUBLE PRECISION NOT NULL,
    routing_ms DOUBLE PRECISION,
    predict_ms DOUBLE PRECISION,
    insert_ms DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_request_log_request_id ON request_log (request_id);