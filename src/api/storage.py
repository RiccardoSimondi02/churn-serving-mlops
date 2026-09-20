import os

from sqlalchemy import create_engine, text


def build_engine():
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return create_engine(database_url)

    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_NAME = os.getenv("DB_NAME")
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@localhost:5432/{DB_NAME}")
    return engine


def insert_prediction(engine, payload, response, time):
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO predictions (
                    request_id, customer_id, gender, "SeniorCitizen", "Partner", "Dependents",
                    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
                    tenure, "MonthlyCharges", "TotalCharges",
                    prediction, threshold, probability, model_version, created_at
                ) VALUES (
                    :request_id, :customer_id, :gender, :SeniorCitizen, :Partner, :Dependents,
                    :PhoneService, :MultipleLines, :InternetService, :OnlineSecurity,
                    :OnlineBackup, :DeviceProtection, :TechSupport, :StreamingTV,
                    :StreamingMovies, :Contract, :PaperlessBilling, :PaymentMethod,
                    :tenure, :MonthlyCharges, :TotalCharges,
                    :prediction, :threshold, :probability, :model_version, :created_at
                )
                """
            ),
            {
                "request_id": response.request_id,
                "customer_id": payload.customer_id,
                "gender": payload.gender,
                "SeniorCitizen": payload.SeniorCitizen,
                "Partner": payload.Partner,
                "Dependents": payload.Dependents,
                "PhoneService": payload.PhoneService,
                "MultipleLines": payload.MultipleLines,
                "InternetService": payload.InternetService,
                "OnlineSecurity": payload.OnlineSecurity,
                "OnlineBackup": payload.OnlineBackup,
                "DeviceProtection": payload.DeviceProtection,
                "TechSupport": payload.TechSupport,
                "StreamingTV": payload.StreamingTV,
                "StreamingMovies": payload.StreamingMovies,
                "Contract": payload.Contract,
                "PaperlessBilling": payload.PaperlessBilling,
                "PaymentMethod": payload.PaymentMethod,
                "tenure": payload.tenure,
                "MonthlyCharges": payload.MonthlyCharges,
                "TotalCharges": payload.TotalCharges,
                "prediction": response.prediction,
                "threshold": response.threshold,
                "probability": response.probability,
                "model_version": response.model_version,
                "created_at": time,
            },
        )
        conn.commit()