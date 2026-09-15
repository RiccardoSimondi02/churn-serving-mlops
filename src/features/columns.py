ID_COLUMN = "customerID"

TARGET_COLUMN = "Churn"
POSITIVE_LABEL = "Yes"  

# Numeric features.
NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
]

# Handled by its own branch of the ColumnTransformer: a null means "customer never
# billed yet" (tenure == 0) and is replaced by 0 through a deterministic rule, not by
# a learned imputer. The API validation guarantees null can only arrive with tenure == 0.
NULLABLE_NUMERIC_FEATURE = "TotalCharges"

# Categorical features and their allowed values. These lists must match the enums of
# the API contract exactly: the contract rejects anything else with a 422, so the
# encoder should never meet an unknown category in production.
CATEGORICAL_VALUES = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

CATEGORICAL_FEATURES = list(CATEGORICAL_VALUES)

ALL_FEATURES = NUMERIC_FEATURES + [NULLABLE_NUMERIC_FEATURE] + CATEGORICAL_FEATURES

SPLITS = ["train", "val", "test", "eval_frozen", "excluded"]

