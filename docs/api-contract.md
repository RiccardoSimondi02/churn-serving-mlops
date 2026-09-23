# API Contract

## Endpoint
POST /predict

## Request schema
| Field             | Type    | Required | Constraints                                                                   |
|-------------------|---------|----------|-------------------------------------------------------------------------------|
| customer_id       | string       | yes      |                                                                          |
| gender            | string       | yes      | one of: ["Female", "Male"]                                               |
| SeniorCitizen     | int          | yes      | one of: [0, 1]                                                           |
| Partner           | string       | yes      | one of: ["No", "Yes"]                                                    |
| Dependents        | string       | yes      | one of: ["No", "Yes"]                                                    |
| tenure            | int          | yes      | >= 0                                                                     |
| PhoneService      | string       | yes      | one of: ["No", "Yes"]                                                    |
| MultipleLines     | string       | yes      | one of: ["No", "No phone service", "Yes"]                                |
| InternetService   | string       | yes      | one of: ["DSL", "Fiber optic", "No"]                                     |
| OnlineSecurity    | string       | yes      | one of: ["No", "No internet service", "Yes"]                             |
| OnlineBackup      | string       | yes      | one of: ["No", "No internet service", "Yes"]                             |
| DeviceProtection  | string       | yes      | one of: ["No", "No internet service", "Yes"]                             |
| TechSupport       | string       | yes      | one of: ["No", "No internet service", "Yes"]                             |
| StreamingTV       | string       | yes      | one of: ["No", "No internet service", "Yes"]                             |
| StreamingMovies   | string       | yes      | one of: ["No", "No internet service", "Yes"]                             |
| Contract          | string       | yes      | one of: ["Month-to-month", "One year", "Two year"]                       |
| PaperlessBilling  | string       | yes      | one of: ["No", "Yes"]                                                    |
| PaymentMethod     | string       | yes      | one of: ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"] |
| MonthlyCharges    | float        | yes      | >= 0                                                                     |
| TotalCharges      | float|null   | yes      | >= 0 or null                                                             |

## Validation layer (rejects with 422)
- Missing required field
- Wrong type (e.g. tenure as string)
- category not in allowed set
- TotalCharges == null -> valid if tenure == 0 (first month), else error

## Pipeline responsibility (accepted, handled internally)
- TotalCharges == null -> set it to 0


## Response schema
| Field        | Type   | Description                              |
|--------------|--------|------------------------------------------|
| prediction   | string | one of: ["churn", "no_churn"]            |
| threshold    | float  | threshold applied for prediction         |
| probability  | float  | probability of the positive class, 0-1   |
| model_version| int    | registry version number of the model used|
| request_id   | string | identifier of the request                |
| customer_id  | string | identifier of the customer               |

## Error responses
| Code | Meaning                                                              |
|------|----------------------------------------------------------------------|
| 422  | validation failed (see above), with a body naming the offending field |
| 500  | inference failed, or the prediction could not be recorded             |
| 503  | service not ready: the model is not loaded                            |


## Health checks

### GET /healthz (liveness)
Returns 200 if the process is running, regardless of the model's state.
It does not verify whether the model is loaded.

### GET /readyz (readiness)
It returns a 200 response only if the model has been successfully loaded into memory and is ready to serve requests at /predict.
It returns a 503 response if the model is not yet loaded (e.g., during startup) or if loading failed.