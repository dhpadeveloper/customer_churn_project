from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib


app = FastAPI(
    title="Customer Churn Prediction API"
)


model = joblib.load(
    "model/churn_model.pkl"
)


class Customer(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str

    tenure: int

    PhoneService: str
    MultipleLines: str
    InternetService: str

    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str

    StreamingTV: str
    StreamingMovies: str

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def predict(customer: Customer):

    try:

        data = customer.model_dump()

        df = pd.DataFrame([data])

        # Feature engineering
        df["TenureGroup"] = df["tenure"].apply(tenure_group)

        service_columns = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies"
        ]

        df["ServiceCount"] = (
            df[service_columns]
            .eq("Yes")
            .sum(axis=1)
        )

        prediction = model.predict(df)[0]

        probability = model.predict_proba(df)[0][1]

        return {
            "prediction": (
                "Yes"
                if prediction == 1
                else "No"
            ),
            "churn_probability": round(
                float(probability),
                2
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )    


def tenure_group(tenure):
    if tenure <= 12:
        return "New"
    elif tenure <= 36:
        return "Medium"
    else:
        return "Long-term"