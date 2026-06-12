import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Customer Churn API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────
# Load Models
# ─────────────────────────────
MODEL_DIR = r"D:\ForNewRepo\Customer Churn\reports\models"

trained_models = {}

for file in os.listdir(MODEL_DIR):
    if file.endswith(".pkl"):
        name = file.replace(".pkl", "")
        with open(os.path.join(MODEL_DIR, file), "rb") as f:
            trained_models[name] = pickle.load(f)


# ─────────────────────────────
# Input Schema
# ─────────────────────────────
class CustomerInput(BaseModel):
    model_name: str

    gender: int            # 0=Female, 1=Male
    SeniorCitizen: int     # 0=No, 1=Yes
    Partner: int           # 0=No, 1=Yes
    Dependents: int        # 0=No, 1=Yes
    tenure: int
    PhoneService: int      # 0=No, 1=Yes

    MultipleLines: str     # "No", "Yes", "No phone service"
    InternetService: str   # "DSL", "Fiber optic", "No"
    OnlineSecurity: str    # "No", "Yes", "No internet service"
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    Contract: int          # 0=Month-to-month, 1=One year, 2=Two year
    PaperlessBilling: int  # 0=No, 1=Yes

    PaymentMethod: str     # "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"

    MonthlyCharges: float
    TotalCharges: float    # Raw value — log1p applied here before prediction


# ─────────────────────────────
# Routes
# ─────────────────────────────
@app.get("/")
def home():
    return {"status": "running"}

@app.get("/models")
def get_models():
    return {"models": list(trained_models.keys())}


@app.post("/predict")
def predict(data: CustomerInput):

    if data.model_name not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")

    model = trained_models[data.model_name]

    try:
        df = pd.DataFrame([data.dict()])
        df = df.drop(columns=["model_name"])

        # Apply log1p to TotalCharges — same transform used during training
        df["TotalCharges"] = np.log1p(df["TotalCharges"])

        pred = model.predict(df)[0]

        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df)[0][1]

        return {
            "model": data.model_name,
            "prediction": int(pred),
            "label": "Churn" if pred == 1 else "No Churn",
            "probability": float(prob) if prob is not None else 0.0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))