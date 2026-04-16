import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model

app = FastAPI(
    title="ANN Bank Churn Prediction API",
    description="Predicts whether a bank customer is likely to exit using an Artificial Neural Network.",
    version="1.0.0"
)

# Load saved model and preprocessing objects
model = load_model("ann_model.h5")
labelencoder_gender = joblib.load("labelencoder_gender.pkl")
column_transformer = joblib.load("column_transformer.pkl")
scaler = joblib.load("scaler.pkl")


class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


@app.get("/")
def home():
    return {"message": "ANN Bank Churn Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: CustomerData):
    sample = np.array([[
        data.CreditScore,
        data.Geography,
        data.Gender,
        data.Age,
        data.Tenure,
        data.Balance,
        data.NumOfProducts,
        data.HasCrCard,
        data.IsActiveMember,
        data.EstimatedSalary
    ]], dtype=object)

    # Encode Gender
    sample[:, 2] = labelencoder_gender.transform(sample[:, 2])

    # Apply one-hot encoding
    sample = np.array(column_transformer.transform(sample), dtype=np.float64)

    # Scale
    sample = scaler.transform(sample)

    # Predict
    probability = float(model.predict(sample)[0][0])
    prediction = 1 if probability > 0.5 else 0

    return {
        "prediction": prediction,
        "churn_probability": round(probability, 4),
        "result": "Customer likely to exit" if prediction == 1 else "Customer likely to stay"
    }