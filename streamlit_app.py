import joblib
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="ANN Bank Churn Prediction", page_icon="📊")

st.title("ANN Bank Churn Prediction")
st.write("Predict whether a bank customer is likely to exit using an Artificial Neural Network.")

# Load model and preprocessors
model = load_model("ann_model.h5")
labelencoder_gender = joblib.load("labelencoder_gender.pkl")
column_transformer = joblib.load("column_transformer.pkl")
scaler = joblib.load("scaler.pkl")

st.subheader("Enter Customer Details")

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=40)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, value=60000.0)
num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

if st.button("Predict"):
    sample = np.array([[
        credit_score,
        geography,
        gender,
        age,
        tenure,
        balance,
        num_products,
        has_card,
        is_active,
        salary
    ]], dtype=object)

    sample[:, 2] = labelencoder_gender.transform(sample[:, 2])
    sample = np.array(column_transformer.transform(sample), dtype=np.float64)
    sample = scaler.transform(sample)

    probability = float(model.predict(sample)[0][0])
    prediction = 1 if probability > 0.5 else 0

    st.subheader("Prediction Result")
    st.write(f"**Churn Probability:** {probability:.4f}")
    st.write(f"**Prediction:** {'Customer likely to exit' if prediction == 1 else 'Customer likely to stay'}")