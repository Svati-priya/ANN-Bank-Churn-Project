## 🚀 Live Demo
👉 https://ann-bank-churn-project-2dv97knytfurytdwnmzvpw.streamlit.app

# ANN Bank Churn Prediction App
## Problem Statement
The goal of this project is to predict whether a bank customer is likely to leave the bank or not using an Artificial Neural Network (ANN).

## What I Did
- Used the bank churn dataset
- Performed preprocessing:
  - Label Encoding for Gender
  - One Hot Encoding for Geography
  - Standard Scaling
- Built an ANN model using TensorFlow/Keras
- Used sigmoid activation
- Used batch size = 25 and epochs = 10

## Solution
I converted the trained ANN model into an interactive Streamlit app where users can enter customer details and get churn prediction and churn probability.

## 🛠 Tech Stack
- Python
- Streamlit
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Joblib

## 📊 Model Performance
- Accuracy: 81.95%
- Loss: 0.41
- Epochs: 10
- Batch Size: 25
- Activation Function: Sigmoid
- Optimizer: Adam

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
