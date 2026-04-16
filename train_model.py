import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
data = pd.read_csv("Artificial_Neural_Network_Case_Study_data.csv")

# Select features and target
X = data.iloc[:, 3:-1].values
y = data.iloc[:, -1].values

# Encode Gender
labelencoder_gender = LabelEncoder()
X[:, 2] = labelencoder_gender.fit_transform(X[:, 2])

# Encode Geography
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(drop="first"), [1])],
    remainder="passthrough"
)
X = np.array(ct.fit_transform(X), dtype=np.float64)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Build model
model = Sequential()
model.add(Dense(units=6, activation="sigmoid", input_dim=X_train.shape[1]))
model.add(Dense(units=6, activation="sigmoid"))
model.add(Dense(units=1, activation="sigmoid"))

# Compile
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train
model.fit(X_train, y_train, batch_size=25, epochs=10)

# Save model and preprocessors
model.save("ann_model.h5")
joblib.dump(labelencoder_gender, "labelencoder_gender.pkl")
joblib.dump(ct, "column_transformer.pkl")
joblib.dump(sc, "scaler.pkl")

print("Model and preprocessing files saved successfully.")
