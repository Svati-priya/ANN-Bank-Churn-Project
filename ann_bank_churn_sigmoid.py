import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
data = pd.read_csv("Artificial_Neural_Network_Case_Study_data.csv")
print(data.head())
print(data.shape)
print(data.columns)
X = data.iloc[:, 3:-1].values
y = data.iloc[:, -1].values
labelencoder_gender = LabelEncoder()
X[:, 2] = labelencoder_gender.fit_transform(X[:, 2])
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(drop="first"), [1])],
    remainder="passthrough"
)
X = np.array(ct.fit_transform(X), dtype=np.float64)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
model = Sequential()
model.add(Dense(units=6, activation="sigmoid", input_dim=X_train.shape[1]))
model.add(Dense(units=6, activation="sigmoid"))
model.add(Dense(units=1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=25, epochs=10)
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("Accuracy:")
print(acc)
print("Classification Report:")
print(classification_report(y_test, y_pred))