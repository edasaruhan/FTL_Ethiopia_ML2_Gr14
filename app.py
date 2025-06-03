import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Transaction Anomaly Detection", layout="centered")
st.title("🚨 Transaction Anomaly Detection")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("transaction_anomalies_dataset.csv")

data = load_data()

# Prepare data for model training
features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']
X = data[features]
y = data['Transaction_Amount'] > (data['Transaction_Amount'].mean() + 2 * data['Transaction_Amount'].std())  # crude anomaly flag
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Isolation Forest
model = IsolationForest(contamination=0.02, random_state=42)
model.fit(X_train)

# UI for user prediction
st.subheader("📥 Predict New Transaction")

user_input = {}
for feature in features:
    user_input[feature] = st.number_input(
        f"Enter value for {feature}:",
        min_value=0.0,
        value=10.0,
        step=1.0,
        format="%.2f"
    )

if st.button("Detect Anomaly"):
    user_df = pd.DataFrame([user_input])
    user_pred = model.predict(user_df)
    is_anomaly = 1 if user_pred[0] == -1 else 0
    if is_anomaly:
        st.error("🚨 Anomaly Detected: This transaction is flagged as **anomalous**.")
    else:
        st.success("✅ No Anomaly: This transaction appears **normal**.")
