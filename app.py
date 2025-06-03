import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="Transaction Anomaly Detection", layout="wide")

st.title("🚨 Transaction Anomaly Detection App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("transaction_anomalies_dataset.csv")
    return df

data = load_data()

# Show data info
st.subheader("📊 Dataset Overview")
st.write(data.head())

# Summary Stats
st.subheader("📋 Summary Statistics")
st.write(data.describe())

# Visualizations
st.subheader("📈 Visualizations")

# Histogram
fig1, ax1 = plt.subplots()
sns.histplot(data=data, x='Transaction_Amount', bins=20, ax=ax1)
ax1.set_title('Distribution of Transaction Amount')
st.pyplot(fig1)

# Boxplot
fig2, ax2 = plt.subplots()
sns.boxplot(data=data, x='Account_Type', y='Transaction_Amount', ax=ax2)
ax2.set_title('Transaction Amount by Account Type')
st.pyplot(fig2)

# Scatterplot
fig3, ax3 = plt.subplots()
sns.scatterplot(data=data, x='Age', y='Average_Transaction_Amount', hue='Account_Type', ax=ax3)
ax3.set_title('Average Transaction Amount vs. Age')
st.pyplot(fig3)

# Day of week
fig4, ax4 = plt.subplots()
sns.histplot(data=data, x='Day_of_Week', ax=ax4)
ax4.set_title('Count of Transactions by Day of the Week')
st.pyplot(fig4)

# Correlation
fig5, ax5 = plt.subplots()
numerical_features = data.select_dtypes(include=["int64", "float64"])
sns.heatmap(numerical_features.corr(), cmap="coolwarm", ax=ax5)
ax5.set_title('Correlation Heatmap')
st.pyplot(fig5)

# Anomaly Detection
mean_amount = data['Transaction_Amount'].mean()
std_amount = data['Transaction_Amount'].std()
anomaly_threshold = mean_amount + 2 * std_amount
data['Is_Anomaly'] = data['Transaction_Amount'] > anomaly_threshold

# Anomaly Plot
st.subheader("🚩 Anomaly Flagging")
fig6, ax6 = plt.subplots()
sns.scatterplot(data=data, x='Transaction_Amount', y='Average_Transaction_Amount', hue='Is_Anomaly', palette='coolwarm', ax=ax6)
ax6.set_title('Anomalies in Transaction Amount')
st.pyplot(fig6)

# Anomaly stats
num_anomalies = data['Is_Anomaly'].sum()
total_instances = data.shape[0]
anomaly_ratio = num_anomalies / total_instances
st.write(f"**Anomaly Ratio:** {anomaly_ratio:.2%} ({num_anomalies}/{total_instances})")

# ML Model
st.subheader("🤖 Isolation Forest Model")

# Train/test split
features = ['Transaction_Amount','Average_Transaction_Amount','Frequency_of_Transactions']
X = data[features]
y = data['Is_Anomaly']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = IsolationForest(contamination=0.02, random_state=42)
model.fit(X_train)

# Prediction form
st.subheader("📥 Predict New Transaction")

user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"Enter value for {feature}:", min_value=0.0, value=10.0)

if st.button("Detect Anomaly"):
    user_df = pd.DataFrame([user_input])
    user_pred = model.predict(user_df)
    is_anomaly = 1 if user_pred[0] == -1 else 0
    if is_anomaly:
        st.error("🚨 Anomaly Detected: This transaction is flagged as **anomalous**.")
    else:
        st.success("✅ No Anomaly: This transaction appears **normal**.")