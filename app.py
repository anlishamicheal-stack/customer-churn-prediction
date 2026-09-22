import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# -------------------------------
# Helper for interactive SHAP plots
# -------------------------------
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# -------------------------------
# Load your trained XGBoost model
# -------------------------------
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("D:/churn_project/models/xgb_churn.json")

# -------------------------------
# Load and preprocess training dataset
# -------------------------------
data = pd.read_csv("D:/churn_project/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean and encode
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data = data.dropna()
data = data.drop("customerID", axis=1)
data = pd.get_dummies(data, drop_first=True)

X = data.drop("Churn_Yes", axis=1)
y = data["Churn_Yes"]

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Customer Churn Prediction Dashboard")
st.write("This app uses an XGBoost model to predict customer churn.")

st.subheader("Dataset Preview")
st.write(data.head())

# -------------------------------
# Prediction on demo dataset
# -------------------------------
st.subheader("Make a Prediction")
row_index = st.number_input("Pick a row index", min_value=0, max_value=len(X)-1, value=0)
sample = X.iloc[[row_index]]
prediction = xgb_model.predict(sample)
st.write("Prediction:", "Churn" if prediction[0] == 1 else "No Churn")

# -------------------------------
# SHAP Explanation (global + per-customer)
# -------------------------------
st.subheader("Model Explanation (SHAP)")

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)

# Global summary plot
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
st.pyplot(fig)

# Per-customer force plot
st.write("Detailed explanation for selected customer:")
st_shap(shap.force_plot(explainer.expected_value, shap_values[row_index,:], X.iloc[row_index,:]))

# -------------------------------
# File Uploader for new data
# -------------------------------
st.subheader("Upload New Customer Data")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)

    st.write("Uploaded Data Preview")
    st.write(new_data.head())

    # Preprocess uploaded data
    if "customerID" in new_data.columns:
        new_data = new_data.drop("customerID", axis=1)
    if "TotalCharges" in new_data.columns:
        new_data["TotalCharges"] = pd.to_numeric(new_data["TotalCharges"], errors="coerce")
        new_data = new_data.dropna()
    new_data = pd.get_dummies(new_data, drop_first=True)

    # Align columns with training features
    new_data = new_data.reindex(columns=X.columns, fill_value=0)

    # Predict
    predictions = xgb_model.predict(new_data)
    st.write("Predictions:", ["Churn" if p == 1 else "No Churn" for p in predictions])
