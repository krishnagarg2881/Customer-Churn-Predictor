import streamlit as st
import pandas as pd
import joblib

# Load model and feature names
model = joblib.load("customer_churn_best_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Customer Churn Prediction")

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")

# ---------- User Inputs ----------
gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["No", "Yes"])
Dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# Create input dictionary
input_data = {
    "SeniorCitizen": SeniorCitizen,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
}

# Convert categorical variables
categorical = {
    "gender_Male": gender == "Male",
    "Partner_Yes": Partner == "Yes",
    "Dependents_Yes": Dependents == "Yes",
    "PhoneService_Yes": PhoneService == "Yes",
    "PaperlessBilling_Yes": PaperlessBilling == "Yes"
}

for col in feature_names:
    if col not in input_data:
        input_data[col] = 0

for key, value in categorical.items():
    if key in input_data:
        input_data[key] = int(value)

# Create DataFrame
input_df = pd.DataFrame([input_data])

# Match training columns
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Prediction
if st.button("Predict"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    st.write(f"**Churn Probability:** {probability:.2%}")