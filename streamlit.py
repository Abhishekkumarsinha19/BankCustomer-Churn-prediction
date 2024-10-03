import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Load the model
st.title("Customer Churn Prediction")
loaded_model = joblib.load(r'C:\Users\hp\ML file\Bank Customer Churn Prediction\best_churn_model.pkl')

# Input features
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure (in years)", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0, value=1000)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=5, value=1)
has_crcard = st.selectbox("Has Credit Card?", [1, 0])
is_active_member = st.selectbox("Is Active Member?", [1, 0])
estimated_salary = st.number_input("Estimated Salary", min_value=0, value=50000)

# Example of missing feature: Customer ID or other missing features
# You can either add a fixed value or collect it from user input if necessary
missing_feature = st.number_input("Missing Feature (e.g. Customer ID or a related feature)", value=1)

# Preprocessing inputs
geography = 0 if geography == 'France' else (1 if geography == 'Germany' else 2)
gender = 0 if gender == 'Female' else 1

# Create input_data array for prediction
input_data = np.array([[credit_score, geography, gender, age, tenure, balance, num_of_products, has_crcard, is_active_member, estimated_salary, missing_feature]])

# Prediction function
def prediction_best_churn_model(input_data):
    prediction = loaded_model.predict(input_data)
    return prediction

# Prediction button
if st.button("Predict"):
    prediction = prediction_best_churn_model(input_data)
    if prediction[0] == 1:  # Access the first element of prediction array
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is not likely to churn.")
