import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained SVM model, scaler, and label encoder
with open('svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Streamlit App
st.set_page_config(
    page_title="Purchase Prediction App",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Add a header with styling
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #4CAF50;">Purchase Prediction App</h1>
        <p style="font-size: 18px; color: #555;">Predict whether a user will make a purchase based on age, gender, and estimated salary.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Input form in a neat layout
st.markdown("---")
st.header("Enter User Details:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    estimated_salary = st.number_input(
        "Estimated Salary (in $)", min_value=10000, max_value=200000, value=50000, step=5000
    )

# Prediction button with styling
if st.button("Predict 🔍"):
    # Encode gender
    gender_encoded = label_encoder.transform([gender])[0]

    # Scale numerical input features
    numerical_features = np.array([[age, estimated_salary]])
    numerical_features_scaled = scaler.transform(numerical_features)

    # Combine scaled numerical features and encoded gender
    input_features_scaled = np.hstack([numerical_features_scaled, [[gender_encoded]]])

    # Make prediction
    prediction = svm_model.predict(input_features_scaled)[0]

    # Display the result with emojis and styling
    st.markdown("---")
    if prediction == 1:
        st.success(
            "🎉 **The model predicts:** Purchased 👍", icon="✅"
        )
    else:
        st.error(
            "❌ **The model predicts:** Not Purchased 👎", icon="❌"
        )

st.markdown(
    """
    <hr style="border: 1px solid #f0f0f0;">
    <div style="text-align: center; color: #888;">
        <p>Developed with ❤️ using Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
