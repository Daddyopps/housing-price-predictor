
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model("housing_price_model.h5")
scaler = joblib.load("scaler.pkl")

# Page title
st.title("California House Price Predictor")

# User input
longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", value=30)
total_rooms = st.number_input("Total Rooms", value=2000)
total_bedrooms = st.number_input("Total Bedrooms", value=400)
population = st.number_input("Population", value=1000)
households = st.number_input("Households", value=300)
median_income = st.number_input("Median Income", value=3.5)

ocean_proximity = st.selectbox("Ocean Proximity", ["INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN", "<1H OCEAN"])
proximity_encoded = [1 if ocean_proximity == option else 0 for option in ["ISLAND", "NEAR BAY", "NEAR OCEAN", "<1H OCEAN"]]

# Combine all inputs
input_data = np.array([[
    longitude, latitude, housing_median_age, total_rooms,
    total_bedrooms, population, households, median_income
] + proximity_encoded])

# Scale input
scaled_input = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(scaled_input)
    price = np.expm1(prediction)[0][0]
    st.success(f"Estimated House Price: ${price:,.2f}")
