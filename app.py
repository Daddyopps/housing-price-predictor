import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load trained model
model = tf.keras.models.load_model("housing_price_model.h5")

# Define input form
st.title("California House Price Predictor")

st.write("Enter the details to predict house price:")

longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", value=30)
total_rooms = st.number_input("Total Rooms", value=2000)
total_bedrooms = st.number_input("Total Bedrooms", value=400)
population = st.number_input("Population", value=1000)
households = st.number_input("Households", value=300)
median_income = st.number_input("Median Income", value=3.5)

# Categorical input
ocean_proximity = st.selectbox("Ocean Proximity", ["INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN", "<1H OCEAN"])

# One-hot encoding for ocean_proximity
proximity_options = ['INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN', '<1H OCEAN']
proximity_encoded = [1 if ocean_proximity == option else 0 for option in proximity_options[1:]]

# Combine all inputs
input_data = np.array([[
    longitude, latitude, housing_median_age, total_rooms,
    total_bedrooms, population, households, median_income
] + proximity_encoded])

# Scale input data using same scaler used in training
scaler = StandardScaler()
# Note: You should fit the scaler using training data and save it, then load it here
# Example: scaler = joblib.load("scaler.pkl")
# For demo, we use scaler.fit_transform again (not ideal for production)

# Dummy scaling step (replace with saved one)
scaled_input = scaler.fit_transform(np.vstack([input_data, input_data]))[0].reshape(1, -1)

if st.button("Predict"):
    prediction = model.predict(scaled_input)
    price = np.expm1(prediction)[0][0]  # Reverse log1p transform
    st.success(f"Estimated House Price: ${price:,.2f}")
