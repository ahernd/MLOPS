import streamlit as st
import requests
import json

# Streamlit Dashboard
st.title("California Housing Price Prediction")

# Create input fields for the user to enter values for the features
MedInc = st.number_input('Median Income (MedInc)', min_value=0.0, max_value=15.0, value=8.32, step=0.01)
HouseAge = st.number_input('House Age (HouseAge)', min_value=1, max_value=100, value=41)
AveRooms = st.number_input('Average Rooms (AveRooms)', min_value=1.0, max_value=100.0, value=6.98, step=0.01)
AveBedrms = st.number_input('Average Bedrooms (AveBedrms)', min_value=0.0, max_value=10.0, value=1.02, step=0.01)
Population = st.number_input('Population', min_value=1, max_value=100000, value=322)
AveOccup = st.number_input('Average Occupancy (AveOccup)', min_value=0.5, max_value=100.0, value=2.56, step=0.01)
Latitude = st.number_input('Latitude', min_value=32.0, max_value=42.0, value=37.88, step=0.01)
Longitude = st.number_input('Longitude', min_value=-125.0, max_value=-114.0, value=-122.23, step=0.01)
Rooms_per_House = st.number_input('Rooms per House (Rooms_per_House)', min_value=0.0, max_value=50.0, value=0.16, step=0.01)

# Once the user has input values, the button will send the request to the Flask API
if st.button('Predict'):
    # The input values to be sent to the API
    input_data = {
        "input": [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, Rooms_per_House]
    }

    # Send the data to the Flask API
    try:
        url = 'http://127.0.0.1:5000/predict'
        response = requests.post(url, json=input_data)
        prediction = response.json().get('prediction')

        # Display the prediction result
        st.success(f"Predicted Housing Price: ${prediction:,.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
