# using streamlit
import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model from the pickle file
@st.cache_resource
def load_model():
    if os.path.exists("california_housing_model.pkl"):
        model = joblib.load('california_housing_model.pkl')
        return model
    else:
        st.error("Model file not found. Please ensure the model file is available.")
        return None

# Streamlit Dashboard
st.title("California Housing Price Prediction")

# Load the model
model = load_model()
# additional comment

if model:
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

    # Add a button for prediction
    if st.button('Predict'):
        # The input values to be used for the prediction
        input_data = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, Rooms_per_House]],
                                  columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'Rooms_per_House'])

        # Perform the prediction
        prediction = model.predict(input_data)[0]

        # Display the prediction result
        st.success(f"Predicted Housing Price: ${prediction:,.2f}")

    # Display a radio button for the next action
    next_action = st.radio("Choose your next action:", ('Make Another Prediction', 'Exit Application'))

    if next_action == 'Make Another Prediction':
        st.write("You can now enter new values to make another prediction.")

    elif next_action == 'Exit Application':
        st.info("Thank you for using the application! To stop the app, use 'Ctrl+C' in the terminal or close the browser window.")
