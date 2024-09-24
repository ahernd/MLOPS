# Import necessary libraries
import pandas as pd
import joblib
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('california_housing_model.pkl')

# Load the feature names
# Assuming the feature names are the same as in training
feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                 'Population', 'AveOccup', 'Latitude', 'Longitude', 'Rooms_per_House']

# Define root route for testing purposes
@app.route('/')
def index():
    return "Flask app is running. Use the /predict endpoint for predictions."

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction based on input JSON data.
    Expected JSON format:
    {
        "input": [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, Rooms_per_House]
    }
    """
    try:
        data = request.get_json(force=True)
        input_data = data.get('input')
        
        if not input_data:
            return jsonify({'error': 'No input data provided.'}), 400
        
        if len(input_data) != len(feature_names):
            return jsonify({'error': f'Expected {len(feature_names)} features, got {len(input_data)}.'}), 400
        
        # Create DataFrame for input
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Return prediction as JSON
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)