# Import necessary libraries
from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS

# Create a Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  


# Load the trained model
model = joblib.load('/Users/malithlekamge/Documents/ATLINKANGULAR/AUTH/flsk/heart_disease_model.pkl')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        input_data = request.json['data']
        
        # Perform the same preprocessing as before
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        # Make a prediction
        prediction = model.predict(input_data_reshaped)
        
        # Return the prediction result
        if prediction[0] == 0:
            result = '0'
        else:
            result = '1'
            
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

