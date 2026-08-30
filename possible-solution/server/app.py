# Create a base Flask server

import pickle
import os
from flask import Flask, request, jsonify

app = Flask(__name__)
BASE_DIR = os.path.dirname(__file__)

# Enable cors
@app.after_request
def after_request(response):
    """
    Enable CORS
    """
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response


# Load model from pickle file
model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))

# Model takes two parameters - day of week and airport id, then returns a prediction of flight delay
@app.route('/predict', methods=['GET'])
def predict():
    """
    Takes two parameters - day of week and airport id, then returns a prediction of flight delay
    """
    # Store day_of_week as int
    day_of_week = int(request.args.get('day_of_week'))
    airport_id = int(request.args.get('airport_id'))
    prediction = model.predict_proba([[day_of_week, airport_id]])[0]

    # Normalize prediction to a string and handle multiple formats (bytes, strings, lists)
    if isinstance(prediction, (list, tuple)):
        # If it's a list/tuple of values, try to parse directly
        try:
            certainty = float(prediction[0])
            delay = float(prediction[1])
        except Exception:
            pred_str = ' '.join(map(str, prediction))
            pred_str = pred_str.strip().strip("'\"")
            parts = pred_str.split()
            certainty = float(parts[0])
            delay = float(parts[1])
    else:
        if isinstance(prediction, (bytes, bytearray)):
            pred_str = prediction.decode('utf-8', errors='ignore')
        else:
            pred_str = str(prediction)
        pred_str = pred_str.strip().strip("'\"")
        parts = pred_str.split()
        certainty = float(parts[0])
        delay = float(parts[1])

    # return prediction as json
    return jsonify({'certainty': certainty, 'delay': delay})

# Create a new route called airports with method of get
@app.route('/airports', methods=['GET'])
def airports():
    # Load airports from csv file
        airports = open(os.path.join(BASE_DIR, 'airports.csv'), 'r').readlines()

    # Remove first line of airports
    airports.pop(0)

    # Create list with dictionary of airports
    # First value is id, second is name
    # Convert id to integer
    # Remove last character from name
    airports = [{'id': int(airport.split(',')[0]), 'name': airport.split(',')[1][:-1]} for airport in airports]
    # Sort by name
    airports = sorted(airports, key=lambda k: k['name'])

    return jsonify(airports)

if __name__ == '__main__':
    app.run(debug=True)