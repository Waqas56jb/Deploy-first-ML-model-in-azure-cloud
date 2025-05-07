from flask import Flask, render_template, request, jsonify
import requests
import json

app = Flask(__name__)

# Azure ML endpoint details (replace with your actual endpoint and key)
AZURE_ENDPOINT = "YOUR_SCORING_URI"  # e.g., http://<guid>.eastus.inference.ml.azure.com/score
AZURE_API_KEY = "YOUR_PRIMARY_KEY"   # Primary Key from Azure ML endpoint

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        size = float(request.form['size'])
        bedrooms = int(request.form['bedrooms'])
        age = int(request.form['age'])

        # Prepare data for Azure ML endpoint
        data = {
            "size_sqft": size,
            "bedrooms": bedrooms,
            "age_years": age
        }

        # Send request to Azure ML endpoint
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {AZURE_API_KEY}'
        }
        response = requests.post(AZURE_ENDPOINT, json=data, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes

        # Parse response
        result = response.json()
        if 'predicted_price' in result:
            return jsonify({'predicted_price': result['predicted_price']})
        else:
            return jsonify({'error': result.get('error', 'Unknown error')}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)