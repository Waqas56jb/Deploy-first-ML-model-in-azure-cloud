from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join('model', 'house_price_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

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

        # Prepare data for prediction
        input_data = pd.DataFrame([[size, bedrooms, age]], columns=['size_sqft', 'bedrooms', 'age_years'])
        
        # Make prediction
        prediction = model.predict(input_data)[0]

        return jsonify({'predicted_price': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)