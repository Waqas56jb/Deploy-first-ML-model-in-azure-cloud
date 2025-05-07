import json
import pickle
import pandas as pd
import os

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR', ''), 'house_price_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        input_df = pd.DataFrame([data], columns=['size_sqft', 'bedrooms', 'age_years'])
        prediction = model.predict(input_df)
        return json.dumps({'predicted_price': float(prediction[0])})
    except Exception as e:
        return json.dumps({'error': str(e)})