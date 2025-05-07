import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Hardcoded dataset
data = {
    'size_sqft': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    'bedrooms': [3, 3, 3, 4, 2, 3, 4, 4, 3, 3],
    'age_years': [5, 10, 2, 15, 20, 8, 5, 10, 12, 3],
    'price': [200000, 250000, 270000, 300000, 150000, 220000, 350000, 360000, 210000, 280000]
}
df = pd.DataFrame(data)

# Features and target
X = df[['size_sqft', 'bedrooms', 'age_years']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model to pickle file
with open('model/house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as model/house_price_model.pkl")