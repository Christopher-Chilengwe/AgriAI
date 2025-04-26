# crop_yield/train_yield_model.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset (USDA or FAO data)
df = pd.read_csv('crop_data.csv')  # Columns: [temp, humidity, soil_moisture, NDVI, yield]

# Feature engineering
df['temp_humidity_interaction'] = df['temperature'] * df['humidity']

# Split data with time-series validation
X = df.drop('yield', axis=1)
y = df['yield']
tscv = TimeSeriesSplit(n_splits=5)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}

model = XGBRegressor(objective='reg:squarederror')
grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_absolute_error')
grid_search.fit(X, y)

# Save best model
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'crop_yield_model.pkl')

# Prediction 
new_data = pd.DataFrame([[28, 70, 0.45, 0.82]], columns=['temperature', 'humidity', 'soil_moisture', 'NDVI'])
predicted_yield = best_model.predict(new_data)
print(f"Predicted Yield: {predicted_yield[0]:.2f} kg/ha")