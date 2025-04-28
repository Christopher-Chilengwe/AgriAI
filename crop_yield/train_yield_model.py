# crop_yield/train_yield_model.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
import joblib
import shap
import json
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DATA_PATH = 'crop_data.csv'
MODEL_PATH = 'crop_yield_model.pkl'
METADATA_PATH = 'model_metadata.json'
FEATURES_PATH = 'feature_importances.csv'

def load_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess dataset"""
    logging.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Convert to datetime if available
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
    
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering"""
    logging.info("Creating features")
    
    # Interaction features
    df['temp_humidity_interaction'] = df['temp'] * df['humidity']
    df['temp_ndvi_interaction'] = df['temp'] * df['NDVI']
    
    # Polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(df[['temp', 'humidity', 'soil_moisture']])
    poly_columns = poly.get_feature_names_out(['temp', 'humidity', 'soil_moisture'])
    df = pd.concat([df, pd.DataFrame(poly_features, columns=poly_columns)], axis=1)
    
    # Temporal features
    if 'date' in df.columns:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
    
    return df

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """Train model with hyperparameter tuning"""
    logging.info("Starting model training")
    
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.001, 0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    model = XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        early_stopping_rounds=50,
        random_state=42
    )
    
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        model,
        param_dist,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_iter=50,
        random_state=42,
        verbose=2
    )
    
    search.fit(X_train, y_train, eval_set=[(X_train, y_train)])
    
    logging.info(f"Best parameters: {search.best_params_}")
    return search.best_estimator_

def evaluate_model(model, X: pd.DataFrame, y: pd.Series, split: str) -> dict:
    """Evaluate model performance"""
    preds = model.predict(X)
    return {
        'split': split,
        'mae': mean_absolute_error(y, preds),
        'rmse': np.sqrt(mean_squared_error(y, preds)),
        'r2': r2_score(y, preds),
        'n_samples': len(y)
    }

def save_metadata(model, features: list, metrics: dict) -> None:
    """Save model metadata and metrics"""
    metadata = {
        'training_date': datetime.now().isoformat(),
        'features': features,
        'best_params': model.get_params(),
        'metrics': metrics
    }
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

def main():
    try:
        # Load and prepare data
        df = load_data(DATA_PATH)
        df = create_features(df)
        
        # Split data
        X = df.drop(['yield', 'date'] if 'date' in df.columns else 'yield', axis=1)
        y = df['yield']
        
        # Time-based train-test split
        split_idx = int(0.8 * len(df))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate
        train_metrics = evaluate_model(model, X_train, y_train, 'train')
        test_metrics = evaluate_model(model, X_test, y_test, 'test')
        
        logging.info(f"\nTraining Metrics:\n{pd.DataFrame([train_metrics])}")
        logging.info(f"\nTest Metrics:\n{pd.DataFrame([test_metrics])}")
        
        # Save artifacts
        joblib.dump(model, MODEL_PATH)
        save_metadata(model, list(X.columns), {'train': train_metrics, 'test': test_metrics})
        
        # Feature importance
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        importances.to_csv(FEATURES_PATH, index=False)
        
        # SHAP explainability
        explainer = shap.Explainer(model)
        shap_values = explainer(X_train)
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        plt.savefig('feature_shap_summary.png', bbox_inches='tight')
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

# prediction (in separate inference script)
def predict_yield(input_data: dict) -> float:
    """Predict yield from input data"""
    model = joblib.load(MODEL_PATH)
    features = joblib.load('feature_columns.pkl')
    
    # Create DataFrame and generate features
    input_df = pd.DataFrame([input_data])
    input_df = create_features(input_df)
    
    # Ensure correct feature order
    input_df = input_df.reindex(columns=features, fill_value=0)
    
    return model.predict(input_df)[0]
