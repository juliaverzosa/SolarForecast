import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib  # To save the scaler objects for later use in the app
import os
import geopandas as gpd
from shapely.geometry import Point
import math

# --- 1. Load the Data from Feature Engineering Output ---
print("Loading processed data from feature engineering...")

# Load the modeling dataset created by feature_engineering.py
try:
    modeling_df = pd.read_csv('../data/processed/modeling_dataset.csv')
    print(f"Modeling dataset shape: {modeling_df.shape}")
except FileNotFoundError:
    print("Modeling dataset not found. Running feature engineering first...")
    # You might want to import and run your feature engineering script here
    # from feature_engineering import main
    # modeling_df = main()
    # For now, we'll exit
    exit(1)

# --- 2. Select Features for Modeling ---
# Based on your feature engineering, these are the available features
print("Available features in modeling dataset:")
print(modeling_df.columns.tolist())

# Define which features to use for modeling
# Adjust this based on your specific modeling goals
feature_columns = [
    'month_sin', 'month_cos', 'season', 
    'ghi', 'temperature', 'humidity', 'clearness_index',
    'sunshine_hours', 'clear_sky_ratio',
    'rooftop_area_sq_m', 'orientation_score', 'tilt_factor', 
    'shading_factor', 'SEI',
    'household_size', 'has_aircon', 'has_water_heater',
    'daytime_load_percentage', 'high_consumption'
]

# Target variable (adjust based on your modeling goal)
target_column = 'average_monthly_consumption_kwh'  # Or another target like 'SEI'

# Check if all selected features exist in the dataset
missing_features = [col for col in feature_columns if col not in modeling_df.columns]
if missing_features:
    print(f"Warning: The following features are missing: {missing_features}")
    # Remove missing features from the list
    feature_columns = [col for col in feature_columns if col not in missing_features]

# --- 3. Handle Missing Values ---
print("Handling missing values...")

# Separate features and target
X = modeling_df[feature_columns].copy()
y = modeling_df[target_column]

# Check for missing values
missing_values = X.isnull().sum()
if missing_values.any():
    print("Missing values found:")
    print(missing_values[missing_values > 0])
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
else:
    print("No missing values found.")
    X_imputed = X.copy()

# --- 4. Normalize/Scale the Features ---
print("Normalizing features...")

# Normalize/Scale the features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

# Save the scaler for later use on new data
os.makedirs('../models', exist_ok=True)
joblib.dump(scaler, '../models/feature_scaler.joblib')
print("Feature scaler saved to '../models/feature_scaler.joblib'")

# --- 5. Prepare Final Dataset ---
print("Preparing final dataset...")

# Combine scaled features with target
final_dataset = X_scaled.copy()
final_dataset[target_column] = y.values

# Add identifiers if needed
final_dataset['building_id'] = modeling_df['building_id'].values
final_dataset['household_id'] = modeling_df['household_id'].values
final_dataset['month'] = modeling_df['month'].values

# --- 6. Save the Processed Dataset to CSV ---
os.makedirs('../data/processed', exist_ok=True)
final_dataset.to_csv('../data/processed/final_training_dataset.csv', index=False)
print("Final dataset saved to '../data/processed/final_training_dataset.csv'")
print(f"Final dataset shape: {final_dataset.shape}")

# --- 7. Additional Analysis (Optional) ---
print("\nAdditional analysis:")
print(f"Number of features: {len(feature_columns)}")
print(f"Target variable: {target_column}")
print(f"Target statistics:")
print(f"  Mean: {y.mean():.2f}")
print(f"  Std: {y.std():.2f}")
print(f"  Min: {y.min():.2f}")
print(f"  Max: {y.max():.2f}")

# Correlation with target (if target is numeric)
if pd.api.types.is_numeric_dtype(y):
    correlations = []
    for col in feature_columns:
        if col in X_scaled.columns:
            corr = np.corrcoef(X_scaled[col], y)[0, 1]
            correlations.append((col, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\nTop features correlated with target:")
    for col, corr in correlations[:10]:
        print(f"  {col}: {corr:.3f}")