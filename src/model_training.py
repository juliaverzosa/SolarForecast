# thesis_model_training.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
import joblib
import os

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# -------------------------------------------------------------------
# 1. LOAD THE PROCESSED FEATURE ENGINEERING DATASET
# -------------------------------------------------------------------
print("[THESIS] Loading processed modeling dataset...")
df = pd.read_csv('../data/processed/modeling_dataset.csv')

# Display basic info
print(f"Dataset Shape: {df.shape}")

# -------------------------------------------------------------------
# 2. DEFINE TARGET AND FEATURES (ALIGNED WITH THESIS OBJECTIVES)
# -------------------------------------------------------------------
target_column = 'average_monthly_consumption_kwh'

# Select feature columns (exclude identifiers and the target)
exclude_columns = ['building_id', 'household_id', 'month', target_column]
feature_columns = [col for col in df.columns if col not in exclude_columns]

X = df[feature_columns]
y = df[target_column]

print(f"\nTarget variable: '{target_column}'")
print(f"Number of features: {len(feature_columns)}")

# -------------------------------------------------------------------
# 3. TRAIN-TEST SPLIT (80-20 split)
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nData Split:")
print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# -------------------------------------------------------------------
# 4. CHECK AND HANDLE INFINITE/LARGE VALUES
# -------------------------------------------------------------------
print("Checking for infinite or extremely large values...")

# Function to check for inf and large values
def check_data_issues(df, name):
    print(f"\nChecking {name}:")
    # Check for infinite values
    inf_mask = np.isinf(df).any()
    if inf_mask.any():
        inf_cols = df.columns[inf_mask].tolist()
        print(f"  WARNING: Infinite values found in columns: {inf_cols}")
        for col in inf_cols:
            print(f"    {col}: {np.sum(np.isinf(df[col]))} infinite values")
    
    # Check for very large values (absolute value > 1e10)
    large_mask = (np.abs(df) > 1e10).any()
    if large_mask.any():
        large_cols = df.columns[large_mask].tolist()
        print(f"  WARNING: Very large values found in columns: {large_cols}")
        for col in large_cols:
            print(f"    {col}: max value = {df[col].max()}")
    
    # Check for NaN values
    nan_mask = df.isna().any()
    if nan_mask.any():
        nan_cols = df.columns[nan_mask].tolist()
        print(f"  WARNING: NaN values found in columns: {nan_cols}")
        for col in nan_cols:
            print(f"    {col}: {df[col].isna().sum()} NaN values")

# Check both training and testing data
check_data_issues(X_train, "X_train")
check_data_issues(pd.DataFrame(y_train), "y_train")

# Handle infinite values by replacing with NaN and then imputing
print("\nHandling data issues...")

# Replace infinite values with NaN
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Impute missing values (including the ones we just created from inf)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Check if y has any issues
y_train = y_train.replace([np.inf, -np.inf], np.nan)
if y_train.isna().any():
    y_imputer = SimpleImputer(strategy='mean')
    y_train = pd.Series(y_imputer.fit_transform(y_train.values.reshape(-1, 1)).flatten())
    y_test = y_test.replace([np.inf, -np.inf], np.nan)
    y_test = pd.Series(y_imputer.transform(y_test.values.reshape(-1, 1)).flatten())

print("Data issues handled. Proceeding with scaling...")

# -------------------------------------------------------------------
# 5. SCALE FEATURES
# -------------------------------------------------------------------
# Important for AdaBoost and crucial for many other algorithms.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Save the scaler for the prototype
os.makedirs('../models', exist_ok=True)
joblib.dump(scaler, '../models/scaler.joblib')
print("Feature scaler saved to '../models/scaler.joblib'.")

# -------------------------------------------------------------------
# 6. DEFINE THE MODELS (ALIGNED WITH THESIS TITLE)
# -------------------------------------------------------------------
# Initialize the two core ensemble models for your thesis
models = {
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# -------------------------------------------------------------------
# 7. HYPERPARAMETER TUNING (Specific Objective #2 & Methodology)
# -------------------------------------------------------------------
print("\n[THESIS] Performing Hyperparameter Tuning...")

# Parameter grids for tuning
param_grids = {
    'AdaBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
}

# Dictionary to store the best tuned models
tuned_models = {}

for name in models:
    print(f"  Tuning {name}...")
    grid_search = GridSearchCV(
        estimator=models[name],
        param_grid=param_grids[name],
        cv=5,  # 5-fold cross-validation
        scoring='neg_root_mean_squared_error',  # We want to minimize RMSE
        n_jobs=-1,  # Use all CPU cores
        verbose=1  # Shows progress
    )
    grid_search.fit(X_train_scaled, y_train)
    
    # Save the best model from the search
    tuned_models[name] = grid_search.best_estimator_
    
    print(f"    Best Parameters for {name}: {grid_search.best_params_}")
    print(f"    Best CV Score (Negative RMSE): {grid_search.best_score_:.4f}\n")

# -------------------------------------------------------------------
# 8. MODEL EVALUATION (Specific Objective #3)
# -------------------------------------------------------------------
print("[THESIS] Evaluating Model Performance on Test Set...")
results = {}

for name, model in tuned_models.items():
    # Predict on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Calculate Key Metrics (RMSE, MAE, R²)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}
    
    # Print results
    print(f"{name:10} | RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# -------------------------------------------------------------------
# 9. VISUALIZATION & ANALYSIS
# -------------------------------------------------------------------
print("\n[THESIS] Generating performance visualizations...")

# 9.1 Create a DataFrame for results and sort by RMSE
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df = results_df.sort_values('RMSE')
print("\nFinal Model Ranking (Lower RMSE is better):")
print(results_df)

# 9.2 Plot Model Comparison
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# RMSE and MAE plot
results_df[['RMSE', 'MAE']].plot(kind='bar', ax=ax[0])
ax[0].set_title('Model Comparison: RMSE & MAE\n(Lower is Better)')
ax[0].set_ylabel('Error')
ax[0].tick_params(axis='x', rotation=45)

# R² plot
results_df[['R²']].plot(kind='bar', ax=ax[1], legend=False, color='green')
ax[1].set_title('Model Comparison: R²\n(Higher is Better)')
ax[1].set_ylabel('R² Score')
ax[1].tick_params(axis='x', rotation=45)
ax[1].axhline(y=0, color='k', linestyle='-', alpha=0.3) # Add a base line at 0

plt.tight_layout()
plt.savefig('../models/thesis_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.3 Actual vs. Predicted Plot for the BEST model
best_model_name = results_df.index[0]
best_model = tuned_models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2) # Perfect prediction line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Actual vs. Predicted Values\n({best_model_name} Model)')
plt.tight_layout()
plt.savefig('../models/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------------------------
# 10. SAVE THE BEST MODEL FOR THE PROTOTYPE
# -------------------------------------------------------------------
# Save the winning model
best_model_path = f'../models/best_model_{best_model_name}.joblib'
joblib.dump(best_model, best_model_path)
print(f"\n[THESIS] Best model ({best_model_name}) saved to '{best_model_path}'.")

# -------------------------------------------------------------------
# 11. FINAL THESIS RESULTS OUTPUT
# -------------------------------------------------------------------
print("\n" + "="*60)
print("THESIS MODEL TRAINING PHASE COMPLETE")
print("="*60)
print("FINAL RESULTS:")
print(results_df)
print(f"\nConclusion: The {best_model_name} model performed best, achieving an RMSE of {results_df.loc[best_model_name, 'RMSE']:.4f} and an R² of {results_df.loc[best_model_name, 'R²']:.4f}.")
print("\nNext step: Use the saved model (.joblib) and scaler to build the Streamlit/Flask prototype.")