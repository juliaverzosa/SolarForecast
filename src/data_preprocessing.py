import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib # To save the scaler objects for later use in the app

# --- 1. Load the Raw Data from CSV ---
print("Loading raw data from CSV files...")
nasa_df = pd.read_csv('../data/raw/nasa_meteo_data.csv', parse_dates=['date'])
household_df = pd.read_csv('../data/raw/synthetic_household_data.csv')

# --- 2. Preprocess NASA Data ---
# Select only numeric columns for imputation and scaling
numeric_cols = nasa_df.select_dtypes(include=[np.number]).columns.tolist()

# Handle missing values
imputer = SimpleImputer(strategy='mean')
nasa_df_imputed = nasa_df.copy()
nasa_df_imputed[numeric_cols] = imputer.fit_transform(nasa_df[numeric_cols])

# Normalize/Scale the features
scaler = StandardScaler()
nasa_df_imputed[numeric_cols] = scaler.fit_transform(nasa_df_imputed[numeric_cols])
# Save the scaler for later use on new data
joblib.dump(scaler, '../models/meteo_scaler.joblib')

# --- 3. Aggregate NASA Data to Monthly ---
print("Aggregating data to monthly level...")
nasa_monthly_df = nasa_df_imputed.groupby(pd.Grouper(key='date', freq='M')).mean().reset_index()
nasa_monthly_df['month'] = nasa_monthly_df['date'].dt.month

# --- 4. Merge with Household Data ---
# For this example, we assign each household to a random month of data.
# In a real project, you'd have a specific location for each household.
np.random.seed(42)
household_df['month'] = np.random.randint(1, 13, size=len(household_df))

# Merge the datasets
merged_df = pd.merge(household_df, nasa_monthly_df, on='month', how='left')

# --- 5. Define Target and Features & Final Processing ---
# Let's assume our target is to predict consumption based on meteo and household data
X = merged_df[['household_size', 'has_aircon', 'has_water_heater', 'ALLSKY_SFC_SW_DWN', 'T2M', 'RH2M']]
y = merged_df['average_monthly_consumption_kwh']

# Handle any potential NaNs introduced during merging
X = X.fillna(X.mean())

# --- 6. Save the Processed Dataset to CSV ---
final_dataset = X.copy()
final_dataset['target_consumption'] = y

os.makedirs('../data/processed', exist_ok=True)
final_dataset.to_csv('../data/processed/final_training_dataset.csv', index=False)
print("Final dataset saved to '../data/processed/final_training_dataset.csv'")