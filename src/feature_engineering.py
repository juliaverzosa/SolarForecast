import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import pvlib
from pvlib import location, irradiance, atmosphere
from datetime import datetime, timedelta
import math
import os

def create_temporal_features(df):
    """
    Create temporal features from NASA data
    """
    print("Creating temporal features...")
    
    # Ensure we have a datetime index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Extract month and create cyclical encoding
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Create seasonal indicator based on PAGASA classification
    # Dry season: Nov-Apr, Wet season: May-Oct
    df['season'] = np.where(df['month'].isin([11, 12, 1, 2, 3, 4]), 0, 1)  # 0 = Dry, 1 = Wet
    
    return df.reset_index()

def create_meteorological_features(df):
    """
    Create meteorological and solar features from NASA data
    """
    print("Creating meteorological features...")
    
    # Calculate sunshine hours (hours with irradiance > 120 W/m²)
    # NASA data provides daily ALLSKY_SFC_SW_DWN (GHI) in kWh/m²/day
    # Convert to average irradiance in W/m²: kWh/m²/day * 1000 / 24
    df['avg_irradiance_wm2'] = df['ALLSKY_SFC_SW_DWN'] * 1000 / 24
    
    # Estimate sunshine hours (simplified approach)
    # Assuming linear relationship between GHI and sunshine hours
    df['sunshine_hours'] = df['ALLSKY_SFC_SW_DWN'] / 5.0 * 12  # 5 kWh/m²/day ≈ 12 sunshine hours
    
    # Calculate clear sky ratio
    # Using the Ineichen-Perez model as implemented in pvlib
    lat, lon = 7.1907, 125.4553  # Davao coordinates
    site = location.Location(lat, lon, tz='Asia/Manila')
    
    # Create times for the year
    times = pd.date_range(
        start='2024-01-01', 
        end='2024-12-31', 
        freq='D', 
        tz=site.tz
    )
    
    # Get clear sky irradiance
    clear_sky = site.get_clearsky(times, model='ineichen')
    
    # Convert clear_sky DataFrame to have a 'date' column
    clear_sky_df = clear_sky.reset_index()  # This creates a column named 'index' from the datetime index
    clear_sky_df.rename(columns={'index': 'date'}, inplace=True)  # Rename 'index' to 'date'
    
    # Extract the date part only for merging
    clear_sky_df['date_only'] = clear_sky_df['date'].dt.date
    
    # Ensure our main DataFrame also has a 'date_only' column
    df['date_only'] = pd.to_datetime(df['date']).dt.date
    
    # Merge the clear sky data with our main DataFrame
    df = pd.merge(df, clear_sky_df[['date_only', 'ghi']],  # Use 'ghi' which is the clear sky global horizontal irradiance
                  on='date_only', how='left')
    
    # Calculate clear sky ratio: actual GHI / clear sky GHI
# Add a small epsilon to prevent division by zero
    epsilon = 1e-10
    df['clear_sky_ratio'] = df['ALLSKY_SFC_SW_DWN'] / (df['ghi'] / 1000 + epsilon)

    # Cap the ratio at a reasonable maximum value (e.g., 2.0)
    df['clear_sky_ratio'] = np.clip(df['clear_sky_ratio'], 0, 2.0)    
    # Clean up: remove temporary columns
    df.drop(columns=['date_only', 'ghi'], inplace=True)
    
    return df

def calculate_orientation_score(azimuth):
    """
    Calculate orientation score for a given azimuth angle
    """
    # For Northern Hemisphere, optimal is 180° (due south)
    optimal_azimuth = 180
    # Calculate absolute difference from optimal
    diff = abs(azimuth - optimal_azimuth)
    # Normalize to 0-180 range and convert to score (1 at optimal, 0 at worst)
    score = 1 - (min(diff, 360 - diff) / 180)
    return max(score, 0)  # Ensure non-negative

def calculate_tilt_factor(roof_tilt):
    """
    Calculate tilt factor based on optimal tilt for latitude
    """
    latitude = 7.2  # Davao's latitude
    optimal_tilt = latitude  # Optimal tilt equals latitude
    # Score how close actual tilt is to optimal
    tilt_factor = np.cos(np.radians(abs(roof_tilt - optimal_tilt)))
    return tilt_factor

def estimate_shading_factor(building, buildings_gdf, buffer_distance=50):
    """
    Estimate shading factor based on nearby building density
    """
    # Create a buffer around the building
    buffer = building.geometry.buffer(buffer_distance)
    
    # Find intersecting buildings (excluding self)
    nearby_buildings = buildings_gdf[buildings_gdf.intersects(buffer) & 
                                    (buildings_gdf.index != building.name)]
    
    # Calculate total area of nearby buildings
    nearby_area = nearby_buildings.geometry.area.sum()
    
    # Calculate area of buffer
    buffer_area = buffer.area
    
    # Shading factor is ratio of nearby area to buffer area
    # Clamped to 0.8 to avoid complete obstruction
    shading_factor = min(nearby_area / buffer_area, 0.8) if buffer_area > 0 else 0
    
    return shading_factor

def create_topographical_features(buildings_gdf):
    """
    Create topographical and structural features from OSM data
    """
    print("Creating topographical features...")
    
    # Calculate orientation (azimuth) for each building
    # For simplicity, we'll assume the longest edge determines orientation
    def calculate_azimuth(geometry):
        if geometry.geom_type == 'Polygon':
            # Get the longest edge of the polygon
            coords = list(geometry.exterior.coords)
            max_length = 0
            azimuth = 0
            
            for i in range(len(coords)-1):
                x1, y1 = coords[i]
                x2, y2 = coords[i+1]
                length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if length > max_length:
                    max_length = length
                    # Calculate angle from north (0°)
                    dx, dy = x2-x1, y2-y1
                    azimuth = (math.degrees(math.atan2(dx, dy)) + 360) % 360
            
            return azimuth
        else:
            # For MultiPolygon, use the first polygon
            return calculate_azimuth(geometry.geoms[0])
    
    buildings_gdf['azimuth'] = buildings_gdf.geometry.apply(calculate_azimuth)
    buildings_gdf['orientation_score'] = buildings_gdf['azimuth'].apply(calculate_orientation_score)
    
    # Calculate tilt factor (assuming flat roofs as per note)
    buildings_gdf['roof_tilt'] = 0  # Default assumption for Philippines
    buildings_gdf['tilt_factor'] = buildings_gdf['roof_tilt'].apply(calculate_tilt_factor)
    
    # Estimate shading factor
    print("  Estimating shading factors (this may take a while)...")
    buildings_gdf['shading_factor'] = buildings_gdf.apply(
        lambda row: estimate_shading_factor(row, buildings_gdf), axis=1
    )
    
    # Calculate Solar Exposure Index (SEI)
    buildings_gdf['SEI'] = (buildings_gdf['orientation_score'] * 
                           buildings_gdf['rooftop_area_sq_m'] * 
                           (1 - buildings_gdf['shading_factor']) * 
                           buildings_gdf['tilt_factor'])
    
    return buildings_gdf

def create_consumption_features(household_df):
    """
    Create household consumption features
    """
    print("Creating consumption features...")
    
    # Estimate daytime load percentage (assume 40-60% of consumption occurs during daylight)
    np.random.seed(42)  # For reproducibility
    household_df['daytime_load_percentage'] = np.random.uniform(0.4, 0.6, len(household_df))
    
    # Calculate estimated daytime consumption
    household_df['daytime_consumption_kwh'] = (
        household_df['average_monthly_consumption_kwh'] * 
        household_df['daytime_load_percentage']
    )
    
    # Create a binary indicator for high consumption households
    # (above median consumption)
    median_consumption = household_df['average_monthly_consumption_kwh'].median()
    household_df['high_consumption'] = (
        household_df['average_monthly_consumption_kwh'] > median_consumption
    ).astype(int)
    
    return household_df

def main():
    """
    Main function to run the feature engineering pipeline
    """
    print("Starting feature engineering pipeline...")
    
    # Load the raw data
    try:
        nasa_df = pd.read_csv('../data/raw/nasa_meteo_data.csv')
        buildings_gdf = gpd.read_file('../data/raw/osm_buildings_data.geojson')
        household_df = pd.read_csv('../data/raw/synthetic_household_data.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the data acquisition script first.")
        return
    
    # Create temporal features
    nasa_with_temporal = create_temporal_features(nasa_df)
    
    # Create meteorological features
    nasa_with_meteo = create_meteorological_features(nasa_with_temporal)
    
    # Create topographical features
    buildings_with_sei = create_topographical_features(buildings_gdf)
    
    # Create consumption features
    household_with_consumption = create_consumption_features(household_df)
    
    # Save the processed data
    os.makedirs('../data/processed', exist_ok=True)
    
    nasa_with_meteo.to_csv('../data/processed/nasa_processed.csv', index=False)
    buildings_with_sei.to_file('../data/processed/buildings_with_sei.geojson', driver='GeoJSON')
    household_with_consumption.to_csv('../data/processed/household_processed.csv', index=False)
    
    print("Feature engineering complete!")
    print("Processed data saved to ../data/processed/")
    
    # Create a summary dataset for modeling
    # For each building, we need to join with meteorological and household data
    # This is a simplified example - you might need to adapt based on your specific modeling approach
    
    # Aggregate NASA data to monthly averages
    nasa_monthly = nasa_with_meteo.groupby('month').agg({
        'ALLSKY_SFC_SW_DWN': 'mean',
        'T2M': 'mean',
        'RH2M': 'mean',
        'ALLSKY_KT': 'mean',
        'sunshine_hours': 'mean',
        'clear_sky_ratio': 'mean'
    }).reset_index()
    
    # Assign each building a random household (in a real scenario, you'd have a proper join key)
    np.random.seed(42)
    building_ids = buildings_with_sei.index.tolist()
    household_ids = household_with_consumption['household_id'].tolist()
    
    # Create a mapping (each building gets one household)
    building_household_map = {
        bid: np.random.choice(household_ids) for bid in building_ids
    }
    
    # Create modeling dataset
    modeling_data = []
    
    for idx, building in buildings_with_sei.iterrows():
        # Get a random household for this building
        household_id = building_household_map[idx]
        household = household_with_consumption[
            household_with_consumption['household_id'] == household_id
        ].iloc[0]
        
        # For each month, create a record
        for month in range(1, 13):
            month_data = nasa_monthly[nasa_monthly['month'] == month].iloc[0]
            
            record = {
                'building_id': idx,
                'household_id': household_id,
                'month': month,
                'month_sin': np.sin(2 * np.pi * month/12),
                'month_cos': np.cos(2 * np.pi * month/12),
                'season': 0 if month in [11, 12, 1, 2, 3, 4] else 1,
                'ghi': month_data['ALLSKY_SFC_SW_DWN'],
                'temperature': month_data['T2M'],
                'humidity': month_data['RH2M'],
                'clearness_index': month_data['ALLSKY_KT'],
                'sunshine_hours': month_data['sunshine_hours'],
                'clear_sky_ratio': month_data['clear_sky_ratio'],
                'rooftop_area_sq_m': building['rooftop_area_sq_m'],
                'orientation_score': building['orientation_score'],
                'tilt_factor': building['tilt_factor'],
                'shading_factor': building['shading_factor'],
                'SEI': building['SEI'],
                'household_size': household['household_size'],
                'has_aircon': household['has_aircon'],
                'has_water_heater': household['has_water_heater'],
                'average_monthly_consumption_kwh': household['average_monthly_consumption_kwh'],
                'daytime_load_percentage': household['daytime_load_percentage'],
                'daytime_consumption_kwh': household['daytime_consumption_kwh'],
                'high_consumption': household['high_consumption']
            }
            
            modeling_data.append(record)
    
    # Convert to DataFrame
    modeling_df = pd.DataFrame(modeling_data)
    
    # Save modeling dataset
    modeling_df.to_csv('../data/processed/modeling_dataset.csv', index=False)
    print("Modeling dataset created and saved to ../data/processed/modeling_dataset.csv")
    
    return modeling_df

if __name__ == "__main__":
    modeling_df = main()