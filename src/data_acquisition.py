import pandas as pd
import numpy as np
import requests
import osmnx as ox
import os
import json
from pandas import json_normalize

# Create the data/raw directory if it doesn't exist
os.makedirs('../data/raw', exist_ok=True)

# --- 1. Fetch NASA Meteorological Data ---
def fetch_nasa_data():
    print("Fetching NASA POWER data...")
    lat, lon = 7.1907, 125.4553  # Davao coordinates
    start, end = '20230101', '20231231'
    
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        'parameters': 'ALLSKY_SFC_SW_DWN,T2M,RH2M,ALLSKY_KT',  # GHI, Temp, Humidity, Cloud
        'start': start,
        'end': end,
        'latitude': lat,
        'longitude': lon,
        'community': 'RE',
        'format': 'JSON'
    }
    
    try:
        response = requests.get(base_url, params=params)
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"Error: API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
        data = response.json()
        
        # Debug: Save raw response to inspect structure
        with open('../data/raw/nasa_raw_response.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        # Check if the expected data is in the response
        if 'properties' not in data:
            print("Error: 'properties' key not found in API response")
            print("Available keys:", list(data.keys()))
            return None
            
        if 'parameter' not in data['properties']:
            print("Error: 'parameter' key not found in properties")
            print("Available keys in properties:", list(data['properties'].keys()))
            return None
        
        # Extract the parameter data which contains our time series
        param_data = data['properties']['parameter']
        
        # Convert to DataFrame - each parameter becomes a column
        df = pd.DataFrame(param_data)
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        df = df.reset_index().rename(columns={'index': 'date'})
        
        # SAVE TO CSV
        df.to_csv('../data/raw/nasa_meteo_data.csv', index=False)
        print("NASA data saved to '../data/raw/nasa_meteo_data.csv'")
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        
        return df
        
    except Exception as e:
        print(f"Error fetching NASA data: {str(e)}")
        return None

# --- 2. Fetch Building Data from OSM ---
# --- 2. Fetch Building Data from OSM ---
def fetch_osm_data():
    print("Fetching OSM building data...")
    place_name = "Davao City, Philippines"
    try:
        # Step 1: Get boundary polygon
        gdf = ox.geocode_to_gdf(place_name)
        polygon = gdf.geometry.iloc[0]

        # Step 2: Fetch building footprints inside polygon
        buildings = ox.features_from_polygon(polygon, tags={"building": True})

        # Filter to only polygons/multipolygons
        buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]

        # Keep only geometry + compute rooftop area
        buildings = buildings[["geometry"]].reset_index(drop=True)
        buildings["rooftop_area_sq_m"] = buildings.geometry.area

        # Save to file
        buildings.to_file("../data/raw/osm_buildings_data.geojson", driver="GeoJSON")
        print("OSM data saved to '../data/raw/osm_buildings_data.geojson'")
        print(f"Columns: {list(buildings.columns)}")
        print(f"Shape: {buildings.shape}")
        return buildings
    except Exception as e:
        print(f"Error fetching OSM data: {e}")
        return None


# --- 3. Generate Synthetic Household Data ---
def generate_household_data(num_households=1000):
    print("Generating synthetic household data...")
    np.random.seed(42)
    profiles = {
        'household_id': range(1, num_households+1),
        'household_size': np.random.randint(1, 6, num_households),
        'has_aircon': np.random.choice([0, 1], size=num_households, p=[0.4, 0.6]),
        'has_water_heater': np.random.choice([0, 1], size=num_households, p=[0.7, 0.3]),
    }
    df = pd.DataFrame(profiles)
    df['base_consumption'] = df['household_size'] * 50
    df['appliance_load'] = (df['has_aircon'] * 200) + (df['has_water_heater'] * 80)
    df['average_monthly_consumption_kwh'] = (df['base_consumption'] + 
                                             df['appliance_load'] + 
                                             np.random.normal(0, 30, num_households)
                                            )
    final_df = df[['household_id', 'household_size', 'average_monthly_consumption_kwh', 'has_aircon', 'has_water_heater']]
    
    # SAVE TO CSV
    final_df.to_csv('../data/raw/synthetic_household_data.csv', index=False)
    print("Household data saved to '../data/raw/synthetic_household_data.csv'")
    return final_df

if __name__ == "__main__":
    # This block runs only if this script is executed directly,
    # not when it's imported into another script.
    nasa_df = fetch_nasa_data()
    osm_gdf = fetch_osm_data()
    household_df = generate_household_data()