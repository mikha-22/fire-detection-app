# check_s2_availability.py
import pandas as pd
import ee
import os

# --- 1. Configuration ---
INPUT_CSV = 'all_peatland_clusters_prioritized.csv'
MAX_CLOUD_PERCENTAGE = 20

# --- Authenticate and Initialize GEE ---
try:
    ee.Initialize(project='haryo-kebakaran')
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print(f"Error initializing GEE: {e}")
    exit()

# --- 2. Main Script Logic ---
def check_availability():
    try:
        df = pd.read_csv(INPUT_CSV)
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{INPUT_CSV}'.")
        return
        
    print(f"Checking {len(df)} clusters for same-day Sentinel-2 availability...")

    features = []
    for index, row in df.iterrows():
        geom = ee.Geometry.Point(row['centroid_lon'], row['centroid_lat'])
        properties = {
            'cluster_id': row['cluster_id'],
            'system:time_start': ee.Date(row['date']).millis()
        }
        features.append(ee.Feature(geom, properties))
    
    cluster_fc = ee.FeatureCollection(features)

    s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_PERCENTAGE)))

    def check_image_for_cluster(feature):
        # --- THIS IS THE CORRECTED LINE ---
        cluster_date = ee.Date(feature.get('system:time_start'))
        # --- END CORRECTION ---

        aoi = feature.geometry().buffer(100).bounds()

        image_count = s2_collection.filterBounds(aoi).filterDate(cluster_date, cluster_date.advance(1, 'day')).size()
        
        return feature.set('s2_image_available', ee.Number(image_count).gt(0))

    availability_fc = cluster_fc.map(check_image_for_cluster)
    
    total_with_images = availability_fc.reduceColumns(
        reducer=ee.Reducer.sum(),
        selectors=['s2_image_available']
    ).get('sum').getInfo()

    print("\n--- CHECK COMPLETE ---")
    print(f"Result: {int(total_with_images)} out of {len(df)} clusters have a usable Sentinel-2 image on the exact date of detection.")

if __name__ == "__main__":
    check_availability()
