# check_s2_availability_batch.py
import pandas as pd
import ee
import os

# --- 1. Configuration ---
INPUT_CSV = 'all_peatland_clusters_prioritized.csv'
MAX_CLOUD_PERCENTAGE = 20
# --- IMPORTANT: Set your Google Cloud Storage bucket name ---
GCS_BUCKET_NAME = 'fire-app-bucket' # <-- EDIT THIS
OUTPUT_FILENAME = 's2_availability_results.csv'

# --- Authenticate and Initialize GEE ---
try:
    ee.Initialize(project='haryo-kebakaran')
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print(f"Error initializing GEE: {e}")
    exit()

# --- 2. Main Script Logic ---
def submit_availability_check_task():
    try:
        df = pd.read_csv(INPUT_CSV)
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{INPUT_CSV}'.")
        return
        
    print(f"Preparing to submit a batch task for {len(df)} clusters...")

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
        cluster_date = ee.Date(feature.get('system:time_start'))
        aoi = feature.geometry().buffer(100).bounds()
        image_count = s2_collection.filterBounds(aoi).filterDate(cluster_date, cluster_date.advance(1, 'day')).size()
        return feature.set('s2_image_available', ee.Number(image_count).gt(0))

    availability_fc = cluster_fc.map(check_image_for_cluster)
    
    # --- BATCH EXPORT LOGIC ---
    # Instead of .getInfo(), we export the result to a CSV in your GCS bucket.
    task = ee.batch.Export.table.toCloudStorage(
        collection=availability_fc,
        description='CheckS2Availability',
        bucket=GCS_BUCKET_NAME,
        fileNamePrefix=f'gee_outputs/{OUTPUT_FILENAME.replace(".csv", "")}',
        fileFormat='CSV',
        # We only need these two columns for the result
        selectors=['cluster_id', 's2_image_available']
    )

    # Start the task
    task.start()

    print("\n--- TASK SUBMITTED ---")
    print(f"A task named 'CheckS2Availability' has been submitted to Google Earth Engine.")
    print("This job will now run in the background on Google's servers and will not time out.")

if __name__ == "__main__":
    submit_availability_check_task()
