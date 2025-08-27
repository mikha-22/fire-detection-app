import pandas as pd
import ee
import os
import datetime
import time

# --- 1. Configuration ---
ALL_CLUSTERS_CSV = 'all_peatland_clusters_prioritized.csv'
AVAILABILITY_CSV_PATH = 'gee_outputs_s2_availability_results.csv'
GCS_BUCKET_NAME = 'fire-app-bucket'
GCS_OUTPUT_FOLDER = 's2_labeling_images'
MAX_CLOUD_PERCENTAGE = 20

# --- Authenticate and Initialize GEE ---
try:
    # Use a high-volume endpoint for submitting many tasks
    ee.Initialize(project='haryo-kebakaran', opt_url='https://earthengine-highvolume.googleapis.com')
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print(f"Error initializing GEE: {e}")
    exit()

# --- 2. Main Script Logic ---
def submit_all_export_tasks():
    try:
        df_all_clusters = pd.read_csv(ALL_CLUSTERS_CSV)
    except FileNotFoundError:
        print(f"ERROR: Prioritized cluster file not found at '{ALL_CLUSTERS_CSV}'.")
        return

    try:
        df_results = pd.read_csv(AVAILABILITY_CSV_PATH)
        valid_clusters_df = df_results[df_results['s2_image_available'] == 1]
        valid_ids_set = set(valid_clusters_df['cluster_id'].tolist())
        print(f"Loaded {len(valid_ids_set)} valid cluster IDs from local file.")
    except Exception as e:
        print(f"Could not load the availability results file '{AVAILABILITY_CSV_PATH}'. Error: {e}")
        return

    # Filter the main DataFrame to only the clusters we need to export
    df_to_export = df_all_clusters[df_all_clusters['cluster_id'].isin(valid_ids_set)].copy()
    df_to_export['date'] = pd.to_datetime(df_to_export['date'])
    print(f"Preparing to submit {len(df_to_export)} export tasks...")

    # Load the base Sentinel-2 collection once
    s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_PERCENTAGE)))

    # --- Loop and Submit Tasks ---
    for index, row in df_to_export.iterrows():
        cluster_id = str(row['cluster_id'])
        fire_date = row['date']
        centroid_lon = row['centroid_lon']
        centroid_lat = row['centroid_lat']

        try:
            # Define exact date and area of interest
            start_date = fire_date.strftime('%Y-%m-%d')
            end_date = (fire_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            aoi = ee.Geometry.Point(centroid_lon, centroid_lat).buffer(7000).bounds()

            # Find the best image for this specific cluster
            best_image = (s2_collection
                          .filterBounds(aoi)
                          .filterDate(start_date, end_date)
                          .sort('CLOUDY_PIXEL_PERCENTAGE')
                          .first())

            # Create the visual PNG representation
            vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3500, 'gamma': 1.4}
            image_to_export = best_image.visualize(**vis_params).clip(aoi).toByte()

            # Define the export task to Google Cloud Storage
            task = ee.batch.Export.image.toCloudStorage(
                image=image_to_export,
                description=f'Export_{cluster_id}',
                bucket=GCS_BUCKET_NAME,
                fileNamePrefix=f'{GCS_OUTPUT_FOLDER}/{cluster_id}',
                scale=30, # A reasonable resolution for visual inspection
                fileFormat='GeoTIFF'
            )
            
            # Start the task
            task.start()
            print(f"({index + 1}/{len(df_to_export)}) Submitted task for cluster: {cluster_id}")
            
            # Brief pause to avoid overwhelming the submission API
            time.sleep(0.1)

        except Exception as e:
            print(f"--> ERROR submitting task for {cluster_id}: {e}")
            continue

    print("\n--- All tasks submitted! ---")
    print("Check the GEE Tasks Tab to monitor progress. Images will appear in your GCS bucket as they are completed.")

if __name__ == "__main__":
    submit_all_export_tasks()
