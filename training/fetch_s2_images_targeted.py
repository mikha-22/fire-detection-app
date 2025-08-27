import pandas as pd
import ee
import os
import datetime
import time
import urllib.request

# --- 1. Configuration ---
ALL_CLUSTERS_CSV = 'all_peatland_clusters_prioritized.csv'
# --- MODIFIED: Path now points to your local downloaded file ---
AVAILABILITY_CSV_PATH = 'gee_outputs_s2_availability_results.csv'
OUTPUT_IMAGE_DIR = 's2_labeling_images_exact_date'
MAX_CLOUD_PERCENTAGE = 20
IMAGE_DIMENSIONS = 768

# --- Authenticate and Initialize GEE ---
try:
    ee.Initialize(project='haryo-kebakaran')
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print(f"Error initializing GEE: {e}")
    exit()

# --- 2. Main Script Logic ---
def fetch_images_for_labeling():
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    try:
        df_all_clusters = pd.read_csv(ALL_CLUSTERS_CSV)
    except FileNotFoundError:
        print(f"ERROR: Prioritized cluster file not found at '{ALL_CLUSTERS_CSV}'.")
        return

    # --- Load the list of valid cluster IDs from the local file ---
    try:
        df_results = pd.read_csv(AVAILABILITY_CSV_PATH)
        valid_clusters_df = df_results[df_results['s2_image_available'] == 1]
        # Use a set for very fast lookups
        valid_cluster_ids = set(valid_clusters_df['cluster_id'].tolist())
        print(f"Loaded {len(valid_cluster_ids)} valid cluster IDs to fetch images for.")
    except Exception as e:
        print(f"Could not load the availability results file '{AVAILABILITY_CSV_PATH}'. Error: {e}")
        return

    total_to_process = len(df_all_clusters)
    images_downloaded = 0

    for index, row in df_all_clusters.iterrows():
        cluster_id = row['cluster_id']

        # Check if the current cluster is in our valid list
        if cluster_id not in valid_cluster_ids:
            continue # Instantly skip this cluster

        centroid_lat = row['centroid_lat']
        centroid_lon = row['centroid_lon']
        
        try:
            fire_date = pd.to_datetime(row['date']).tz_localize(None)
        except (TypeError, ValueError):
            fire_date = datetime.datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S')

        output_filename = os.path.join(OUTPUT_IMAGE_DIR, f"{cluster_id}.png")

        if os.path.exists(output_filename):
            continue

        print(f"\n({index + 1}/{total_to_process}) Fetching image for valid cluster: {cluster_id}")

        try:
            exact_date = fire_date.strftime('%Y-%m-%d')
            next_day = (fire_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            aoi = ee.Geometry.Point(centroid_lon, centroid_lat).buffer(7000).bounds()

            s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                             .filterBounds(aoi)
                             .filterDate(exact_date, next_day)
                             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_PERCENTAGE)))

            best_image = s2_collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
            
            vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3500, 'gamma': 1.4}
            url = best_image.getThumbURL({
                'region': aoi.getInfo()['coordinates'],
                'dimensions': IMAGE_DIMENSIONS,
                'format': 'png',
                **vis_params
            })
            
            urllib.request.urlretrieve(url, output_filename)
            print(f"--> Successfully downloaded image.")
            images_downloaded += 1
            time.sleep(1)

        except Exception as e:
            print(f"--> ERROR processing {cluster_id}: {e}")
            if "Too Many Requests" in str(e) or "HTTPError" in str(e):
                print("--> Server busy, waiting for 10 seconds...")
                time.sleep(10)
            continue
            
    print(f"\nTargeted image fetching complete. Downloaded {images_downloaded} images.")

if __name__ == "__main__":
    fetch_images_for_labeling()
