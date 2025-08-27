import pandas as pd
import ee
import os
import datetime
import time
import urllib.request

# --- 1. Configuration ---
INPUT_CSV = 'all_peatland_clusters_prioritized.csv'
OUTPUT_IMAGE_DIR = 's2_labeling_images_exact_date' # New folder for this strict output
MAX_CLUSTERS_TO_FETCH = 200 # You can increase this, as many will be skipped
MAX_CLOUD_PERCENTAGE = 30
IMAGE_DIMENSIONS = 768

# --- Authenticate and Initialize GEE ---
try:
    ee.Initialize(project='haryo-kebakaran')
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print(f"Error initializing GEE. Please run 'gcloud auth application-default login'. Error: {e}")
    exit()

# --- 2. Main Script Logic ---
def fetch_images_for_labeling():
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{INPUT_CSV}'.")
        return

    print(f"Starting exact-date image fetch for the top {MAX_CLUSTERS_TO_FETCH} clusters...")
    images_found = 0

    for index, row in df.head(MAX_CLUSTERS_TO_FETCH).iterrows():
        cluster_id = row['cluster_id']
        centroid_lat = row['centroid_lat']
        centroid_lon = row['centroid_lon']
        
        try:
            fire_date = pd.to_datetime(row['date']).tz_localize(None)
        except (TypeError, ValueError):
            fire_date = datetime.datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S')

        output_filename = os.path.join(OUTPUT_IMAGE_DIR, f"{cluster_id}.png")

        if os.path.exists(output_filename):
            print(f"Skipping {cluster_id} (image already exists).")
            continue

        print(f"\nProcessing cluster: {cluster_id} (Searching on date: {fire_date.strftime('%Y-%m-%d')})")

        try:
            # --- MODIFIED: Exact Date Filtering ---
            # Set the start to the beginning of the fire date and the end to the next day
            exact_date = fire_date.strftime('%Y-%m-%d')
            next_day = (fire_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            
            aoi = ee.Geometry.Point(centroid_lon, centroid_lat).buffer(7000).bounds()

            s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                             .filterBounds(aoi)
                             .filterDate(exact_date, next_day) # Search only within this single day
                             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_PERCENTAGE)))
            # --- END MODIFICATION ---

            if s2_collection.size().getInfo() == 0:
                print(f"--> No image found for {cluster_id} on the exact date.")
                continue

            best_image = s2_collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
            
            image_date = datetime.datetime.fromtimestamp(best_image.get('system:time_start').getInfo() / 1000).strftime('%Y-%m-%d')
            cloud_cover = best_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()

            print(f"--> Found image: Date={image_date}, Clouds={cloud_cover:.2f}%")

            vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3500, 'gamma': 1.4}
            url = best_image.getThumbURL({
                'region': aoi.getInfo()['coordinates'],
                'dimensions': IMAGE_DIMENSIONS,
                'format': 'png',
                **vis_params
            })
            
            urllib.request.urlretrieve(url, output_filename)
            print(f"--> Successfully downloaded image to '{output_filename}'")
            images_found += 1
            
            time.sleep(1)

        except Exception as e:
            print(f"--> ERROR processing {cluster_id}: {e}")
            if "Too Many Requests" in str(e) or "HTTPError" in str(e):
                print("--> Server busy, waiting for 10 seconds...")
                time.sleep(10)
            continue
            
    print(f"\nImage fetching process complete. Found images for {images_found} out of {MAX_CLUSTERS_TO_FETCH} clusters processed.")

if __name__ == "__main__":
    fetch_images_for_labeling()
