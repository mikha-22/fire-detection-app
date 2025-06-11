# src/cloud_functions/incident_detector/main.py

import os
import json
import logging
from datetime import datetime

import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN
from google.cloud import pubsub
import numpy as np # <-- ADDED THIS IMPORT

# Note: We now import the retriever from the shared src directory
from src.firms_data_retriever.retriever import FirmsDataRetriever

# --- Configuration ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
FIRMS_API_KEY = os.environ.get("FIRMS_API_KEY")
# This is the new topic we publish to, triggering the next function
OUTPUT_TOPIC_NAME = "wildfire-cluster-detected"

# --- MODIFIED TO ACCOUNT FOR EARTH'S CURVATURE (Haversine Distance) ---
# Define the desired cluster radius in kilometers
DESIRED_EPS_KM = 10 # Example: 10 kilometers. Adjust as needed for your definition of a "cluster"
EARTH_RADIUS_KM = 6371 # Mean Earth radius in kilometers (approximate)

# Convert the desired EPS from kilometers to radians for the Haversine metric
DBSCAN_EPS_RADIANS = DESIRED_EPS_KM / EARTH_RADIUS_KM

# Minimum number of fire points to form a cluster
# Keeping this low (e.g., 2) for testing purposes to ensure clusters are found.
DBSCAN_MIN_SAMPLES = 2 # TEMPORARY: Reduced for testing to force cluster detection
# --- END MODIFIED FOR TESTING ---

# --- CORRECTED PATH ---
# Path to the shapefile within the deployment package
PEATLAND_SHP_PATH = "src/geodata/Indonesia_peat_lands.shp"

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def incident_detector_cloud_function(event, context):
    """
    This Cloud Function:
    1. Fetches global FIRMS fire data.
    2. Filters it to Indonesian peatlands using a shapefile.
    3. Finds significant clusters of fire activity.
    4. Publishes a message for each cluster to trigger the next stage.
    """
    logging.info("Incident Detector function triggered.")

    # 1. Fetch Global FIRMS Data
    # We pass an empty list `[]` to the original function to tell it *not* to
    # filter by bounding box, so we get the full global dataset.
    firms_retriever = FirmsDataRetriever(api_key=FIRMS_API_KEY, base_url="https://firms.modaps.eosdis.nasa.gov/api/area/csv/", sensors=["VIIRS_SNPP_NRT"])
    firms_df = firms_retriever.get_and_filter_firms_data([]) # Pass empty list to get global data

    if firms_df.empty:
        logging.warning("No FIRMS hotspots found globally. Exiting.")
        return

    # Convert to a GeoDataFrame for spatial analysis. This creates data in EPSG:4326.
    gdf = gpd.GeoDataFrame(
        firms_df, geometry=gpd.points_from_xy(firms_df.longitude, firms_df.latitude), crs="EPSG:4326"
    )

    # 2. Filter by Peatlands using the Shapefile
    try:
        logging.info(f"Reading peatland boundaries from local file: {PEATLAND_SHP_PATH}")
        peatland_boundary = gpd.read_file(PEATLAND_SHP_PATH)

        # --- CRS MISMATCH FIX ---
        # Before comparing, re-project the peatland boundary data to match the CRS
        # of the fire hotspot data (EPSG:4326). This ensures an accurate spatial join.
        peatland_boundary = peatland_boundary.to_crs(gdf.crs)

    except Exception as e:
        logging.error(f"CRITICAL: Could not load or reproject shapefile at '{PEATLAND_SHP_PATH}'. Error: {e}", exc_info=True)
        return

    # Keep only the fire points that are 'within' the peatland polygons.
    # The sjoin is now accurate because both GeoDataFrames are in the same CRS.
    gdf_peatland_fires = gpd.sjoin(gdf, peatland_boundary, how="inner", predicate='within')

    if gdf_peatland_fires.empty:
        logging.info("No FIRMS hotspots found on Indonesian peatlands after filtering. Exiting.")
        return

    logging.info(f"Found {len(gdf_peatland_fires)} total fire points on Indonesian peatlands.")

    # 3. Analyse for Clusters
    # Convert latitude and longitude to radians for Haversine distance metric
    # DBSCAN's haversine metric expects (latitude, longitude) in radians, NOT (longitude, latitude)
    coords_radians = np.radians(gdf_peatland_fires[['latitude', 'longitude']].values)

    # Use Haversine distance metric
    db = DBSCAN(eps=DBSCAN_EPS_RADIANS, min_samples=DBSCAN_MIN_SAMPLES,
                algorithm='ball_tree', metric='haversine').fit(coords_radians)

    cluster_labels = db.labels_
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    # Log the EPS in kilometers for clarity
    logging.info(f"Found {n_clusters} significant fire clusters with DBSCAN(eps={DESIRED_EPS_KM}km, min_samples={DBSCAN_MIN_SAMPLES}).")

    if n_clusters == 0:
        logging.warning("No clusters met the criteria. The pipeline will stop here.")
        return

    # 4. Prepare and Publish Data for Each Cluster
    gdf_peatland_fires['cluster_id'] = cluster_labels
    # Ignore noise points (label -1)
    clustered_fires = gdf_peatland_fires[gdf_peatland_fires['cluster_id'] != -1]

    publisher = pubsub.PublisherClient()
    topic_path = publisher.topic_path(GCP_PROJECT_ID, OUTPUT_TOPIC_NAME)

    for cluster_id in sorted(clustered_fires['cluster_id'].unique()):
        cluster_gdf = clustered_fires[clustered_fires['cluster_id'] == cluster_id]

        centroid = cluster_gdf.geometry.unary_union.centroid

        cluster_data = {
            "cluster_id": f"fire_cluster_{datetime.utcnow().strftime('%Y%m%d')}_{cluster_id}",
            "point_count": len(cluster_gdf),
            "centroid_latitude": centroid.y,
            "centroid_longitude": centroid.x,
            "hotspots": json.loads(cluster_gdf.to_json())['features']
        }

        message_json = json.dumps(cluster_data, default=str)
        message_bytes = message_json.encode('utf-8')

        try:
            # Publish one message per cluster
            publish_future = publisher.publish(topic_path, data=message_bytes)
            publish_future.result()
            logging.info(f"Published message for cluster {cluster_id} to {OUTPUT_TOPIC_NAME}.")
        except Exception as e:
            logging.error(f"Failed to publish message for cluster {cluster_id}. Error: {e}")

    logging.info("Incident Detector function finished successfully.")
