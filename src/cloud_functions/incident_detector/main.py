# src/cloud_functions/incident_detector/main.py

import os
import json
import logging
from datetime import datetime

import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN
from google.cloud import pubsub, storage
import numpy as np

from src.firms_data_retriever.retriever import FirmsDataRetriever

# --- Configuration ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME") # Added GCS Bucket
FIRMS_API_KEY = os.environ.get("FIRMS_API_KEY")
OUTPUT_TOPIC_NAME = "wildfire-cluster-detected"
INCIDENTS_GCS_PREFIX = "incidents" # New GCS prefix for outputs
DESIRED_EPS_KM = 10
EARTH_RADIUS_KM = 6371
DBSCAN_EPS_RADIANS = DESIRED_EPS_KM / EARTH_RADIUS_KM
DBSCAN_MIN_SAMPLES = 2
PEATLAND_SHP_PATH = "src/geodata/Indonesia_peat_lands.shp"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def incident_detector_cloud_function(event, context):
    logging.info("Incident Detector function triggered.")

    if not GCS_BUCKET_NAME:
        logging.critical("GCS_BUCKET_NAME environment variable not set. Cannot proceed.")
        return

    # --- MODIFIED: Establish a single run_date for this execution ---
    run_date_str = datetime.utcnow().strftime('%Y-%m-%d')
    logging.info(f"Processing for run_date: {run_date_str}")

    firms_retriever = FirmsDataRetriever(api_key=FIRMS_API_KEY, base_url="https://firms.modaps.eosdis.nasa.gov/api/area/csv/", sensors=["VIIRS_SNPP_NRT"])
    firms_df = firms_retriever.get_and_filter_firms_data([])

    if firms_df.empty:
        logging.warning("No FIRMS hotspots found globally. Exiting.")
        return

    gdf = gpd.GeoDataFrame(
        firms_df, geometry=gpd.points_from_xy(firms_df.longitude, firms_df.latitude), crs="EPSG:4326"
    )

    try:
        peatland_boundary = gpd.read_file(PEATLAND_SHP_PATH).to_crs(gdf.crs)
    except Exception as e:
        logging.error(f"CRITICAL: Could not load or reproject shapefile. Error: {e}", exc_info=True)
        return

    gdf_peatland_fires = gpd.sjoin(gdf, peatland_boundary, how="inner", predicate='within')

    if gdf_peatland_fires.empty:
        logging.info("No FIRMS hotspots found on Indonesian peatlands. Exiting.")
        return

    logging.info(f"Found {len(gdf_peatland_fires)} total fire points on Indonesian peatlands.")

    coords_radians = np.radians(gdf_peatland_fires[['latitude', 'longitude']].values)
    db = DBSCAN(eps=DBSCAN_EPS_RADIANS, min_samples=DBSCAN_MIN_SAMPLES, algorithm='ball_tree', metric='haversine').fit(coords_radians)

    cluster_labels = db.labels_
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    
    if n_clusters == 0:
        logging.warning("No clusters met the criteria. The pipeline will stop here.")
        return
        
    logging.info(f"Found {n_clusters} significant fire clusters.")

    gdf_peatland_fires['cluster_id'] = cluster_labels
    clustered_fires = gdf_peatland_fires[gdf_peatland_fires['cluster_id'] != -1]
    
    all_incidents = []
    for cluster_id_num in sorted(clustered_fires['cluster_id'].unique()):
        cluster_gdf = clustered_fires[clustered_fires['cluster_id'] == cluster_id_num]
        centroid = cluster_gdf.geometry.unary_union.centroid

        incident_data = {
            "cluster_id": f"fire_cluster_{run_date_str.replace('-', '')}_{cluster_id_num}",
            "point_count": len(cluster_gdf),
            "centroid_latitude": centroid.y,
            "centroid_longitude": centroid.x,
            "hotspots": json.loads(cluster_gdf.to_json())['features']
        }
        all_incidents.append(incident_data)

    # --- MODIFIED: Write incidents to GCS instead of putting in Pub/Sub ---
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    output_blob_name = f"{INCIDENTS_GCS_PREFIX}/{run_date_str}/detected_incidents.jsonl"
    
    # Convert each incident dict to a JSON string and join with newlines
    jsonl_content = "\n".join([json.dumps(incident) for incident in all_incidents])
    
    blob = bucket.blob(output_blob_name)
    blob.upload_from_string(jsonl_content, content_type='application/jsonl')
    logging.info(f"Successfully wrote {len(all_incidents)} incidents to gs://{GCS_BUCKET_NAME}/{output_blob_name}")

    # --- MODIFIED: Publish a lightweight notification message ---
    publisher = pubsub.PublisherClient()
    topic_path = publisher.topic_path(GCP_PROJECT_ID, OUTPUT_TOPIC_NAME)
    
    notification_payload = {"run_date": run_date_str}
    message_json = json.dumps(notification_payload)
    message_bytes = message_json.encode('utf-8')

    try:
        publish_future = publisher.publish(topic_path, data=message_bytes)
        publish_future.result()
        logging.info(f"Successfully published notification for run {run_date_str}.")
    except Exception as e:
        logging.error(f"Failed to publish notification message. Error: {e}")

    logging.info("Incident Detector function finished successfully.")
