# src/cloud_functions/incident_detector/main.py

import os
import json
import logging
from datetime import datetime, timezone

import pandas as pd
import geopandas as gpd
import hdbscan
import numpy as np
from google.cloud import pubsub, storage

from src.firms_data_retriever.retriever import FirmsDataRetriever, FIRMS_SENSORS
from src.common.config import GCS_PATHS, FILE_NAMES, GCS_BUCKET_NAME

# --- Configuration ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
FIRMS_API_KEY = os.environ.get("FIRMS_API_KEY")
OUTPUT_TOPIC_NAME = "wildfire-cluster-detected"
PEATLAND_SHP_PATH = "src/geodata/Indonesia_peat_lands.shp"
PEATLAND_BUFFER_METERS = 1000
HDBSCAN_MIN_CLUSTER_SIZE = 2

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def incident_detector_cloud_function(event, context):
    logging.info("Incident Detector function triggered.")

    if not GCS_BUCKET_NAME:
        logging.critical("GCS_BUCKET_NAME environment variable not set. Cannot proceed.")
        return

    run_date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    logging.info(f"Processing for run_date: {run_date_str}")

    firms_retriever = FirmsDataRetriever(
        api_key=FIRMS_API_KEY,
        base_url="https://firms.modaps.eosdis.nasa.gov/api/area/csv/",
        sensors=FIRMS_SENSORS
    )
    firms_df = firms_retriever.get_and_filter_firms_data([])

    if firms_df.empty:
        logging.warning("No FIRMS hotspots found globally. Exiting.")
        return

    gdf = gpd.GeoDataFrame(
        firms_df, geometry=gpd.points_from_xy(firms_df.longitude, firms_df.latitude), crs="EPSG:4326"
    )

    try:
        peatland_boundary_meters = gpd.read_file(PEATLAND_SHP_PATH)
        logging.info(f"Applying a {PEATLAND_BUFFER_METERS} meter buffer to peatland boundaries.")
        peatland_boundary_meters['geometry'] = peatland_boundary_meters.geometry.buffer(PEATLAND_BUFFER_METERS)
        peatland_final = peatland_boundary_meters.to_crs(gdf.crs)
    except Exception as e:
        logging.error(f"CRITICAL: Could not load, buffer, or reproject shapefile. Error: {e}", exc_info=True)
        return

    gdf_peatland_fires = gpd.sjoin(gdf, peatland_final, how="inner", predicate='within')

    if gdf_peatland_fires.empty:
        logging.info(f"No FIRMS hotspots found on or within {PEATLAND_BUFFER_METERS}m of Indonesian peatlands. Exiting.")
        return

    logging.info(f"Found {len(gdf_peatland_fires)} total fire points on or within {PEATLAND_BUFFER_METERS}m of Indonesian peatlands.")

    coords_radians = np.radians(gdf_peatland_fires[['latitude', 'longitude']].values)
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        metric='haversine'
    )
    clusterer.fit(coords_radians)

    cluster_labels = clusterer.labels_
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

    if n_clusters == 0:
        logging.warning("No clusters met the criteria. The pipeline will stop here.")
        return

    logging.info(f"Found {n_clusters} significant fire clusters using HDBSCAN with min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}.")

    gdf_peatland_fires['cluster_id'] = cluster_labels
    clustered_fires = gdf_peatland_fires[gdf_peatland_fires['cluster_id'] != -1]

    all_incidents = []
    for cluster_id_num in sorted(clustered_fires['cluster_id'].unique()):
        cluster_gdf = clustered_fires[clustered_fires['cluster_id'] == cluster_id_num]
        centroid = cluster_gdf.geometry.union_all().centroid

        incident_data = {
            "cluster_id": f"fire_cluster_{run_date_str.replace('-', '')}_{cluster_id_num}",
            "point_count": len(cluster_gdf),
            "centroid_latitude": centroid.y,
            "centroid_longitude": centroid.x,
            "hotspots": json.loads(cluster_gdf.to_json())['features']
        }
        all_incidents.append(incident_data)

    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    output_blob_path = f"{GCS_PATHS['INCIDENTS_DETECTED']}/{run_date_str}/{FILE_NAMES['incident_data']}"

    jsonl_content = "\n".join([json.dumps(incident) for incident in all_incidents])

    blob = bucket.blob(output_blob_path)
    blob.upload_from_string(jsonl_content, content_type='application/jsonl')
    logging.info(f"Successfully wrote {len(all_incidents)} incidents to gs://{GCS_BUCKET_NAME}/{output_blob_path}")

    publisher = pubsub.PublisherClient()
    topic_path = publisher.topic_path(GCP_PROJECT_ID, OUTPUT_TOPIC_NAME)

    notification_payload = {
        "status": "incidents_detected",
        "incident_count": len(all_incidents),
        "run_date": run_date_str,
        "output_path": f"gs://{GCS_BUCKET_NAME}/{output_blob_path}",
        "completion_time": datetime.now(timezone.utc).isoformat()
    }
    message_json = json.dumps(notification_payload)
    message_bytes = message_json.encode('utf-8')

    try:
        publish_future = publisher.publish(topic_path, data=message_bytes)
        publish_future.result()
        logging.info(f"Successfully published completion signal for {run_date_str}.")
    except Exception as e:
        logging.error(f"Failed to publish notification message. Error: {e}")

    logging.info("Incident Detector function finished successfully.")

if __name__ == "__main__":
    print("--- Running Incident Detector locally ---")
    os.environ['GCP_PROJECT_ID'] = 'haryo-kebakaran'
    os.environ['GCS_BUCKET_NAME'] = 'fire-app-bucket'
    if 'FIRMS_API_KEY' not in os.environ:
        os.environ['FIRMS_API_KEY'] = '0331973a7ee830ca7f026493faaa367a'
    incident_detector_cloud_function(event=None, context=None)
    print("--- Local run of Incident Detector finished ---")
