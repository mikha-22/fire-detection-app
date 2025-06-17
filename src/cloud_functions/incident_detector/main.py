# src/cloud_functions/incident_detector/main.py

import os
import json
import logging
from datetime import datetime, timezone

import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN
from google.cloud import pubsub, storage

from src.firms_data_retriever.retriever import FirmsDataRetriever
from src.common.config import GCS_PATHS, FILE_NAMES, GCS_BUCKET_NAME

# --- Configuration ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
FIRMS_API_KEY = os.environ.get("FIRMS_API_KEY")
OUTPUT_TOPIC_NAME = "wildfire-cluster-detected"
PEATLAND_SHP_PATH = "src/geodata/Indonesia_peat_lands.shp"
PEATLAND_BUFFER_METERS = 1000
MIN_SAMPLES_PER_CLUSTER = 2
DBSCAN_MAX_DISTANCE_KM = 15
DBSCAN_EPS_RAD = DBSCAN_MAX_DISTANCE_KM / 6371

# --- Bounding Box for the entire region of interest (Indonesia) ---
INDONESIA_BBOX = [95.0, -11.0, 141.0, 6.0]

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def incident_detector_cloud_function(event, context):
    logging.info("Incident Detector function triggered.")
    run_date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    logging.info(f"Processing for run_date: {run_date_str}")

    # 1. Retrieve FIRMS data ONLY for the Indonesia bounding box
    firms_retriever = FirmsDataRetriever(
        api_key=FIRMS_API_KEY,
        base_url="https://firms.modaps.eosdis.nasa.gov/api/area/csv/",
        sensors=["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]
    )
    regions_to_fetch = [{"id": "indonesia", "bbox": INDONESIA_BBOX}]
    firms_df = firms_retriever.get_and_filter_firms_data(regions_to_fetch)

    if firms_df.empty:
        logging.warning("No FIRMS hotspots found within Indonesia. Exiting.")
        return

    # 2. Filter hotspots to those on or near Indonesian peatlands
    hotspots_gdf = gpd.GeoDataFrame(
        firms_df, geometry=gpd.points_from_xy(firms_df.longitude, firms_df.latitude), crs="EPSG:4326"
    )
    try:
        peatlands = gpd.read_file(PEATLAND_SHP_PATH)
        peatlands_proj = peatlands.to_crs(epsg=3857)
        logging.info(f"Applying a {PEATLAND_BUFFER_METERS}-meter buffer to peatland boundaries.")
        peatlands_proj['geometry'] = peatlands_proj.geometry.buffer(PEATLAND_BUFFER_METERS)
        peatlands_buffered = peatlands_proj.to_crs(hotspots_gdf.crs)
        gdf_peatland_fires = gpd.sjoin(hotspots_gdf, peatlands_buffered, how="inner", predicate='within')
    except Exception as e:
        logging.error(f"CRITICAL: Could not load or process shapefile. Error: {e}", exc_info=True)
        return

    if gdf_peatland_fires.empty:
        logging.info(f"No FIRMS hotspots found on or near peatlands. Exiting.")
        return
    logging.info(f"Found {len(gdf_peatland_fires)} fire points on peatlands to be clustered.")

    # 3. Cluster the filtered peatland fires
    coords_radians = np.radians(gdf_peatland_fires[['latitude', 'longitude']].values)
    clusterer = DBSCAN(
        eps=DBSCAN_EPS_RAD, min_samples=MIN_SAMPLES_PER_CLUSTER,
        algorithm='ball_tree', metric='haversine'
    )
    clusterer.fit(coords_radians)
    
    gdf_peatland_fires['cluster_id'] = clusterer.labels_
    clustered_fires = gdf_peatland_fires[gdf_peatland_fires['cluster_id'] != -1]

    # 4. Process and save the resulting incidents
    if clustered_fires.empty:
        logging.info("No significant fire incidents were formed after clustering. Exiting.")
        return
        
    all_incidents = []
    for cluster_id_num in sorted(clustered_fires['cluster_id'].unique()):
        cluster_gdf = clustered_fires[clustered_fires['cluster_id'] == cluster_id_num]
        centroid = cluster_gdf.geometry.union_all().centroid
        incident_data = {
            "cluster_id": f"fire_cluster_{run_date_str.replace('-', '')}_{cluster_id_num}",
            "point_count": len(cluster_gdf), "centroid_latitude": centroid.y,
            "centroid_longitude": centroid.x,
            "hotspots": json.loads(cluster_gdf.copy().to_json())['features']
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
    message_json = json.dumps({
        "status": "incidents_detected", "incident_count": len(all_incidents),
        "run_date": run_date_str, "output_path": f"gs://{GCS_BUCKET_NAME}/{output_blob_path}",
        "completion_time": datetime.now(timezone.utc).isoformat()
    })
    publisher.publish(topic_path, data=message_json.encode('utf-8')).result()
    logging.info("Successfully published completion signal.")
    logging.info("Incident Detector function finished successfully.")

if __name__ == "__main__":
    print("--- Running Incident Detector (Production Logic) locally ---")
    os.environ['GCP_PROJECT_ID'] = 'haryo-kebakaran'
    os.environ['GCS_BUCKET_NAME'] = 'fire-app-bucket'
    if 'FIRMS_API_KEY' not in os.environ:
        os.environ['FIRMS_API_KEY'] = '0331973a7ee830ca7f026493faaa367a'
    incident_detector_cloud_function(event=None, context=None)
    print("--- Local run of Incident Detector finished ---")
