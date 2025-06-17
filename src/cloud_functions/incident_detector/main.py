# src/cloud_functions/incident_detector/main.py

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional
import pytz 

# Data Handling and Geospatial
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN

# GCP and External Services
from google.cloud import pubsub, storage

# --- Local Application Imports ---
from src.firms_data_retriever.retriever import FirmsDataRetriever
from src.jaxa_data_retriever.retriever import JaxaDataRetriever
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
INDONESIA_BBOX = [95.0, -11.0, 141.0, 6.0]

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Data Standardization Utility ---
def standardize_hotspot_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    if source_name == 'JAXA': pass
    else: df.rename(columns={'acq_date': 'acq_date', 'acq_time': 'acq_time'}, inplace=True)
    df['source'] = source_name
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        logging.error(f"Latitude/Longitude columns not found in {source_name} data.")
        return pd.DataFrame()
    try:
        if source_name == 'FIRMS':
            time_str = df['acq_time'].astype(str).str.zfill(4)
            df['acq_datetime'] = pd.to_datetime(df['acq_date'] + ' ' + time_str, format='%Y-%m-%d %H%M', utc=True)
        elif source_name == 'JAXA':
            df['acq_datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']]).dt.tz_localize('UTC')
    except Exception as e:
        logging.warning(f"Datetime parsing error for {source_name}: {e}.")
        df['acq_datetime'] = pd.NaT
    df.dropna(subset=['latitude', 'longitude', 'acq_datetime'], inplace=True)
    for col in ['latitude', 'longitude', 'frp_mean', 'frp_max', 'confidence', 'n_detections']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    canonical_cols = ['latitude', 'longitude', 'acq_datetime', 'source', 'frp_mean', 'frp_max', 'confidence', 'n_detections']
    return df[[col for col in canonical_cols if col in df.columns]]

# --- Main Cloud Function ---
def incident_detector_cloud_function(event, context, run_date_str: Optional[str] = None):
    
    if run_date_str:
        # This branch is now only for special, manual test cases
        target_date_for_processing = datetime.strptime(run_date_str, '%Y-%m-%d')
        logging.info(f"--- RUNNING IN OVERRIDE MODE FOR DATE: {run_date_str} ---")
    else:
        # This is the production logic that will now run during local tests
        jakarta_tz = pytz.timezone('Asia/Jakarta')
        now_in_jakarta = datetime.now(jakarta_tz)
        target_date_for_processing = now_in_jakarta - timedelta(days=1)
        run_date_str = target_date_for_processing.strftime('%Y-%m-%d')
        logging.info(f"--- RUNNING IN PRODUCTION MODE ---")

    logging.info(f"Target processing date (Jakarta Time): {run_date_str}")
    
    run_date_utc = datetime.strptime(run_date_str, '%Y-%m-%d').replace(tzinfo=pytz.utc)

    all_hotspots_dfs = []

    # Ingest from FIRMS
    try:
        firms_retriever = FirmsDataRetriever(api_key=FIRMS_API_KEY, base_url="https://firms.modaps.eosdis.nasa.gov/api/area/csv/", sensors=["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"])
        firms_df = firms_retriever.get_and_filter_firms_data([{"id": "indonesia", "bbox": INDONESIA_BBOX}], date_str=run_date_str)
        if not firms_df.empty:
            all_hotspots_dfs.append(standardize_hotspot_df(firms_df, 'FIRMS'))
    except Exception as e:
        logging.error(f"Failed to process FIRMS data: {e}", exc_info=True)

    # Ingest from JAXA
    try:
        jaxa_retriever = JaxaDataRetriever()
        jaxa_df = jaxa_retriever.get_l3_hourly_data(target_date=run_date_utc)
        if not jaxa_df.empty:
            all_hotspots_dfs.append(standardize_hotspot_df(jaxa_df, 'JAXA'))
    except Exception as e:
        logging.error(f"Failed to process JAXA data: {e}", exc_info=True)

    if not all_hotspots_dfs:
        logging.warning("No hotspot data could be retrieved from any source. Exiting.")
        return

    fused_df = pd.concat(all_hotspots_dfs, ignore_index=True)
    fused_df.drop_duplicates(subset=['latitude', 'longitude', 'acq_datetime'], inplace=True)
    logging.info(f"Fused {len(fused_df)} unique hotspots from {len(all_hotspots_dfs)} sources.")

    hotspots_gdf = gpd.GeoDataFrame(fused_df, geometry=gpd.points_from_xy(fused_df.longitude, fused_df.latitude), crs="EPSG:4326")
    try:
        peatlands = gpd.read_file(PEATLAND_SHP_PATH).to_crs(epsg=3857)
        peatlands['geometry'] = peatlands.geometry.buffer(PEATLAND_BUFFER_METERS)
        gdf_peatland_fires = gpd.sjoin(hotspots_gdf, peatlands.to_crs(hotspots_gdf.crs), how="inner", predicate='within')
    except Exception as e:
        logging.error(f"CRITICAL: Peatland filtering failed: {e}", exc_info=True)
        return

    if gdf_peatland_fires.empty:
        logging.info("No hotspots on or near peatlands. Exiting.")
        return
    logging.info(f"Found {len(gdf_peatland_fires)} fire points on peatlands to be clustered.")

    coords_radians = np.radians(gdf_peatland_fires[['latitude', 'longitude']].values)
    clusterer = DBSCAN(eps=DBSCAN_EPS_RAD, min_samples=MIN_SAMPLES_PER_CLUSTER, algorithm='ball_tree', metric='haversine').fit(coords_radians)
    gdf_peatland_fires['cluster_id'] = clusterer.labels_
    clustered_fires = gdf_peatland_fires[gdf_peatland_fires['cluster_id'] != -1]

    if clustered_fires.empty:
        logging.info("No significant fire incidents formed after clustering. Exiting.")
        return

    all_incidents = []
    for cid in sorted(clustered_fires['cluster_id'].unique()):
        cluster_gdf = clustered_fires[clustered_fires['cluster_id'] == cid]
        centroid = cluster_gdf.geometry.union_all().centroid
        incident_data = {
            "cluster_id": f"fire_cluster_{run_date_str.replace('-', '')}_{cid}",
            "point_count": len(cluster_gdf), "centroid_latitude": centroid.y, "centroid_longitude": centroid.x,
            "hotspots": json.loads(cluster_gdf.to_json(default=str))['features']
        }
        all_incidents.append(incident_data)

    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    output_blob_path = f"{GCS_PATHS['INCIDENTS_DETECTED']}/{run_date_str}/{FILE_NAMES['incident_data']}"
    jsonl_content = "\n".join([json.dumps(incident) for incident in all_incidents])
    bucket.blob(output_blob_path).upload_from_string(jsonl_content, content_type='application/jsonl')
    logging.info(f"Successfully wrote {len(all_incidents)} incidents to gs://{GCS_BUCKET_NAME}/{output_blob_path}")
    
    publisher = pubsub.PublisherClient()
    topic_path = publisher.topic_path(GCP_PROJECT_ID, OUTPUT_TOPIC_NAME)
    message_json = json.dumps({"status": "incidents_detected", "incident_count": len(all_incidents), "run_date": run_date_str})
    publisher.publish(topic_path, data=message_json.encode('utf-8')).result()
    logging.info("Successfully published completion signal.")
    logging.info("Incident Detector function finished successfully.")

if __name__ == "__main__":
    print("--- Running Incident Detector in LOCAL PRODUCTION MODE ---")
    print("--- This will use timezone-aware logic to determine yesterday's date ---")
    os.environ['GCP_PROJECT_ID'] = 'haryo-kebakaran'
    os.environ['GCS_BUCKET_NAME'] = 'fire-app-bucket'
    if 'FIRMS_API_KEY' not in os.environ:
        os.environ['FIRMS_API_KEY'] = 'your_firms_api_key_here'

    # --- KEY CHANGE: Call the function without a date to test the production logic ---
    incident_detector_cloud_function(event=None, context=None)

    print("--- Local run of Incident Detector finished ---")
