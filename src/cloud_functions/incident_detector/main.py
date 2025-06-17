# src/cloud_functions/incident_detector/main.py

import os
import json
import logging
from datetime import datetime, timezone, timedelta
import io

# Data Handling and Geospatial
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN

# GCP and External Services
from google.cloud import pubsub, storage
import paramiko

# Local Application Imports
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
INDONESIA_BBOX = [95.0, -11.0, 141.0, 6.0]

JAXA_SFTP_HOST = 'ftp.ptree.jaxa.jp'
JAXA_SFTP_PORT = 2051
JAXA_WILDFIRE_BASE_PATH = '/pub/himawari/L2/WLF/'

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Utility Functions for Data Acquisition and Standardization ---

def fetch_jaxa_wildfire_data_sftp(lookback_hours: int = 24) -> pd.DataFrame:
    """
    Connects to the JAXA P-Tree SFTP server and downloads wildfire hotspot
    CSV files for the specified lookback period.
    """
    logging.info("Fetching JAXA wildfire data via SFTP...")

    username = "nathaniell.wijaya_gmail.com"
    password = "SP+wari8"

    all_hotspots = []
    try:
        with paramiko.Transport((JAXA_SFTP_HOST, JAXA_SFTP_PORT)) as transport:
            transport.connect(username=username, password=password)
            with paramiko.SFTPClient.from_transport(transport) as sftp:
                logging.info(f"Successfully connected to JAXA SFTP server: {JAXA_SFTP_HOST}")

                now_utc = datetime.now(timezone.utc)
                for i in range(lookback_hours):
                    target_time = now_utc - timedelta(hours=i)
                    remote_dir = f"{JAXA_WILDFIRE_BASE_PATH}{target_time.year}/{target_time.month:02d}/{target_time.day:02d}/"

                    try:
                        for filename in sftp.listdir(remote_dir):
                            if 'CSV' in filename and f"_{target_time.year}{target_time.month:02d}{target_time.day:02d}_{target_time.hour:02d}" in filename and filename.endswith('.csv'):
                                remote_filepath = remote_dir + filename
                                logging.info(f"Found matching JAXA file: {remote_filepath}")
                                with sftp.open(remote_filepath, 'r') as f:
                                    content = f.read().decode('utf-8')
                                    df = pd.read_csv(io.StringIO(content))
                                    all_hotspots.append(df)
                    except FileNotFoundError:
                        logging.debug(f"No directory found for {remote_dir}, skipping.")
                        continue
                    except Exception as e:
                        logging.warning(f"Could not process JAXA directory {remote_dir}: {e}")
                        continue

    except paramiko.AuthenticationException:
        logging.error("Authentication failed for JAXA SFTP. Check credentials.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An unexpected error occurred during JAXA SFTP operation: {e}")
        return pd.DataFrame()

    if not all_hotspots:
        logging.info("No JAXA wildfire data found for the lookback period.")
        return pd.DataFrame()

    combined_df = pd.concat(all_hotspots, ignore_index=True)
    logging.info(f"Successfully fetched and combined {len(combined_df)} hotspots from JAXA.")
    return combined_df


def standardize_hotspot_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Standardizes a hotspot dataframe to a common schema.
    """
    if df.empty:
        return df

    df = df.copy()
    rename_map = {}
    if source_name == 'FIRMS':
        rename_map = {
            'acq_date': 'acq_date',
            'acq_time': 'acq_time',
        }
    elif source_name == 'JAXA':
        # --- CORRECTED: Using exact column names from JAXA documentation ---
        rename_map = {
            '# Pixel-latitude[deg]': 'latitude',
            ' Pixel-longitude[deg]': 'longitude',
            'Date': 'acq_date',
            ' Time[UTC]': 'acq_time',
            ' FRP[MW]': 'frp'
        }

    df.rename(columns=rename_map, inplace=True)
    df['source'] = source_name

    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        logging.error(f"Latitude/Longitude columns not found in {source_name} data after mapping.")
        return pd.DataFrame()
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    try:
        if source_name == 'FIRMS':
            time_str = df['acq_time'].astype(str).str.zfill(4)
            df['acq_datetime'] = pd.to_datetime(df['acq_date'] + ' ' + time_str, format='%Y-%m-%d %H%M', utc=True)
        else:
             # JAXA uses YYYY-MM-DD and HH:MM:SS formats
             df['acq_datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'], format='%Y-%m-%d %H:%M:%S', utc=True)
    except Exception as e:
        logging.warning(f"Could not parse datetime for {source_name}: {e}. Setting to NaT.")
        df['acq_datetime'] = pd.NaT

    df.dropna(subset=['latitude', 'longitude', 'acq_datetime'], inplace=True)

    canonical_cols = ['latitude', 'longitude', 'acq_datetime', 'source']
    for col in ['frp', 'confidence']:
        if col in df.columns:
            canonical_cols.append(col)

    return df[[col for col in canonical_cols if col in df.columns]]


# --- Main Cloud Function ---

def incident_detector_cloud_function(event, context):
    logging.info("Incident Detector function triggered with data fusion.")
    run_date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    # 1. Ingest and Standardize Data from All Sources
    all_hotspots_dfs = []

    # Ingest from FIRMS
    try:
        logging.info("Fetching FIRMS data...")
        firms_retriever = FirmsDataRetriever(
            api_key=FIRMS_API_KEY,
            base_url="https://firms.modaps.eosdis.nasa.gov/api/area/csv/",
            sensors=["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]
        )
        regions_to_fetch = [{"id": "indonesia", "bbox": INDONESIA_BBOX}]
        firms_df = firms_retriever.get_and_filter_firms_data(regions_to_fetch)
        if not firms_df.empty:
            standardized_firms = standardize_hotspot_df(firms_df, 'FIRMS')
            all_hotspots_dfs.append(standardized_firms)
            logging.info(f"Successfully processed {len(standardized_firms)} hotspots from FIRMS.")
    except Exception as e:
        logging.error(f"Failed to process FIRMS data: {e}", exc_info=True)

    # Ingest from JAXA
    try:
        jaxa_df = fetch_jaxa_wildfire_data_sftp()
        if not jaxa_df.empty:
            standardized_jaxa = standardize_hotspot_df(jaxa_df, 'JAXA')
            all_hotspots_dfs.append(standardized_jaxa)
            logging.info(f"Successfully processed {len(standardized_jaxa)} hotspots from JAXA.")
    except Exception as e:
        logging.error(f"Failed to process JAXA data: {e}", exc_info=True)

    # 2. Fuse Data and Handle No-Data Scenario
    if not all_hotspots_dfs:
        logging.warning("No hotspot data could be retrieved from any source. Exiting.")
        return

    fused_df = pd.concat(all_hotspots_dfs, ignore_index=True)
    fused_df.drop_duplicates(subset=['latitude', 'longitude', 'acq_datetime'], inplace=True)
    logging.info(f"Fused {len(fused_df)} unique hotspots from {len(all_hotspots_dfs)} sources.")

    # 3. Filter by Peatlands and Cluster
    hotspots_gdf = gpd.GeoDataFrame(
        fused_df, geometry=gpd.points_from_xy(fused_df.longitude, fused_df.latitude), crs="EPSG:4326"
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
        logging.info(f"No hotspots found on or near peatlands. Exiting.")
        return
    logging.info(f"Found {len(gdf_peatland_fires)} fire points on peatlands to be clustered.")

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
            "point_count": len(cluster_gdf),
            "centroid_latitude": centroid.y,
            "centroid_longitude": centroid.x,
            "hotspots": json.loads(cluster_gdf.to_json(default=str))['features']
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
        os.environ['FIRMS_API_KEY'] = 'your_firms_api_key_here'
    incident_detector_cloud_function(event=None, context=None)
    print("--- Local run of Incident Detector finished ---")
