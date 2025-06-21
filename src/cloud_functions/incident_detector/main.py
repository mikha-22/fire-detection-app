# src/cloud_functions/incident_detector/main.py

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
import pytz

import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN
import ee
from google.cloud import pubsub, storage

from src.firms_data_retriever.retriever import FirmsDataRetriever
from src.jaxa_data_retriever.retriever import JaxaDataRetriever
from src.weather_data_acquirer.acquirer import WeatherDataAcquirer
from src.air_quality_acquirer.acquirer import AirQualityAcquirer
from src.common.config import GCS_PATHS, FILE_NAMES, GCS_BUCKET_NAME

# --- Configuration ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
FIRMS_API_KEY = os.environ.get("FIRMS_API_KEY")
OUTPUT_TOPIC_NAME = "wildfire-cluster-detected"
PEATLAND_SHP_PATH = "src/geodata/Indonesia_peat_lands.shp"
PEATLAND_BUFFER_METERS = 1000
MIN_SAMPLES_PER_CLUSTER = 8
DBSCAN_MAX_DISTANCE_KM = 5
DBSCAN_EPS_RAD = DBSCAN_MAX_DISTANCE_KM / 6371
INDONESIA_BBOX = [95.0, -11.0, 141.0, 6.0]
AIR_QUALITY_BUFFER_KM = 50.0

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def _log_json(severity: str, message: str, **kwargs):
    log_entry = {"severity": severity.upper(), "message": message, "timestamp": datetime.now(timezone.utc).isoformat(), "component": "IncidentDetector", **kwargs}
    print(json.dumps(log_entry, default=str))

def standardize_hotspot_df(df: pd.DataFrame, source_name: str) -> Optional[pd.DataFrame]:
    if df.empty: return None
    df = df.copy()

    if source_name == 'FIRMS':
        if 'frp' in df.columns:
            df.rename(columns={'frp': 'frp_mean'}, inplace=True)
        confidence_map = {'l': 30, 'n': 75, 'h': 100}
        df['confidence'] = df['confidence'].str.lower().map(confidence_map)
        df['n_detections'] = 1
        df['acq_datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'].astype(str).str.zfill(4), format='%Y-%m-%d %H%M', utc=True)

    elif source_name == 'JAXA':
        df.rename(columns={'lat': 'latitude', 'lon': 'longitude', 'ave(frp)': 'frp_mean', 'max(frp)': 'frp_max', 'ave(confidence)': 'confidence', 'N': 'n_detections'}, inplace=True)
        df['acq_datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']]).dt.tz_localize('UTC')

    df['source'] = source_name
    
    golden_schema = ['latitude', 'longitude', 'acq_datetime', 'frp_mean', 'frp_max', 'confidence', 'n_detections', 'source']
    
    return df[[col for col in golden_schema if col in df.columns]]

def assess_fire_severity(point_count: int, weather: dict, air_quality: dict) -> dict:
    severity_score, factors = 0, []
    if point_count >= 15: severity_score += 3; factors.append("High hotspot density (>=15)")
    elif point_count >= 5: severity_score += 2; factors.append("Moderate hotspot density (>=5)")
    else: severity_score += 1; factors.append("Low hotspot density")
    
    weather_severity = "low"
    if weather and not weather.get("error"):
        if weather.get("relative_humidity_percent", 100) < 40:
            severity_score += 2; factors.append("Low humidity (<40%)")
            weather_severity = "high"
        if weather.get("wind_speed_kmh", 0) > 20:
            severity_score += 1; factors.append("High wind speed (>20km/h)")
            weather_severity = "high"
            
    aq_severity = air_quality.get("air_quality_severity", "unknown")
    if aq_severity == "severe": severity_score += 3; factors.append("Severe air quality impact")
    elif aq_severity == "high": severity_score += 2; factors.append("High air quality impact")
    elif aq_severity == "moderate": severity_score += 1; factors.append("Moderate air quality impact")

    if severity_score >= 8: overall = "critical"
    elif severity_score >= 6: overall = "severe"
    elif severity_score >= 4: overall = "high"
    elif severity_score >= 2: overall = "moderate"
    else: overall = "low"
    return {"overall_severity": overall, "severity_score": severity_score, "contributing_factors": factors, "details": {"hotspot_count": point_count, "weather_severity": weather_severity, "air_quality_severity": aq_severity}}

def incident_detector_cloud_function(event, context, run_date_str: Optional[str] = None):
    if not run_date_str:
        jakarta_tz = pytz.timezone('Asia/Jakarta')
        run_date_str = (datetime.now(jakarta_tz) - timedelta(days=1)).strftime('%Y-%m-%d')
    _log_json("INFO", f"Starting incident detection for Indonesian date: {run_date_str}")

    firms_retriever = FirmsDataRetriever(api_key=FIRMS_API_KEY, base_url="https://firms.modaps.eosdis.nasa.gov/api/area/csv/", sensors=["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"])
    jaxa_retriever = JaxaDataRetriever()
    weather_acquirer = WeatherDataAcquirer()
    air_quality_acquirer = AirQualityAcquirer()

    standardized_dfs = []
    try:
        firms_df = firms_retriever.get_data_for_indonesian_day([{"id": "indonesia", "bbox": INDONESIA_BBOX}], target_date_str=run_date_str)
        if firms_df is not None: standardized_dfs.append(standardize_hotspot_df(firms_df, 'FIRMS'))
    except Exception as e: _log_json("ERROR", f"Failed to process FIRMS data: {e}", exc_info=True)
    try:
        jaxa_df = jaxa_retriever.get_data_for_indonesian_day(target_date_str=run_date_str)
        if jaxa_df is not None: standardized_dfs.append(standardize_hotspot_df(jaxa_df, 'JAXA'))
    except Exception as e: _log_json("ERROR", f"Failed to process JAXA data: {e}", exc_info=True)

    if not standardized_dfs: _log_json("WARNING", "No hotspot data could be retrieved from any source. Exiting."); return
    
    fused_df = pd.concat(standardized_dfs, ignore_index=True)
    fused_df['lat_round'] = fused_df['latitude'].round(2)
    fused_df['lon_round'] = fused_df['longitude'].round(2)
    fused_df['time_key'] = fused_df['acq_datetime'].dt.floor('10min')
    fused_df.sort_values(by='confidence', ascending=False, inplace=True)
    fused_df.drop_duplicates(subset=['lat_round', 'lon_round', 'time_key'], keep='first', inplace=True)
    fused_df.drop(columns=['lat_round', 'lon_round', 'time_key'], inplace=True)
    
    _log_json("INFO", f"Fused and deduplicated to {len(fused_df)} unique hotspots")

    hotspots_gdf = gpd.GeoDataFrame(fused_df, geometry=gpd.points_from_xy(fused_df.longitude, fused_df.latitude), crs="EPSG:4326")
    try:
        peatlands = gpd.read_file(PEATLAND_SHP_PATH).to_crs(epsg=3857)
        peatlands['geometry'] = peatlands.geometry.buffer(PEATLAND_BUFFER_METERS)
        gdf_peatland_fires = gpd.sjoin(hotspots_gdf, peatlands.to_crs(hotspots_gdf.crs), how="inner", predicate='within')
    except Exception as e: _log_json("CRITICAL", f"Peatland filtering failed: {e}", exc_info=True); return
    if gdf_peatland_fires.empty: _log_json("INFO", "No hotspots on or near peatlands. Exiting."); return
    _log_json("INFO", f"Found {len(gdf_peatland_fires)} fire points on peatlands to be clustered")

    coords_radians = np.radians(gdf_peatland_fires[['latitude', 'longitude']].values)
    clusterer = DBSCAN(eps=DBSCAN_EPS_RAD, min_samples=MIN_SAMPLES_PER_CLUSTER, algorithm='ball_tree', metric='haversine').fit(coords_radians)
    gdf_peatland_fires['cluster_id'] = clusterer.labels_
    clustered_fires = gdf_peatland_fires[gdf_peatland_fires['cluster_id'] != -1]
    if clustered_fires.empty: _log_json("INFO", "No significant fire incidents formed after clustering. Exiting."); return
    
    num_clusters = len(clustered_fires['cluster_id'].unique())
    _log_json("INFO", f"Found {num_clusters} distinct fire clusters to enrich")

    all_incidents = []
    for cid in sorted(clustered_fires['cluster_id'].unique()):
        cluster_gdf = clustered_fires[clustered_fires['cluster_id'] == cid].copy()
        centroid = cluster_gdf.geometry.union_all().centroid
        latest_timestamp = cluster_gdf['acq_datetime'].max().to_pydatetime()
        
        try:
            weather_data = weather_acquirer.get_historical_weather_data(centroid.y, centroid.x, latest_timestamp)
            if weather_data.get("error") or any(pd.isna(v) for v in weather_data.values()):
                _log_json("WARNING", "Weather data acquisition returned an error or NaN values.", cluster_id=int(cid), result=weather_data)
                weather_data = {"error": "Weather data acquisition failed or returned incomplete data."}
        except Exception as e:
            weather_data = {"error": f"Unhandled exception in WeatherDataAcquirer: {e}"}
            _log_json("ERROR", "Unhandled exception in WeatherDataAcquirer.", cluster_id=int(cid), error=str(e), exc_info=True)

        try:
            air_quality_data = air_quality_acquirer.get_air_quality_for_incident(latitude=centroid.y, longitude=centroid.x, incident_timestamp=latest_timestamp, buffer_km=AIR_QUALITY_BUFFER_KM)
        except Exception as e:
            air_quality_data = {"error": f"Unhandled exception in AirQualityAcquirer: {e}"}
            _log_json("ERROR", "Unhandled exception in AirQualityAcquirer.", cluster_id=int(cid), error=str(e), exc_info=True)
        
        hotspot_columns_to_keep = ['latitude', 'longitude', 'acq_datetime', 'frp_mean', 'frp_max', 'confidence', 'n_detections', 'source', 'cluster_id']
        final_hotspot_df = cluster_gdf[[col for col in hotspot_columns_to_keep if col in cluster_gdf.columns]].copy()
        final_hotspot_df['id'] = final_hotspot_df.index.astype(str)
        
        # --- FIX: Convert datetime to string *before* creating the JSON ---
        final_hotspot_df['acq_datetime'] = final_hotspot_df['acq_datetime'].astype(str)
        hotspots_list = json.loads(final_hotspot_df.to_json(orient='records'))
        
        incident_data = {
            "cluster_id": f"fire_cluster_{run_date_str.replace('-', '')}_{cid}", "point_count": len(cluster_gdf),
            "centroid_latitude": round(centroid.y, 6), "centroid_longitude": round(centroid.x, 6),
            "detected_by": list(cluster_gdf['source'].unique()), "first_detected_utc": cluster_gdf['acq_datetime'].min().isoformat(),
            "last_detected_utc": latest_timestamp.isoformat(),
            "realtime_context": {"weather": weather_data, "air_quality": air_quality_data},
            "fire_severity_assessment": assess_fire_severity(point_count=len(cluster_gdf), weather=weather_data, air_quality=air_quality_data),
            "hotspots": hotspots_list
        }
        all_incidents.append(incident_data)

    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    output_blob_path = f"{GCS_PATHS['INCIDENTS_DETECTED']}/{run_date_str}/{FILE_NAMES['incident_data']}"
    jsonl_content = "\n".join([json.dumps(incident, indent=2, default=str) for incident in all_incidents])
    bucket.blob(output_blob_path).upload_from_string(jsonl_content, content_type='application/jsonl')
    _log_json("INFO", f"Successfully wrote {len(all_incidents)} enriched incidents to gs://{GCS_BUCKET_NAME}/{output_blob_path}")
    
    severity_counts = {inc["fire_severity_assessment"]["overall_severity"]: 0 for inc in all_incidents}
    for inc in all_incidents: severity_counts[inc["fire_severity_assessment"]["overall_severity"]] += 1
    _log_json("INFO", "Incident severity summary", **severity_counts)
    
    publisher = pubsub.PublisherClient()
    topic_path = publisher.topic_path(GCP_PROJECT_ID, OUTPUT_TOPIC_NAME)
    message_json = json.dumps({"status": "incidents_detected", "incident_count": len(all_incidents), "run_date": run_date_str, "severity_summary": severity_counts}, default=str)
    publisher.publish(topic_path, data=message_json.encode('utf-8')).result()
    _log_json("INFO", "Successfully published completion signal")

if __name__ == "__main__":
    print("--- Running Incident Detector with Full Enrichment ---")
    os.environ['GCP_PROJECT_ID'] = 'haryo-kebakaran'
    os.environ['GCS_BUCKET_NAME'] = 'fire-app-bucket'
    if 'FIRMS_API_KEY' not in os.environ:
        os.environ['FIRMS_API_KEY'] = 'your_firms_api_key_here'
    incident_detector_cloud_function(event=None, context=None, run_date_str="2025-05-20")
