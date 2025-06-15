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

# --- FIX: Import both the class and the sensor list ---
from src.firms_data_retriever.retriever import FirmsDataRetriever, FIRMS_SENSORS

# --- Configuration ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME") 
FIRMS_API_KEY = os.environ.get("FIRMS_API_KEY")
OUTPUT_TOPIC_NAME = "wildfire-cluster-detected"
INCIDENTS_GCS_PREFIX = "incidents"
DESIRED_EPS_KM = 10
EARTH_RADIUS_KM = 6371
DBSCAN_EPS_RADIANS = DESIRED_EPS_KM / EARTH_RADIUS_KM
DBSCAN_MIN_SAMPLES = 3
PEATLAND_SHP_PATH = "src/geodata/Indonesia_peat_lands.shp"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def incident_detector_cloud_function(event, context):
    logging.info("Incident Detector function triggered.")

    if not GCS_BUCKET_NAME:
        logging.critical("GCS_BUCKET_NAME environment variable not set. Cannot proceed.")
        return

    run_date_str = datetime.utcnow().strftime('%Y-%m-%d')
    logging.info(f"Processing for run_date: {run_date_str}")

    # --- FIX: Pass the imported FIRMS_SENSORS list to the constructor ---
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

<<<<<<< HEAD
            try:
                # --- BUG FIX & SOLUTION ---
                # The line from Vertex AI contains a JSON object.
                full_output_line = json.loads(line)
                
                # The key from the predictor is 'predictions' (plural) and it contains a LIST.
                # We get this list to iterate through it.
                predictions_list = full_output_line.get('predictions', [])

                for prediction in predictions_list:
                    # The 'instance_id' in the prediction corresponds to our 'cluster_id'
                    cluster_id = prediction.get('instance_id')
                    
                    if not cluster_id:
                        logger.warning(f"Skipping a prediction because it was missing an 'instance_id': {prediction}")
                        continue
                        
                    input_data = input_metadata.get(cluster_id)
                    if not input_data:
                        logger.warning(f"Could not find matching input metadata for cluster_id '{cluster_id}'. Skipping.")
                        continue
=======
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
>>>>>>> parent of a4caff3 (.)

    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    output_blob_name = f"{INCIDENTS_GCS_PREFIX}/{run_date_str}/detected_incidents.jsonl"
    
    jsonl_content = "\n".join([json.dumps(incident) for incident in all_incidents])
    
    blob = bucket.blob(output_blob_name)
    blob.upload_from_string(jsonl_content, content_type='application/jsonl')
    logging.info(f"Successfully wrote {len(all_incidents)} incidents to gs://{GCS_BUCKET_NAME}/{output_blob_name}")

<<<<<<< HEAD
                    cluster_hotspots = hotspots_by_cluster.get(cluster_id, [])
                    firms_df = pd.DataFrame()
                    if cluster_hotspots:
                        hotspot_records = [h['properties'] for h in cluster_hotspots]
                        firms_df = pd.DataFrame.from_records(hotspot_records)
                    
                    visualizer = MapVisualizer()
                    final_map_image = visualizer.generate_fire_map(
                        base_image_bytes=image_bytes, 
                        image_bbox=image_bbox, 
                        ai_detections=[prediction], # Pass the single prediction object for this cluster
                        firms_hotspots_df=firms_df,
                        acquisition_date_str=run_date
                    )
                    
                    img_byte_arr = BytesIO()
                    final_map_image.save(img_byte_arr, format='PNG')
                    encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

                    folium_map_data.append({
                        "cluster_id": cluster_id,
                        "latitude": (image_bbox[1] + image_bbox[3]) / 2,
                        "longitude": (image_bbox[0] + image_bbox[2]) / 2,
                        "detected": prediction.get("detected"),
                        "confidence": prediction.get("confidence", 0),
                        "encoded_png": encoded_image
                    })
                # --- END OF FIX ---
=======
    publisher = pubsub.PublisherClient()
    topic_path = publisher.topic_path(GCP_PROJECT_ID, OUTPUT_TOPIC_NAME)
    
    notification_payload = {
        "status": "incidents_detected",
        "incident_count": len(all_incidents),
        "completion_time": datetime.utcnow().isoformat() + "Z"
    }
    message_json = json.dumps(notification_payload)
    message_bytes = message_json.encode('utf-8')

    try:
        publish_future = publisher.publish(topic_path, data=message_bytes)
        publish_future.result()
        logging.info(f"Successfully published completion signal for {run_date_str}.")
    except Exception as e:
        logging.error(f"Failed to publish notification message. Error: {e}")
>>>>>>> parent of a4caff3 (.)

    logging.info("Incident Detector function finished successfully.")
