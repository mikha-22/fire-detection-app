# src/cloud_functions/image_processor/main.py

import os
import json
import base64
import logging
from datetime import datetime

from google.cloud import aiplatform, storage
from src.satellite_imagery_acquirer.acquirer import SatelliteImageryAcquirer

# --- Configuration ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
VERTEX_AI_MODEL_NAME = os.environ.get("VERTEX_AI_OBJECT_DETECTION_MODEL")
BATCH_PREDICTION_MACHINE_TYPE = "n1-standard-4"
VERTEX_AI_BATCH_SERVICE_ACCOUNT = "fire-app-vm-service-account@haryo-kebakaran.iam.gserviceaccount.com"
INCIDENTS_GCS_PREFIX = "incidents" # Folder where incident data is stored

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Helper for logging structured data ---
def _log_json(severity: str, message: str, **kwargs):
    log_entry = {
        "severity": severity.upper(),
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "component": "ImageProcessor",
        **kwargs
    }
    print(json.dumps(log_entry))

def image_processor_cloud_function(event, context):
    _log_json("INFO", "Image Processor function triggered.")

    run_date = datetime.utcnow().strftime('%Y-%m-%d')
    run_timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    job_id = f"job_{run_timestamp}"
    
    _log_json("INFO", f"Processing for run_date: {run_date}, job_id: {job_id}")

    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    incidents_blob_name = f"{INCIDENTS_GCS_PREFIX}/{run_date}/detected_incidents.jsonl"
    
    try:
        incidents_content = bucket.blob(incidents_blob_name).download_as_string().decode('utf-8')
        incidents = [json.loads(line) for line in incidents_content.strip().split('\n')]
        _log_json("INFO", f"Successfully loaded {len(incidents)} incidents from GCS for date {run_date}.")
    except Exception as e:
        _log_json("ERROR", f"Failed to read incidents file from GCS: gs://{GCS_BUCKET_NAME}/{incidents_blob_name}. Error: {e}")
        return

    if not incidents:
        _log_json("WARNING", "Incidents file was empty or could not be parsed. Exiting.")
        return

    regions_to_acquire = []
    for incident in incidents:
        cluster_id = incident.get("cluster_id")
        lat = incident.get("centroid_latitude")
        lon = incident.get("centroid_longitude")
        if not all([cluster_id, lat, lon]): continue

        point_count = incident.get("point_count", 0)
        KM_PER_DEGREE = 111.0 

        if point_count <= 10:
            analysis_level = "Small"
            bbox_size_km = 10.0
        elif point_count <= 50:
            analysis_level = "Medium"
            bbox_size_km = 20.0
        else:
            analysis_level = "Large"
            bbox_size_km = 40.0
        
        bbox_size_degrees = bbox_size_km / KM_PER_DEGREE

        _log_json("INFO", f"Determined analysis level for cluster {cluster_id}",
                  details={
                      "cluster_id": cluster_id,
                      "point_count": point_count,
                      "analysis_level": analysis_level,
                      "bbox_km": bbox_size_km,
                      "bbox_degrees": round(bbox_size_degrees, 4)
                  })
        
        cluster_bbox = [lon - bbox_size_degrees/2, lat - bbox_size_degrees/2, lon + bbox_size_degrees/2, lat + bbox_size_degrees/2]
        regions_to_acquire.append({"id": cluster_id, "name": f"Incident area for {cluster_id}", "bbox": cluster_bbox})

    try:
        imagery_acquirer = SatelliteImageryAcquirer(gcs_bucket_name=GCS_BUCKET_NAME)
        gcs_image_uris = imagery_acquirer.acquire_and_export_imagery(regions_to_acquire, acquisition_date=run_date)

        if not gcs_image_uris:
            _log_json("ERROR", "Failed to acquire any satellite images for the batch.")
            return
        
        _log_json("INFO", f"Successfully initiated export for {len(gcs_image_uris)} of {len(regions_to_acquire)} requested images.")

        clusters_for_batch = []
        for region in regions_to_acquire:
            cluster_id = region["id"]
            if cluster_id in gcs_image_uris:
                cluster_data = {
                    "cluster_id": cluster_id,
                    "gcs_image_uri": gcs_image_uris[cluster_id],
                    "image_bbox": region["bbox"]
                }
                clusters_for_batch.append(cluster_data)
        
        if not clusters_for_batch:
            _log_json("ERROR", "No images were successfully processed to create a batch input file.")
            return

        jsonl_lines = []
        for cluster in clusters_for_batch:
            instance = {
                "instance_id": cluster["cluster_id"],
                "clusters": [cluster]
            }
            jsonl_lines.append(json.dumps(instance))
        
        jsonl_content = "\n".join(jsonl_lines)
        input_filename = f"incident_inputs/{run_date}/{job_id}/input.jsonl"
        blob = bucket.blob(input_filename)
        blob.upload_from_string(jsonl_content)
        _log_json("INFO", f"Uploaded {len(jsonl_lines)} instances to gs://{GCS_BUCKET_NAME}/{input_filename}")

        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
        model = aiplatform.Model.list(filter=f'display_name="{VERTEX_AI_MODEL_NAME}"')[0]

        job_display_name = f"batch_prediction_{run_date}_{job_id}"
        output_uri_prefix = f"gs://{GCS_BUCKET_NAME}/incident_outputs/{run_date}/{job_id}/"
        
        job = model.batch_predict(
            job_display_name=job_display_name,
            gcs_source=f"gs://{GCS_BUCKET_NAME}/{input_filename}",
            gcs_destination_prefix=output_uri_prefix,
            sync=False,
            machine_type=BATCH_PREDICTION_MACHINE_TYPE,
            service_account=VERTEX_AI_BATCH_SERVICE_ACCOUNT,
        )
        
        # This log message is now slightly different because job.resource_name is not yet available
        _log_json("INFO", "Successfully submitted Vertex AI Batch Prediction job.",
                  job_display_name=job_display_name,
                  job_id=job_id,
                  output_location=output_uri_prefix)
        
        # --- FIX: REMOVED THE CODE THAT CAUSED THE CRASH ---
        # The following lines were removed because 'job.resource_name' is not
        # available immediately after an async call. The manifest file is now the
        # primary way to track job status.

        # Also create/update a manifest file that lists all jobs for this date
        manifest_filename = f"incident_outputs/{run_date}/manifest.json"
        manifest_blob = bucket.blob(manifest_filename)
        
        try:
            existing_manifest = json.loads(manifest_blob.download_as_string())
        except:
            existing_manifest = {"run_date": run_date, "jobs": []}
        
        existing_manifest["jobs"].append({
            "job_id": job_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "vertex_ai_job_display_name": job_display_name, # Storing the display name
            "status": "submitted"
        })
        
        manifest_blob.upload_from_string(json.dumps(existing_manifest, indent=2))
        _log_json("INFO", f"Updated job manifest for date {run_date} with new job {job_id}")

    except Exception as e:
        _log_json("ERROR", f"An error occurred during batch image processing. Error: {e}", exc_info=True)
        raise

    _log_json("INFO", "Image Processor function finished for the batch.")
