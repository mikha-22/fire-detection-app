# src/cloud_functions/image_processor/main.py

import os
import json
import logging
from datetime import datetime, timezone

from google.cloud import firestore, aiplatform, storage
from src.satellite_imagery_acquirer.acquirer import SatelliteImageryAcquirer
from src.common.config import GCS_PATHS, FILE_NAMES, GCS_BUCKET_NAME

# --- Configuration ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")
VERTEX_AI_MODEL_NAME = os.environ.get("VERTEX_AI_OBJECT_DETECTION_MODEL")
BATCH_PREDICTION_MACHINE_TYPE = "n1-standard-4"
VERTEX_AI_BATCH_SERVICE_ACCOUNT = "fire-app-vm-service-account@haryo-kebakaran.iam.gserviceaccount.com"
FIRESTORE_DATABASE_ID = "fire-app-firestore-db"
BOUNDING_BOX_PADDING_DEGREES = 0.1

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
db = firestore.Client(database=FIRESTORE_DATABASE_ID)

def _log_json(severity: str, message: str, **kwargs):
    log_entry = {
        "severity": severity.upper(), "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "component": "ImageProcessor", **kwargs
    }
    print(json.dumps(log_entry))

def check_and_set_processed_flag(event_id: str) -> bool:
    doc_ref = db.collection('processed_events').document(event_id)
    @firestore.transactional
    def _check_and_set(transaction, doc_ref):
        doc = doc_ref.get(transaction=transaction)
        if doc.exists:
            _log_json("WARNING", f"Duplicate event ID detected: {event_id}.")
            return False
        else:
            transaction.set(doc_ref, {'processed_at': datetime.now(timezone.utc)})
            return True
    transaction = db.transaction()
    return _check_and_set(transaction, doc_ref)

def image_processor_cloud_function(event, context):
    _log_json("INFO", "Image Processor function triggered.")

    if not os.environ.get("IS_LOCAL_TEST"):
        if not context or not context.event_id:
            _log_json("ERROR", "Cannot ensure idempotency: context.event_id is missing.")
            return
        if not check_and_set_processed_flag(context.event_id):
            return

    run_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    run_timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    job_id = f"job_{run_timestamp}"
    
    _log_json("INFO", f"Processing for run_date: {run_date}, job_id: {job_id}")

    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    incidents_blob_path = f"{GCS_PATHS['INCIDENTS_DETECTED']}/{run_date}/{FILE_NAMES['incident_data']}"

    try:
        incidents = [json.loads(line) for line in bucket.blob(incidents_blob_path).download_as_string().decode('utf-8').strip().split('\n')]
        _log_json("INFO", f"Successfully loaded {len(incidents)} incidents from GCS.")
    except Exception as e:
        _log_json("ERROR", f"Failed to read incidents file: {incidents_blob_path}", error=str(e))
        return

    if not incidents:
        _log_json("WARNING", "Incidents file was empty or could not be parsed.")
        return

    regions_to_acquire = []
    for incident in incidents:
        cluster_id, hotspots = incident.get("cluster_id"), incident.get("hotspots", [])
        if not all([cluster_id, hotspots]): continue
        
        longitudes = [h['geometry']['coordinates'][0] for h in hotspots]
        latitudes = [h['geometry']['coordinates'][1] for h in hotspots]
        
        cluster_bbox = [
            min(longitudes) - BOUNDING_BOX_PADDING_DEGREES, min(latitudes) - BOUNDING_BOX_PADDING_DEGREES,
            max(longitudes) + BOUNDING_BOX_PADDING_DEGREES, max(latitudes) + BOUNDING_BOX_PADDING_DEGREES
        ]
        regions_to_acquire.append({"id": cluster_id, "name": f"Incident {cluster_id}", "bbox": cluster_bbox})

    try:
        imagery_acquirer = SatelliteImageryAcquirer(gcs_bucket_name=GCS_BUCKET_NAME)
        gcs_uris_by_cluster = imagery_acquirer.acquire_and_export_imagery(regions_to_acquire, acquisition_date=run_date)

        if not gcs_uris_by_cluster:
            _log_json("ERROR", "Failed to acquire any satellite images.")
            return

        instances_for_batch = []
        for region in regions_to_acquire:
            cluster_id = region["id"]
            if cluster_id in gcs_uris_by_cluster and gcs_uris_by_cluster[cluster_id]:
                for image_info in gcs_uris_by_cluster[cluster_id]:
                    instances_for_batch.append({
                        "instance_id": f"{cluster_id}_{image_info['source']}",
                        "clusters": [{
                            "cluster_id": cluster_id,
                            "gcs_image_uri": image_info['uri'],
                            "image_bbox": region["bbox"],
                            "source": image_info['source']
                        }]
                    })
        
        if not instances_for_batch:
            _log_json("ERROR", "No instances to process after image acquisition.")
            return

        jsonl_content = "\n".join([json.dumps(inst) for inst in instances_for_batch])
        batch_input_path = f"{GCS_PATHS['PREDICTION_JOBS']}/{run_date}/{job_id}/{FILE_NAMES['batch_input']}"
        bucket.blob(batch_input_path).upload_from_string(jsonl_content)

        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
        model = aiplatform.Model.list(filter=f'display_name="{VERTEX_AI_MODEL_NAME}"')[0]
        job_display_name = f"batch_prediction_{run_date}_{job_id}"
        output_uri_prefix = f"gs://{GCS_BUCKET_NAME}/{GCS_PATHS['PREDICTION_JOBS']}/{run_date}/{job_id}/{GCS_PATHS['JOB_RAW_OUTPUT']}/"
        
        job = model.batch_predict(
            job_display_name=job_display_name, gcs_source=f"gs://{GCS_BUCKET_NAME}/{batch_input_path}",
            gcs_destination_prefix=output_uri_prefix, sync=False, machine_type=BATCH_PREDICTION_MACHINE_TYPE,
            service_account=VERTEX_AI_BATCH_SERVICE_ACCOUNT
        )
        _log_json("INFO", "Submitted Vertex AI Batch Prediction job.", job_display_name=job_display_name, job_id=job_id)

        job_metadata = {
            "job_id": job_id, "run_date": run_date, "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "submitted",
            "vertex_ai": {"job_display_name": job_display_name, "model_name": VERTEX_AI_MODEL_NAME, "output_location": output_uri_prefix},
            "input": {"incident_count": len(incidents), "instance_count": len(instances_for_batch), "input_path": f"gs://{GCS_BUCKET_NAME}/{batch_input_path}"}
        }
        metadata_path = f"{GCS_PATHS['PREDICTION_JOBS']}/{run_date}/{job_id}/{FILE_NAMES['job_metadata']}"
        bucket.blob(metadata_path).upload_from_string(json.dumps(job_metadata, indent=2))
        
        manifest_path = f"{GCS_PATHS['PREDICTION_JOBS']}/{run_date}/{FILE_NAMES['job_manifest']}"
        try:
            manifest_blob = bucket.blob(manifest_path)
            existing_manifest = json.loads(manifest_blob.download_as_string())
        except Exception:
            existing_manifest = {"run_date": run_date, "jobs": []}
        
        existing_manifest["jobs"].append({"job_id": job_id, "created_at": job_metadata["created_at"], "status": "submitted"})
        bucket.blob(manifest_path).upload_from_string(json.dumps(existing_manifest, indent=2))
        _log_json("INFO", f"Updated job manifest with new job {job_id}")

    except Exception as e:
        _log_json("ERROR", "An error occurred during batch image processing.", error=str(e), exc_info=True)
        raise

    _log_json("INFO", "Image Processor function finished.")

if __name__ == "__main__":
    print("--- Running Image Processor locally (idempotency check is bypassed) ---")
    os.environ['IS_LOCAL_TEST'] = 'true'
    os.environ['GCP_PROJECT_ID'] = 'haryo-kebakaran'
    os.environ['GCP_REGION'] = 'asia-southeast2'
    os.environ['GCS_BUCKET_NAME'] = 'fire-app-bucket'
    os.environ['VERTEX_AI_OBJECT_DETECTION_MODEL'] = 'wildfire-cpr-predictor-model'
    image_processor_cloud_function(event=None, context=None)
    print("--- Local run of Image Processor finished ---")
