# Automated Wildfire Detection & Monitoring System on GCP

This project implements an automated system for detecting and monitoring wildfires using Python and Google Cloud Platform (GCP) services. The system ingests data from external APIs (NASA FIRMS) and satellite imagery (Google Earth Engine), processes it using an AI model on Vertex AI, generates map visualizations and metadata, and presents the information on a simple web dashboard.

## Table of Contents

- [Overall Architecture Overview](#overall-architecture-overview)
- [Key Technologies](#key-technologies)
- [Project Structure](#project-structure)
- [Data Schemas](#data-schemas)
- [Setup and Deployment](#setup-and-deployment)
  - [Prerequisites](#prerequisites)
  - [Configuration](#configuration)
  - [Deploying Cloud Functions](#deploying-cloud-functions)
  - [Deploying AI Model to Vertex AI](#deploying-ai-model-to-vertex-ai)
  - [Deploying Web Dashboard (Cloud Run)](#deploying-web-dashboard-cloud-run)
  - [Setting up Cloud Scheduler](#setting-up-cloud-scheduler)
- [Local Development and Testing](#local-development-and-testing)
- [Cross-Cutting Concerns](#cross-cutting-concerns)
- [Future Enhancements](#future-enhancements)

## Overall Architecture Overview

The system is composed of several layers and components:

1.  **Data Ingestion & Acquisition Layer:**
    *   **Component 1: FIRMS Data Retriever & Filter (`src/firms_data_retriever/retriever.py`):** Fetches and filters active fire data from the NASA FIRMS API for predefined monitored regions.
    *   **Component 2: Satellite Imagery Acquirer (MVP) (`src/satellite_imagery_acquirer/acquirer.py`):** Obtains standard RGB satellite imagery for monitored regions using the Google Earth Engine Python API and exports it to Google Cloud Storage (GCS).

2.  **AI-Powered Detection Layer (Vertex AI):**
    *   **Component 3: Fire Detection Model & Inference Service (`src/ml_model/`):** A custom PyTorch model (currently a dummy model for MVP) served via TorchServe. It's packaged into a `.mar` file and intended for use with Vertex AI Batch Prediction using a pre-built PyTorch serving container. The handler (`handler.py`) loads images from GCS and performs preprocessing.

3.  **Output Generation Layer:**
    *   **Component 4: Map Visualizer (`src/map_visualizer/visualizer.py`):** Creates map images overlaying AI detections and FIRMS hotspots on satellite imagery using the Pillow library.

4.  **Orchestration, Finalization & Scheduling Layer (Asynchronous):**
    *   **Component 5a (Cloud Function 1 - Pipeline Initiator - `src/cloud_functions/pipeline_initiator/main.py`):**
        *   Triggered daily by Cloud Scheduler (via Pub/Sub).
        *   Orchestrates data acquisition (Components 1 & 2).
        *   Prepares input JSONL data for the AI model and uploads it to GCS.
        *   Submits a Vertex AI Batch Prediction job (for Component 3) asynchronously.
        *   **Note:** This function does *not* directly set up Pub/Sub notifications from the Vertex AI job due to client library limitations for `NotificationSpec`. A Cloud Logging Sink must be configured separately to forward Vertex AI Batch Prediction job completion events to a Pub/Sub topic.
    *   **Component 5b (Cloud Function 2 - Result Processor & Finalizer - `src/cloud_functions/result_processor/main.py`):**
        *   Triggered by a Pub/Sub notification upon Vertex AI Batch Job completion (via the aforementioned Cloud Logging Sink).
        *   Fetches AI prediction results from GCS.
        *   Invokes map visualization (Component 4).
        *   Assembles final metadata (`wildfire_status_latest.json`).
        *   Stores map images and the final JSON metadata in GCS.

5.  **Presentation Layer (MVP):**
    *   **Component 6 (Flask Web Application on Cloud Run - *Not yet implemented in provided files*):** A simple read-only dashboard that will display the latest status by reading `wildfire_status_latest.json` from GCS.

The data flow is event-driven, primarily orchestrated by the two Cloud Functions and Pub/Sub messages.

## Key Technologies

*   **Python 3.11**
*   **Google Cloud Platform:**
    *   Cloud Functions (Gen2)
    *   Vertex AI (Batch Prediction, Custom Models using Pre-built Containers, Model Registry)
    *   Cloud Storage (GCS)
    *   Pub/Sub
    *   Cloud Scheduler
    *   Secret Manager (planned for API key management)
    *   Cloud Run (for the web dashboard)
    *   Cloud Build (for CI/CD of Cloud Functions and Docker images)
    *   Artifact Registry (for Docker images)
*   **External Data APIs:**
    *   NASA FIRMS API
    *   Google Earth Engine Python API
*   **Machine Learning:**
    *   PyTorch
    *   TorchServe (for model serving definition, packaged in `.mar`)
*   **Core Python Libraries:**
    *   `requests` (for HTTP calls)
    *   `Pillow` (for image manipulation)
    *   `pandas` (for data handling, especially FIRMS)
    *   `Flask` (for the web dashboard - TBD)
*   **Testing:** `pytest` (planned)

## Project Structure

```.
├── .gcloudignore                 # Specifies files to ignore when deploying to GCP
├── .gitignore                    # Specifies intentionally untracked files that Git should ignore
├── archive_model.sh              # Script to package the PyTorch model into a .mar file
├── cloudbuild.yaml               # Cloud Build config for building the TorchServe Docker image (if custom container route is chosen)
├── cloudbuild-pipeline-initiator.yaml # Cloud Build config for deploying PipelineInitiatorCF
├── cloudbuild-result-processor.yaml   # Cloud Build config for deploying ResultProcessorCF
├── requirements.txt              # Top-level Python dependencies (primarily for local dev/testing tools)
├── src/                          # Source code directory
│   ├── cloud_functions/
│   │   ├── pipeline_initiator/
│   │   │   ├── main.py           # Cloud Function 1: Orchestrates data acquisition and starts AI job
│   │   │   └── requirements.txt  # Dependencies for Pipeline Initiator CF
│   │   └── result_processor/
│   │       ├── main.py           # Cloud Function 2: Processes AI results, generates maps, finalizes output
│   │       └── requirements.txt  # Dependencies for Result Processor CF
│   ├── common/
│   │   └── config.py           # Shared configuration (GCS bucket name, monitored regions)
│   ├── firms_data_retriever/
│   │   └── retriever.py        # Component 1: Fetches and filters NASA FIRMS data
│   ├── map_visualizer/
│   │   └── visualizer.py       # Component 4: Generates map images
│   ├── ml_model/                 # AI Model components
│   │   ├── Dockerfile            # Dockerfile for custom TorchServe container (alternative to pre-built)
│   │   ├── fire_detection_model.py # PyTorch model definition (currently dummy)
│   │   ├── handler.py            # TorchServe custom handler for preprocessing, inference, postprocessing
│   │   ├── model.mar             # Packaged model archive for TorchServe (generated by archive_model.sh)
│   │   ├── model.pth             # Saved PyTorch model weights (generated by fire_detection_model.py)
│   │   ├── register_vertex_model.py # Script to register the model with Vertex AI Model Registry (using pre-built container)
│   │   └── requirements.txt      # Dependencies for the ML model and handler
│   └── satellite_imagery_acquirer/
│       └── acquirer.py         # Component 2: Acquires satellite imagery via Google Earth Engine
└── README.md                     # This file

