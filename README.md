
# Wildfire Detection and Analysis Pipeline

## 1. Overview

This repository contains a comprehensive, event-driven pipeline for detecting, assessing, and reporting on wildfire incidents, with a primary focus on peatland fires in Indonesia. The system is built on Google Cloud Platform services, leveraging Cloud Functions for orchestration, Vertex AI for machine learning, and various satellite data providers for real-time and historical analysis.

The core objective is to automate the process from raw satellite hotspot detection to a fully enriched, actionable incident report, complete with a risk assessment and visual map.

---

## 2. System Architecture & Pipeline

The system operates as a sequential, event-driven pipeline orchestrated by Google Cloud Pub/Sub and Cloud Functions.

 <!-- Placeholder for a diagram -->

**The flow is as follows:**

1.  **Trigger:** A Cloud Scheduler job publishes a message to the `wildfire-pipeline-initiator` topic on a daily schedule.

2.  **Stage 1: Incident Detection (`IncidentDetectorCF`)**
    *   This function is triggered by the Pub/Sub message.
    *   It fetches raw hotspot data from **FIRMS (VIIRS)** and **JAXA (Himawari-9)**.
    *   It standardizes and fuses the data from these disparate sources.
    *   It filters for hotspots located on or near Indonesian peatlands.
    *   It uses the **DBSCAN** algorithm to group hotspots into significant fire clusters.
    *   It enriches each cluster with **weather** and **air quality** data.
    *   The final enriched incident data is saved as `incidents.jsonl` to a GCS bucket.
    *   Upon completion, it publishes a message to the `wildfire-cluster-detected` topic.

3.  **Stage 2: Prediction Job Initiation (`PredictionJobInitiatorCF`)**
    *   Triggered by the message from Stage 1.
    *   Its sole responsibility is to start a **Vertex AI Batch Prediction Job**, pointing it to the newly created `incidents.jsonl` file and the registered ML model.

4.  **Stage 3: Heuristic Risk Prediction (Vertex AI)**
    *   This stage runs entirely on Vertex AI using a custom prediction routine.
    *   It loads the ML model (`src/ml_model/predictor.py`), which uses a configuration file (`src/ml_model/config.json`) to define risk criteria.
    *   It preprocesses the incident data (e.g., calculating average Fire Radiative Power) and predicts a `confidence_score` for each incident.
    *   Upon completion, Vertex AI saves the prediction results to GCS and publishes a message to the `wildfire-prediction-completed` topic.

5.  **Stage 4: Result Processing & Reporting (`ResultProcessorCF`)**
    *   Triggered by the message from Vertex AI.
    *   It fetches both the prediction scores and the original enriched incident data.
    *   It uses `folium` to generate a final `report.html` dashboard, an interactive map visualizing the fire clusters, their locations, and their predicted risk levels.
    *   The final report is saved to GCS.

---

## 3. Data Pipeline In-Depth

### 3.1. Hotspot Data Acquisition

The system fuses data from two complementary sources to maximize coverage and accuracy.

*   **Source A: FIRMS (VIIRS NRT)**
    *   **Sensors:** `VIIRS_SNPP_NRT`, `VIIRS_NOAA20_NRT`, `VIIRS_NOAA21_NRT`.
    *   **Technical Rationale:** Chosen for its **high spatial resolution (375m)**, which is crucial for accurate fire localization and reducing false positives. The Near Real-Time (NRT) feed ensures data availability within ~3-4 hours. We use three satellites to maximize daily coverage.
    *   **Fetch Implementation (`src/firms_data_retriever/retriever.py`):** The `get_data_for_indonesian_day` function fetches data for the two UTC days that overlap with the target Indonesian day (UTC+7) and then filters the results in memory to the precise 24-hour window.

*   **Source B: JAXA (Himawari-9 L3 Hourly)**
    *   **Sensor:** Advanced Himawari Imager (AHI).
    *   **Technical Rationale:** Chosen for its **high temporal resolution**. As a geostationary satellite, Himawari provides a new scan every 10 minutes, aggregated hourly. This is invaluable for tracking rapid fire spread and filling gaps between VIIRS passes.
    *   **Fetch Implementation (`src/jaxa_data_retriever/retriever.py`):** The `get_data_for_indonesian_day` function connects to the JAXA SFTP server and downloads the 24 specific hourly files that correspond to the Indonesian day window from the `/pub/himawari/L3/WLF/010/` directory structure.

### 3.2. Aggregation & Standardization

This phase, implemented in `incident_detector/main.py`, ensures data quality.

1.  **Standardization (`standardize_hotspot_df`):** This function creates a "Golden Schema" by renaming columns (`lat` -> `latitude`), creating a unified `acq_datetime` column, and normalizing values (e.g., converting FIRMS's confidence from strings to a numeric scale).
2.  **Deduplication:** After merging, the code sorts the combined data by `confidence` and removes duplicates based on a spatiotemporal key (rounded coordinates and a 10-minute time window), ensuring only the highest-quality detection for each event is kept.

### 3.3. Contextual Enrichment

*   **Weather Context (`src/weather_data_acquirer/acquirer.py`)**
    *   **Source:** Open-Meteo Real-Time Forecast API (`https://api.open-meteo.com/v1/forecast`).
    *   **Technical Rationale:** The forecast API provides the most up-to-the-hour data available, which is essential for assessing the current conditions of a newly detected fire.
    *   **Implementation:** Queries the API for the fire's centroid coordinates at the time of the latest hotspot detection.

*   **Atmospheric Impact (`src/air_quality_acquirer/acquirer.py`)**
    *   **Source:** Google Earth Engine, using Sentinel-5P Near Real-Time (NRTI) data (e.g., `COPERNICUS/S5P/NRTI/L3_CO`).
    *   **Technical Rationale:** The NRTI feed is used exclusively to ensure data is available within hours of detection, which is critical for a real-time pipeline.
    *   **Implementation:** Queries Earth Engine for the average pollutant value in a 50km radius around the fire's centroid.

---

## 4. Machine Learning Model Integration

*   **Input Schema:** The model receives a batch of JSON objects, where each object represents a single fire incident with the structure detailed in the final section of this document.
*   **Predictor Interface (`src/ml_model/predictor.py`):** The `WildfireHeuristicPredictor` class is the entry point for Vertex AI.
    *   The `preprocess()` method is where feature engineering should occur. It currently calculates the average Fire Radiative Power (`avg_frp`).
    *   The `predict()` method should contain the core model logic.
*   **Current Model:** A simple heuristic model is implemented in `FireRiskHeuristicModel`. It calculates a weighted score based on criteria defined in `src/ml_model/config.json`. This module is the primary target for replacement with a more sophisticated ML model.

---

## 5. Setup and Local Development

### 5.1. Prerequisites

*   Google Cloud SDK (gcloud CLI)
*   Python 3.11+
*   A Python virtual environment tool (e.g., `venv`)

### 5.2. Installation

1.  **Authenticate with GCP:**
    ```bash
    gcloud auth application-default login
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv fire-venv
    source fire-venv/bin/activate
    ```
3.  **Install dependencies:** The project uses a consolidated requirements file.
    ```bash
    pip install -r requirements.txt
    ```
4.  **Environment Variables:** Set the following environment variable for local testing:
    ```bash
    export FIRMS_API_KEY="your_firms_api_key_here"
    ```

### 5.3. Running Local Tests

*   **Incident Detector:** To run the entire data fetching and enrichment pipeline locally for a specific date:
    ```bash
    python -m src.cloud_functions.incident_detector.main
    ```
    *Note: The test date is hardcoded at the bottom of the file.*

*   **Result Processor:** To test the HTML report generation from mock data:
    ```bash
    python -m src.cloud_functions.result_processor.testlocal
    ```

*   **ML Model Logic:** To test the heuristic model's logic without a container:
    ```bash
    python -m src.ml_model.test_logic_offline
    ```

---

## 6. Deployment

The system is deployed using Cloud Build. Each major component has a corresponding `cloudbuild-*.yaml` file.

*   **Deploy Incident Detector:**
    ```bash
    gcloud builds submit --config=cloudbuild-incident-detector.yaml
    ```*   **Deploy Prediction Job Initiator:**
    ```bash
    gcloud builds submit --config=cloudbuild-prediction-job-initiator.yaml
    ```
*   **Deploy Result Processor:**
    ```bash
    gcloud builds submit --config=cloudbuild-result-processor.yaml
    ```
*   **Register ML Model:**
    ```bash
    python -m src.ml_model.register_vertex_model
    ```

---

## 7. Configuration

*   **GCS Paths & Filenames:** Global paths for data artifacts are defined in `src/common/config.py`.
*   **Service Environment Variables:** Runtime variables (API keys, project IDs, etc.) are set in the `--set-env-vars` flag within rach component's `cloudbuild-*.yaml` file.
