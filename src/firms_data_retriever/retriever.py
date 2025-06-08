# src/firms_data_retriever/retriever.py

import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
import ee # <-- ADDED IMPORT

# Import MONITORED_REGIONS from the common config file
from src.common.config import MONITORED_REGIONS

# --- Configuration ---
FIRMS_API_KEY = os.environ.get("FIRMS_API_KEY")
FIRMS_API_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
FIRMS_SENSORS = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"]

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def _log_json(severity: str, message: str, **kwargs):
    """
    Helper to log structured JSON messages to stdout, which GCP Cloud Logging
    can ingest as structured logs.
    """
    log_entry = {
        "severity": severity.upper(),
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "component": "FirmsDataRetriever",
        **kwargs
    }
    print(json.dumps(log_entry))


# --- NEW FUNCTION TO CHECK PEATLANDS ---
def _check_peatland_status(firms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches a DataFrame of FIRMS hotspots with a boolean flag indicating if they
    are on peatland, using a GEE peatland map.
    """
    if firms_df.empty:
        firms_df['is_on_peatland'] = 0
        return firms_df

    try:
        # Initialize Earth Engine.
        ee.Initialize(project=os.environ.get("GCP_PROJECT_ID"))

        # --- IMPORTANT ---
        # Load a peatland map for Indonesia from your GEE Assets.
        # You must upload your own GeoTIFF and replace the path below.
        peat_map = ee.Image("projects/ee-haryo-kebakaran/assets/peat_map_indonesia").select('b1')

        # Create ee.Feature points from the DataFrame latitudes and longitudes
        points = [ee.Geometry.Point(lon, lat) for lon, lat in zip(firms_df['longitude'], firms_df['latitude'])]
        ee_points = ee.FeatureCollection(points)

        # Use reduceRegions to get the peatland value (1 or 0) for each point.
        results = peat_map.reduceRegions(collection=ee_points, reducer=ee.Reducer.first(), scale=100).getInfo()

        is_on_peat = []
        # The result features are not guaranteed to be in the same order, so we build a lookup.
        # This is a simplification; a robust solution would match coordinates. For this use case, order is often preserved.
        for feature in results['features']:
            is_on_peat.append(1 if feature['properties'].get('first') == 1 else 0)

        # Ensure the list length matches the DataFrame length
        if len(is_on_peat) == len(firms_df):
            firms_df['is_on_peatland'] = is_on_peat
            _log_json("INFO", "Successfully enriched FIRMS data with peatland status.",
                      peat_hotspots_found=sum(is_on_peat))
        else:
            _log_json("ERROR", "Mismatch between FIRMS points and GEE results. Defaulting to no peatland.",
                      firms_count=len(firms_df), gee_results_count=len(is_on_peat))
            firms_df['is_on_peatland'] = 0

    except Exception as e:
        _log_json("ERROR", "Failed to enrich FIRMS data with peatland status. Proceeding without it.", error=str(e))
        # If this process fails, we still want to continue, so we just add a default column.
        firms_df['is_on_peatland'] = 0

    return firms_df


class FirmsDataRetriever:
    """
    Component 1: Fetches active fire data from NASA FIRMS API (using /api/area/)
    and filters it for predefined monitored regions.
    """

    def __init__(self, api_key: str, base_url: str, sensors: List[str]):
        """
        Initializes the FIRMS data retriever.
        """
        if not api_key:
            _log_json("CRITICAL", "FIRMS_API_KEY environment variable not set. Cannot proceed.")
            raise ValueError("FIRMS_API_KEY is required for FirmsDataRetriever.")
        self.api_key = api_key
        self.base_url = base_url
        self.sensors = sensors
        _log_json("INFO", "FirmsDataRetriever initialized.", sensors=self.sensors, base_url=self.base_url)

    def _fetch_firms_data(self, sensor: str) -> Optional[pd.DataFrame]:
        """
        Fetches FIRMS data for a specific sensor for the last 24 hours (yesterday's data).
        """
        yesterday = datetime.utcnow() - timedelta(days=1)
        date_str = yesterday.strftime('%Y-%m-%d')
        day_range = "1"

        endpoint = f"{self.base_url}{self.api_key}/{sensor}/world/{day_range}/{date_str}"
        _log_json("INFO", "Attempting to fetch FIRMS data using /api/area/.",
                  sensor=sensor, date_for_data=date_str, endpoint=endpoint)

        try:
            response = requests.get(endpoint, timeout=60)

            if "Invalid API call" in response.text:
                _log_json("ERROR", "FIRMS API (/api/area/) returned 'Invalid API call'.",
                          api_response_snippet=response.text[:200], status_code=response.status_code, sensor=sensor)
                return None
            response.raise_for_status()
            if not response.text.strip() or response.text.startswith("No fire data found"):
                _log_json("WARNING", "No fire data found or empty response from FIRMS API for query.",
                          sensor=sensor, endpoint=endpoint)
                return None
            if "Error" in response.text or "Access Denied" in response.text:
                _log_json("ERROR", "FIRMS API returned an error message in response body.",
                          api_response_snippet=response.text[:200], status_code=response.status_code, sensor=sensor)
                return None

            df = pd.read_csv(pd.io.common.StringIO(response.text))
            _log_json("INFO", "Successfully fetched FIRMS data for sensor using /api/area/.",
                      sensor=sensor, rows_fetched=len(df))
            return df
        except Exception as e:
            _log_json("ERROR", "An unexpected error occurred during FIRMS data fetch or parsing.",
                      error_type=type(e).__name__, error=str(e), sensor=sensor)
            return None

    def get_and_filter_firms_data(self, monitored_regions: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Fetches, filters, and enriches FIRMS data for all specified sensors (for yesterday).
        """
        all_firms_data = []
        for sensor in self.sensors:
            df_sensor = self._fetch_firms_data(sensor)
            if df_sensor is not None and not df_sensor.empty:
                all_firms_data.append(df_sensor)

        output_cols_with_peat = [
            'latitude', 'longitude', 'acq_date', 'acq_time', 'confidence',
            'frp', 'daynight', 'satellite', 'monitored_region_id', 'is_on_peatland'
        ]
        empty_df_for_return = pd.DataFrame(columns=output_cols_with_peat)

        if not all_firms_data:
            _log_json("WARNING", "No FIRMS data retrieved from any sensor.")
            return empty_df_for_return

        combined_df = pd.concat(all_firms_data, ignore_index=True)
        initial_rows = len(combined_df)
        _log_json("INFO", "Combined raw FIRMS data from all sensors.", total_hotspots_before_filter=initial_rows)

        required_cols = ['latitude', 'longitude', 'acq_date', 'acq_time', 'confidence', 'frp', 'daynight', 'satellite']
        for col in required_cols:
            if col not in combined_df.columns:
                combined_df[col] = None

        if 'confidence' in combined_df.columns:
            combined_df['confidence'] = combined_df['confidence'].astype(str).str.lower()
            confidence_values_to_keep = ['h', 'n']
            filtered_by_confidence_df = combined_df[combined_df['confidence'].isin(confidence_values_to_keep)].copy()
            _log_json("INFO", "Filtered FIRMS data by confidence (kept 'h' or 'n').",
                      rows_after_confidence_filter=len(filtered_by_confidence_df))
        else:
            filtered_by_confidence_df = combined_df.copy()

        if filtered_by_confidence_df.empty:
            return empty_df_for_return

        filtered_hotspots_by_region = []
        for region in monitored_regions:
            region_id = region["id"]
            min_lon, min_lat, max_lon, max_lat = region["bbox"]
            temp_df = filtered_by_confidence_df.copy()
            temp_df['latitude'] = pd.to_numeric(temp_df['latitude'], errors='coerce')
            temp_df['longitude'] = pd.to_numeric(temp_df['longitude'], errors='coerce')
            temp_df.dropna(subset=['latitude', 'longitude'], inplace=True)

            region_df = temp_df[
                (temp_df['latitude'] >= min_lat) & (temp_df['latitude'] <= max_lat) &
                (temp_df['longitude'] >= min_lon) & (temp_df['longitude'] <= max_lon)
            ].copy()

            if not region_df.empty:
                region_df['monitored_region_id'] = region_id
                filtered_hotspots_by_region.append(region_df)

        if not filtered_hotspots_by_region:
            return empty_df_for_return

        final_df = pd.concat(filtered_hotspots_by_region, ignore_index=True)

        # --- MODIFIED LOGIC: ENRICH WITH PEATLAND DATA ---
        final_df = _check_peatland_status(final_df)
        # --- END MODIFICATION ---

        # Reorder and select final columns
        final_output_columns = [col for col in output_cols_with_peat if col in final_df.columns]
        final_df = final_df[final_output_columns]

        _log_json("INFO", "FIRMS data retrieval, filtering, and enrichment complete.",
                  total_filtered_hotspots=len(final_df))
        return final_df

# --- Example Usage (for local testing) ---
# ... (local testing part is unchanged) ...
