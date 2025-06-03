# src/firms_data_retriever/retriever.py

import os
import logging
import requests # Make sure this import is present
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json # For _log_json helper

# Import MONITORED_REGIONS from the common config file
from src.common.config import MONITORED_REGIONS

# --- Configuration ---
FIRMS_API_KEY = os.environ.get("FIRMS_API_KEY")
# Use the /api/area/ endpoint which works with the MAP KEY
FIRMS_API_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"

# Use sensor names compatible with the /api/area/ endpoint (and that worked in manual tests)
FIRMS_SENSORS = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"]

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s') # Basic config
logger = logging.getLogger(__name__) # Standard logger

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
    # In Cloud Functions, print() goes to Cloud Logging.
    # For local, it goes to console. json.dumps ensures it's a single line JSON string.
    print(json.dumps(log_entry))


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
        self.base_url = base_url # This will be https://firms.modaps.eosdis.nasa.gov/api/area/csv/
        self.sensors = sensors
        _log_json("INFO", "FirmsDataRetriever initialized.", sensors=self.sensors, base_url=self.base_url)

    def _fetch_firms_data(self, sensor: str) -> Optional[pd.DataFrame]:
        """
        Fetches FIRMS data for a specific sensor for the last 24 hours (yesterday's data).
        Uses the /api/area/ endpoint.
        """
        # For "last 24 hours" using the /api/area/ endpoint:
        # We need yesterday's date and a day_range of 1.
        yesterday = datetime.utcnow() - timedelta(days=1)
        date_str = yesterday.strftime('%Y-%m-%d')
        day_range = "1"  # For one day of data ending on date_str

        # The base_url is "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        # Endpoint structure for /api/area/ is: /<KEY>/<SOURCE>/<AREA>/<DAY_RANGE>/<DATE>
        # Using lowercase "world" as per your working manual test.
        endpoint = f"{self.base_url}{self.api_key}/{sensor}/world/{day_range}/{date_str}"
        _log_json("INFO", "Attempting to fetch FIRMS data using /api/area/.",
                  sensor=sensor, date_for_data=date_str, endpoint=endpoint)

        try:
            response = requests.get(endpoint, timeout=30)

            # Check for "Invalid API call" specifically, as this endpoint might return it directly in body
            if "Invalid API call" in response.text:
                _log_json("ERROR", "FIRMS API (/api/area/) returned 'Invalid API call'. Check API key, sensor name, or endpoint structure.",
                          api_response_snippet=response.text[:200], status_code=response.status_code, sensor=sensor, endpoint=endpoint)
                return None

            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

            # FIRMS /api/area/ might return "No fire data found for the specified query." as plain text
            # with a 200 OK status.
            if not response.text.strip() or response.text.startswith("No fire data found"):
                _log_json("WARNING", "No fire data found or empty response from FIRMS API for query.",
                          sensor=sensor, endpoint=endpoint, response_text_snippet=response.text[:100])
                return None # Return empty DataFrame later, or None here is also fine if handled by caller.

            # Other error messages in body (less likely if status is 200, but good to keep)
            if "Error" in response.text or "Access Denied" in response.text: # Unlikely with 200 OK
                _log_json("ERROR", "FIRMS API returned an error message in response body despite 200 OK.",
                          api_response_snippet=response.text[:200], status_code=response.status_code, sensor=sensor)
                return None

            df = pd.read_csv(pd.io.common.StringIO(response.text))
            _log_json("INFO", "Successfully fetched FIRMS data for sensor using /api/area/.",
                      sensor=sensor, rows_fetched=len(df))
            return df
        except requests.exceptions.HTTPError as e:
            _log_json("ERROR", "HTTP error fetching FIRMS data (/api/area/).",
                      error=str(e), status_code=e.response.status_code if e.response else 'N/A', sensor=sensor)
            return None
        except requests.exceptions.ConnectionError as e:
            _log_json("ERROR", "Connection error fetching FIRMS data (/api/area/).", error=str(e), sensor=sensor)
            return None
        except requests.exceptions.Timeout as e:
            _log_json("ERROR", "Timeout fetching FIRMS data (/api/area/).", error=str(e), sensor=sensor)
            return None
        except requests.exceptions.RequestException as e: # Catch-all for other requests issues
            _log_json("ERROR", "An unexpected requests error occurred (/api/area/).", error=str(e), sensor=sensor)
            return None
        except pd.errors.EmptyDataError: # Should be caught by "No fire data found" check, but as a fallback
            _log_json("WARNING", "FIRMS CSV data is empty or malformed after successful fetch (pd.errors.EmptyDataError).", sensor=sensor)
            return None
        except Exception as e: # General unexpected errors
            _log_json("ERROR", "An unexpected error occurred during FIRMS data fetch or parsing (/api/area/).",
                      error_type=type(e).__name__, error=str(e), sensor=sensor)
            return None

    def get_and_filter_firms_data(self, monitored_regions: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Fetches FIRMS data for all specified sensors (for yesterday)
        and filters it by the provided monitored regions.
        """
        all_firms_data = []

        for sensor in self.sensors:
            df_sensor = self._fetch_firms_data(sensor)
            if df_sensor is not None and not df_sensor.empty:
                all_firms_data.append(df_sensor)

        empty_df_for_return = pd.DataFrame(columns=[
            'latitude', 'longitude', 'acq_date', 'acq_time', 'confidence',
            'frp', 'daynight', 'satellite', 'monitored_region_id'
        ])

        if not all_firms_data:
            _log_json("WARNING", "No FIRMS data retrieved from any sensor after attempts using /api/area/.")
            return empty_df_for_return

        combined_df = pd.concat(all_firms_data, ignore_index=True)
        initial_rows = len(combined_df)
        _log_json("INFO", "Combined raw FIRMS data from all sensors.", total_hotspots_before_filter=initial_rows)

        # Column names from /api/area/ CSV are typically:
        # latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,confidence,version,bright_t31,frp,daynight
        # Ensure your required_cols are present.
        required_cols = [
            'latitude', 'longitude', 'acq_date', 'acq_time', 'confidence',
            'frp', 'daynight', 'satellite'
        ]
        for col in required_cols:
            if col not in combined_df.columns:
                _log_json("WARNING", f"Missing expected column in combined FIRMS data: {col}. Adding with None.", column=col)
                combined_df[col] = None

        if 'confidence' in combined_df.columns:
            combined_df['confidence'] = combined_df['confidence'].astype(str).str.lower()
            # The /api/area/ might return confidence as numeric (0-100) or string ('low', 'nominal', 'high')
            # For simplicity with VIIRS, we'll keep the string check. If MODIS is added, this needs refinement.
            # A more robust way for numeric: pd.to_numeric(combined_df['confidence'], errors='coerce') >= 75 (for example)
            # Or check type: if is_numeric_dtype(combined_df['confidence']): ... else: ...
            filtered_by_confidence_df = combined_df[
                combined_df['confidence'].isin(['high', 'nominal']) # Keep as is for VIIRS assumption
            ].copy()
            _log_json("INFO", "Filtered FIRMS data by confidence ('high' or 'nominal').",
                      original_rows_before_confidence_filter=initial_rows,
                      rows_after_confidence_filter=len(filtered_by_confidence_df))
        else:
            _log_json("WARNING", "No 'confidence' column found in FIRMS data. Skipping confidence filter.")
            filtered_by_confidence_df = combined_df.copy()

        if filtered_by_confidence_df.empty:
            _log_json("INFO", "No FIRMS hotspots with 'high' or 'nominal' confidence after initial fetch /api/area/.")
            return empty_df_for_return

        filtered_hotspots_by_region = []
        for region in monitored_regions:
            region_id = region["id"]
            min_lon, min_lat, max_lon, max_lat = region["bbox"]

            region_df = filtered_by_confidence_df[
                (pd.to_numeric(filtered_by_confidence_df['latitude'], errors='coerce') >= min_lat) &
                (pd.to_numeric(filtered_by_confidence_df['latitude'], errors='coerce') <= max_lat) &
                (pd.to_numeric(filtered_by_confidence_df['longitude'], errors='coerce') >= min_lon) &
                (pd.to_numeric(filtered_by_confidence_df['longitude'], errors='coerce') <= max_lon)
            ].copy()

            if not region_df.empty:
                region_df['monitored_region_id'] = region_id
                filtered_hotspots_by_region.append(region_df)
                _log_json("INFO", "Found FIRMS hotspots in monitored region.", region_id=region_id, count=len(region_df))
            else:
                _log_json("INFO", "No FIRMS hotspots found in monitored region.", region_id=region_id)

        if not filtered_hotspots_by_region:
            _log_json("INFO", "No FIRMS hotspots found across all monitored regions after spatial filtering.")
            return empty_df_for_return

        final_df = pd.concat(filtered_hotspots_by_region, ignore_index=True)

        output_columns = [
            'latitude', 'longitude', 'acq_date', 'acq_time', 'confidence',
            'frp', 'daynight', 'satellite', 'monitored_region_id'
        ]
        for col in output_columns: # Ensure all output columns exist
            if col not in final_df.columns:
                final_df[col] = None

        final_df = final_df[output_columns] # Reorder and select

        # Ensure correct data types
        final_df['latitude'] = pd.to_numeric(final_df['latitude'], errors='coerce')
        final_df['longitude'] = pd.to_numeric(final_df['longitude'], errors='coerce')
        final_df['frp'] = pd.to_numeric(final_df['frp'], errors='coerce')
        final_df['acq_date'] = pd.to_datetime(final_df['acq_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        final_df['acq_time'] = final_df['acq_time'].astype(str).str.zfill(4) # Ensure HHMM format
        final_df['confidence'] = final_df['confidence'].astype(str)
        final_df['daynight'] = final_df['daynight'].astype(str)
        final_df['satellite'] = final_df['satellite'].astype(str)
        final_df['monitored_region_id'] = final_df['monitored_region_id'].astype(str)

        _log_json("INFO", "FIRMS data retrieval and filtering complete using /api/area/.",
                  total_filtered_hotspots=len(final_df))
        return final_df

# --- Example Usage (for local testing) ---
if __name__ == "__main__":
    if "FIRMS_API_KEY" not in os.environ:
        print("WARNING: FIRMS_API_KEY environment variable not set for local testing.")
        # Replace with your actual key if you want to run this block successfully
        os.environ["FIRMS_API_KEY"] = "0331973a7ee830ca7f026493faaa367a" # Using your key for test
        if os.environ["FIRMS_API_KEY"] == "YOUR_DUMMY_KEY_FOR_LOCAL_TESTING": # Check if it's still a dummy
             _log_json("WARNING", "Using placeholder FIRMS_API_KEY. Local test might fail or return 'Invalid API call'.")


    try:
        firms_retriever = FirmsDataRetriever(
            api_key=FIRMS_API_KEY,
            base_url=FIRMS_API_BASE_URL, # Uses the updated /api/area/ base_url
            sensors=FIRMS_SENSORS       # Uses the _NRT sensor names
        )

        _log_json("INFO", "Starting FIRMS data retrieval and filtering process (local test)...")
        filtered_firms_data = firms_retriever.get_and_filter_firms_data(MONITORED_REGIONS)

        _log_json("INFO", "Filtered FIRMS Hotspots Data (first 5 rows if any):")
        print(filtered_firms_data.head().to_string())
        _log_json("INFO", f"Total filtered hotspots: {len(filtered_firms_data)}")

        if not filtered_firms_data.empty:
            hotspots_by_region = filtered_firms_data.groupby('monitored_region_id').size().reset_index(name='hotspot_count')
            _log_json("INFO", "Hotspots per Monitored Region:")
            print(hotspots_by_region.to_string())
        else:
            _log_json("INFO", "No hotspots to group by region.")

    except ValueError as e: # Config error from __init__
        _log_json("ERROR", f"Configuration Error during local test: {e}")
    except Exception as e:
        _log_json("CRITICAL", "An unhandled error occurred during FIRMS local test.",
                  error_type=type(e).__name__, error=str(e))
        import traceback
        traceback.print_exc()
