# src/firms_data_retriever/retriever.py

import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json # For _log_json helper

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
        Uses the /api/area/ endpoint.
        Includes debug print for raw CSV.
        """
        yesterday = datetime.utcnow() - timedelta(days=1)
        date_str = yesterday.strftime('%Y-%m-%d')
        day_range = "1"

        endpoint = f"{self.base_url}{self.api_key}/{sensor}/world/{day_range}/{date_str}"
        _log_json("INFO", "Attempting to fetch FIRMS data using /api/area/.",
                  sensor=sensor, date_for_data=date_str, endpoint=endpoint)

        try:
            response = requests.get(endpoint, timeout=60) # Increased timeout

            if "Invalid API call" in response.text:
                _log_json("ERROR", "FIRMS API (/api/area/) returned 'Invalid API call'. Check API key, sensor name, or endpoint structure.",
                          api_response_snippet=response.text[:200], status_code=response.status_code, sensor=sensor, endpoint=endpoint)
                return None

            response.raise_for_status()

            if not response.text.strip() or response.text.startswith("No fire data found"):
                _log_json("WARNING", "No fire data found or empty response from FIRMS API for query.",
                          sensor=sensor, endpoint=endpoint, response_text_snippet=response.text[:100])
                return None

            # --- DEBUG PRINT START ---
            print(f"\n--- RAW FIRMS CSV Output (first 5 lines) for sensor: {sensor} ---")
            lines = response.text.splitlines()
            for i in range(min(5, len(lines))): # Print header + up to 4 data lines
                print(lines[i])
            print("--- END RAW FIRMS CSV Output ---\n")
            # --- DEBUG PRINT END ---

            if "Error" in response.text or "Access Denied" in response.text:
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
        except requests.exceptions.RequestException as e:
            _log_json("ERROR", "An unexpected requests error occurred (/api/area/).", error=str(e), sensor=sensor)
            return None
        except pd.errors.EmptyDataError:
            _log_json("WARNING", "FIRMS CSV data is empty or malformed after successful fetch (pd.errors.EmptyDataError).", sensor=sensor)
            return None
        except Exception as e:
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

        # Ensure required columns exist, even if they are all None initially from some sensors.
        # The FIRMS /api/area/ CSV should have these columns:
        # latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
        # We are interested in: latitude, longitude, acq_date, acq_time, satellite, confidence, frp, daynight
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
            
            # Filter using the single-letter codes 'h' (high) and 'n' (nominal)
            confidence_values_to_keep = ['h', 'n']
            filtered_by_confidence_df = combined_df[
                combined_df['confidence'].isin(confidence_values_to_keep)
            ].copy()
            
            _log_json("INFO", "Filtered FIRMS data by confidence (kept 'h' or 'n').",
                      original_rows_before_confidence_filter=initial_rows,
                      rows_after_confidence_filter=len(filtered_by_confidence_df),
                      confidence_values_kept=confidence_values_to_keep)
        else:
            _log_json("WARNING", "No 'confidence' column found in FIRMS data. Skipping confidence filter.")
            filtered_by_confidence_df = combined_df.copy()

        if filtered_by_confidence_df.empty:
            _log_json("INFO", "No FIRMS hotspots with 'h' or 'n' confidence after filtering.")
            return empty_df_for_return

        filtered_hotspots_by_region = []
        for region in monitored_regions:
            region_id = region["id"]
            min_lon, min_lat, max_lon, max_lat = region["bbox"]

            # Create a working copy for this region's filter to avoid SettingWithCopyWarning
            temp_df_for_region_filter = filtered_by_confidence_df.copy()
            
            # Ensure latitude and longitude are numeric before filtering
            temp_df_for_region_filter['latitude'] = pd.to_numeric(temp_df_for_region_filter['latitude'], errors='coerce')
            temp_df_for_region_filter['longitude'] = pd.to_numeric(temp_df_for_region_filter['longitude'], errors='coerce')
            
            # Drop rows where lat/lon could not be coerced to numeric, as they can't be spatially filtered
            temp_df_for_region_filter.dropna(subset=['latitude', 'longitude'], inplace=True)

            region_df = temp_df_for_region_filter[
                (temp_df_for_region_filter['latitude'] >= min_lat) &
                (temp_df_for_region_filter['latitude'] <= max_lat) &
                (temp_df_for_region_filter['longitude'] >= min_lon) &
                (temp_df_for_region_filter['longitude'] <= max_lon)
            ].copy() # .copy() here ensures region_df is a new DataFrame

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
        # Ensure all output columns exist, fill with None if not
        for col in output_columns:
            if col not in final_df.columns:
                final_df[col] = None
        
        final_df = final_df[output_columns] # Reorder and select

        # Ensure correct data types for final output
        final_df['latitude'] = pd.to_numeric(final_df['latitude'], errors='coerce')
        final_df['longitude'] = pd.to_numeric(final_df['longitude'], errors='coerce')
        final_df['frp'] = pd.to_numeric(final_df['frp'], errors='coerce')
        
        final_df['acq_date'] = pd.to_datetime(final_df['acq_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        # Ensure acq_time is string, remove potential '.0' from float conversion, then zfill
        final_df['acq_time'] = final_df['acq_time'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(4)
        
        final_df['confidence'] = final_df['confidence'].astype(str) # Already lowercased
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
        # Example: os.environ["FIRMS_API_KEY"] = "YOUR_ACTUAL_FIRMS_KEY"
        # For this test, let's assume it's set or use a placeholder that might fail
        os.environ["FIRMS_API_KEY"] = os.environ.get("FIRMS_API_KEY", "0331973a7ee830ca7f026493faaa367a") # Using your key or placeholder
        if os.environ["FIRMS_API_KEY"] == "YOUR_DUMMY_KEY_FOR_LOCAL_TESTING":
             _log_json("WARNING", "Using placeholder FIRMS_API_KEY. Local test might fail or return 'Invalid API call'.")

    try:
        firms_retriever = FirmsDataRetriever(
            api_key=FIRMS_API_KEY,
            base_url=FIRMS_API_BASE_URL,
            sensors=FIRMS_SENSORS
        )

        _log_json("INFO", "Starting FIRMS data retrieval and filtering process (local test)...")
        # MONITORED_REGIONS is imported from src.common.config
        filtered_firms_data = firms_retriever.get_and_filter_firms_data(MONITORED_REGIONS)

        _log_json("INFO", "Filtered FIRMS Hotspots Data (first 5 rows if any):")
        print(filtered_firms_data.head().to_string()) # .to_string() for better console output
        _log_json("INFO", f"Total filtered hotspots: {len(filtered_firms_data)}")

        if not filtered_firms_data.empty:
            hotspots_by_region = filtered_firms_data.groupby('monitored_region_id').size().reset_index(name='hotspot_count')
            _log_json("INFO", "Hotspots per Monitored Region:")
            print(hotspots_by_region.to_string())
        else:
            _log_json("INFO", "No hotspots to group by region.")

    except ValueError as e:
        _log_json("ERROR", f"Configuration Error during local test: {e}")
    except Exception as e:
        _log_json("CRITICAL", "An unhandled error occurred during FIRMS local test.",
                  error_type=type(e).__name__, error=str(e))
        import traceback
        traceback.print_exc()
