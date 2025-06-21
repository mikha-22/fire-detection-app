# src/firms_data_retriever/retriever.py

import os
import json
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
from typing import List, Dict, Any, Optional

FIRMS_API_KEY = os.environ.get("FIRMS_API_KEY")
FIRMS_API_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
FIRMS_SENSORS = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def _log_json(severity: str, message: str, **kwargs):
    print(json.dumps({"severity": severity.upper(), "message": message, **kwargs}))

class FirmsDataRetriever:
    def __init__(self, api_key: str, base_url: str, sensors: List[str]):
        if not api_key:
            _log_json("CRITICAL", "FIRMS_API_KEY environment variable not set.")
            raise ValueError("API key required.")
        self.api_key, self.base_url, self.sensors = api_key, base_url, sensors
        _log_json("INFO", "FirmsDataRetriever initialized.", sensors=self.sensors)

    def _fetch_firms_data(self, sensor: str, bbox_str: str, date_str: str) -> Optional[pd.DataFrame]:
        """
        Fetches FIRMS data for a specific sensor, area, and a 1-day range
        starting from the given UTC date.
        """
        date_range = 1
        endpoint = f"{self.base_url}{self.api_key}/{sensor}/{bbox_str}/{date_range}/{date_str}"
        _log_json("INFO", "Fetching FIRMS data for UTC date.", endpoint=endpoint)
        try:
            response = requests.get(endpoint, timeout=90)
            response.raise_for_status()
            if not response.text.strip() or response.text.startswith("No fire data found"):
                logging.info(f"No hotspots found for {sensor} on {date_str}.")
                return None
            return pd.read_csv(pd.io.common.StringIO(response.text))
        except requests.RequestException as e:
            _log_json("ERROR", "FIRMS data fetch failed.", error=str(e), sensor=sensor)
            return None

    def get_data_for_indonesian_day(self, monitored_regions: List[Dict[str, Any]], target_date_str: str) -> pd.DataFrame:
        """
        Gets and filters FIRMS data for a full Indonesian calendar day (UTC+7).
        This involves fetching data from two consecutive UTC days and filtering the results.
        """
        if not monitored_regions:
            _log_json("ERROR", "No monitored regions provided.")
            return pd.DataFrame()

        target_date = datetime.strptime(target_date_str, '%Y-%m-%d')

        # Define the start and end of the Indonesian day in UTC
        # An Indonesian day (WIB) starts at 17:00 UTC the previous day.
        start_utc = datetime(target_date.year, target_date.month, target_date.day, 17, 0, 0, tzinfo=timezone.utc) - timedelta(days=1)
        end_utc = datetime(target_date.year, target_date.month, target_date.day, 16, 59, 59, tzinfo=timezone.utc)

        _log_json("INFO", "Querying for Indonesian Day.", target_day_wib=target_date_str, utc_window_start=start_utc.isoformat(), utc_window_end=end_utc.isoformat())

        # We need to fetch data for two UTC days to cover the full Indonesian day
        utc_day_1_str = (target_date - timedelta(days=1)).strftime('%Y-%m-%d')
        utc_day_2_str = target_date.strftime('%Y-%m-%d')
        
        all_dfs = []
        for region in monitored_regions:
            bbox_str = ",".join(map(str, region["bbox"]))
            for sensor in self.sensors:
                # Fetch data for the first UTC day
                df1 = self._fetch_firms_data(sensor, bbox_str, utc_day_1_str)
                if df1 is not None and not df1.empty:
                    all_dfs.append(df1)
                # Fetch data for the second UTC day
                df2 = self._fetch_firms_data(sensor, bbox_str, utc_day_2_str)
                if df2 is not None and not df2.empty:
                    all_dfs.append(df2)

        if not all_dfs:
            _log_json("INFO", "No FIRMS hotspots found in any of the specified regions for the required UTC window.")
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates()

        # Create a proper UTC datetime column for filtering
        combined_df['acq_datetime'] = pd.to_datetime(
            combined_df['acq_date'] + ' ' + combined_df['acq_time'].astype(str).str.zfill(4),
            format='%Y-%m-%d %H%M', utc=True
        )

        # Filter the combined data to the precise Indonesian day window
        filtered_df = combined_df[(combined_df['acq_datetime'] >= start_utc) & (combined_df['acq_datetime'] <= end_utc)].copy()

        # Standard confidence filtering
        if 'confidence' in filtered_df.columns:
            filtered_df['confidence'] = filtered_df['confidence'].astype(str)
            filtered_df = filtered_df[filtered_df['confidence'].str.lower().isin(['h', 'n'])]

        _log_json("INFO", "FIRMS data retrieval and filtering for Indonesian day complete.", total_hotspots=len(filtered_df))
        return filtered_df
