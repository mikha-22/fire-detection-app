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

    def _fetch_firms_data(self, sensor: str, bbox_str: str) -> Optional[pd.DataFrame]:
        date_str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
        endpoint = f"{self.base_url}{self.api_key}/{sensor}/{bbox_str}/1/{date_str}"
        _log_json("INFO", "Fetching FIRMS data for specific area.", endpoint=endpoint)
        try:
            response = requests.get(endpoint, timeout=90)
            response.raise_for_status()
            if not response.text.strip() or response.text.startswith("No fire data found"):
                logging.info(f"No hotspots found for {sensor} in the given area.")
                return None
            return pd.read_csv(pd.io.common.StringIO(response.text))
        except requests.RequestException as e:
            _log_json("ERROR", "FIRMS data fetch failed.", error=str(e), sensor=sensor)
            return None

    def get_and_filter_firms_data(self, monitored_regions: List[Dict[str, Any]]) -> pd.DataFrame:
        all_dfs = []
        if not monitored_regions:
            _log_json("ERROR", "No monitored regions provided to fetch data for.")
            return pd.DataFrame()

        for region in monitored_regions:
            bbox_str = ",".join(map(str, region["bbox"]))
            for sensor in self.sensors:
                df = self._fetch_firms_data(sensor, bbox_str)
                if df is not None and not df.empty:
                    all_dfs.append(df)
        
        if not all_dfs:
            _log_json("INFO", "No FIRMS hotspots found in any of the specified regions.")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
        if 'confidence' in combined_df.columns:
            combined_df = combined_df[combined_df['confidence'].astype(str).str.lower().isin(['h', 'n'])]
        
        _log_json("INFO", "FIRMS data retrieval complete.", total_hotspots=len(combined_df))
        return combined_df
