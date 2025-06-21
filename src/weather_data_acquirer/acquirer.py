# src/weather_data_acquirer/acquirer.py

import logging
from datetime import datetime
import pandas as pd
import numpy as np
import pytz
import openmeteo_requests
import requests_cache
from retry_requests import retry

class WeatherDataAcquirer:
    """
    A dedicated class for acquiring real-time weather data from Open-Meteo.
    Uses the live forecast API for up-to-the-hour data.
    """
    def __init__(self):
        """Initializes the Open-Meteo client with caching and retry logic."""
        logging.info("WeatherDataAcquirer initialized (Real-Time Forecast API).")
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        self.openmeteo = openmeteo_requests.Client(session=retry(cache_session, retries=5, backoff_factor=0.2))

    def get_historical_weather_data(self, latitude: float, longitude: float, timestamp: datetime) -> dict:
        """
        Fetches real-time weather data for a specific location and time using the
        Open-Meteo Forecast API.
        """
        logging.info(f"Fetching real-time weather for lat={latitude}, lon={longitude}")
        
        # --- CORRECTED: Use the real-time forecast API endpoint ---
        url = "https://api.open-meteo.com/v1/forecast"
        
        params = {
            "latitude": latitude, 
            "longitude": longitude,
            "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
                       "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
            "timezone": "UTC" # Ensure all data is in UTC
        }
        try:
            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]
            
            hourly = response.Hourly()
            
            hourly_data = {"date": pd.to_datetime(hourly.Time(), unit = 's', utc=True)}
            hourly_data["temperature_2m"] = hourly.Variables(0).ValuesAsNumpy()
            hourly_data["relative_humidity_2m"] = hourly.Variables(1).ValuesAsNumpy()
            hourly_data["dew_point_2m"] = hourly.Variables(2).ValuesAsNumpy()
            hourly_data["precipitation"] = hourly.Variables(3).ValuesAsNumpy()
            hourly_data["wind_speed_10m"] = hourly.Variables(4).ValuesAsNumpy()
            hourly_data["wind_direction_10m"] = hourly.Variables(5).ValuesAsNumpy()
            hourly_data["wind_gusts_10m"] = hourly.Variables(6).ValuesAsNumpy()

            hourly_df = pd.DataFrame(data=hourly_data)
            if hourly_df.empty: return {"error": "No weather data returned from API."}

            # Find the closest hourly forecast to the incident's timestamp
            target_timestamp_utc = timestamp.astimezone(pytz.utc)
            closest_row = hourly_df.iloc[(hourly_df['date'] - target_timestamp_utc).abs().argsort()[0]]

            if pd.isna(closest_row['temperature_2m']):
                logging.warning("Weather data contained NaN values. Returning error.")
                return {"error": "Valid weather data not available for the specific timestamp."}

            raw_weather = {
                "timestamp_utc": closest_row['date'].isoformat(),
                "temperature_celsius": closest_row['temperature_2m'],
                "relative_humidity_percent": closest_row['relative_humidity_2m'],
                "dew_point_celsius": closest_row['dew_point_2m'],
                "precipitation_mm": closest_row['precipitation'],
                "wind_speed_kmh": closest_row['wind_speed_10m'],
                "wind_direction_deg": closest_row['wind_direction_10m'],
                "wind_gusts_kmh": closest_row['wind_gusts_10m'],
            }

            sanitized_weather = {}
            for key, value in raw_weather.items():
                if isinstance(value, (np.floating, float)) and not pd.isna(value):
                    sanitized_weather[key] = round(float(value), 2)
                elif isinstance(value, (np.integer, int)) and not pd.isna(value):
                    sanitized_weather[key] = int(value)
                else:
                    sanitized_weather[key] = value
            
            return sanitized_weather
            
        except Exception as e:
            logging.warning(f"Could not fetch or process weather data: {e}")
            return {"error": str(e)}
