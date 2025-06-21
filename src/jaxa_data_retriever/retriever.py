# src/jaxa_data_retriever/retriever.py

import os
import io
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
import paramiko

# --- JAXA Configuration ---
JAXA_SFTP_HOST = 'ftp.ptree.jaxa.jp'
JAXA_SFTP_PORT = 2051
JAXA_L3_BASE_PATH = '/pub/himawari/L3/WLF/010/'
USERNAME = "nathaniell.wijaya_gmail.com"
PASSWORD = "SP+wari8"

class JaxaDataRetriever:
    """
    A dedicated class for retrieving hourly Level 3 wildfire data from the JAXA P-Tree SFTP server.
    """
    def __init__(self):
        logging.info("JaxaDataRetriever initialized for L3 Hourly Product.")
        self.host = JAXA_SFTP_HOST
        self.port = JAXA_SFTP_PORT
        self.base_path = JAXA_L3_BASE_PATH
        self.username = USERNAME
        self.password = PASSWORD

    def _get_hourly_file_paths_for_indonesian_day(self, target_date_str: str):
        """
        Generates the list of 24 hourly SFTP directories and corresponding file
        prefixes for a given Indonesian day (UTC+7).
        """
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
        start_utc = datetime(target_date.year, target_date.month, target_date.day, 17, 0, 0, tzinfo=timezone.utc) - timedelta(days=1)
        
        paths = []
        for i in range(24):
            current_hour = start_utc + timedelta(hours=i)
            remote_dir = f"{self.base_path}{current_hour.strftime('%Y%m')}/{current_hour.strftime('%d')}/"
            filename_prefix = f"H09_{current_hour.strftime('%Y%m%d')}_{current_hour.strftime('%H')}00"
            paths.append((remote_dir, filename_prefix))
        return paths

    def get_data_for_indonesian_day(self, target_date_str: str) -> pd.DataFrame:
        """
        Connects to the JAXA SFTP server and downloads all hourly Level 3 wildfire
        CSV files that fall within a specific Indonesian day.
        """
        logging.info(f"Fetching JAXA L3 hourly data for Indonesian date: {target_date_str}")
        hourly_paths = self._get_hourly_file_paths_for_indonesian_day(target_date_str)
        all_hotspots_df = []

        try:
            with paramiko.Transport((self.host, self.port)) as transport:
                transport.connect(username=self.username, password=self.password)
                with paramiko.SFTPClient.from_transport(transport) as sftp:
                    logging.info(f"Successfully connected to JAXA SFTP server: {self.host}")

                    for remote_dir, filename_prefix in hourly_paths:
                        try:
                            files_in_dir = sftp.listdir(remote_dir)
                            target_filename = next((f for f in files_in_dir if f.startswith(filename_prefix) and f.endswith('.csv')), None)

                            if not target_filename:
                                logging.warning(f"No file found in {remote_dir} with prefix {filename_prefix}")
                                continue

                            remote_filepath = remote_dir + target_filename
                            with sftp.open(remote_filepath, 'r') as f:
                                content = f.read().decode('utf-8')
                                
                                # --- CORRECTED: Use the actual column names from the file ---
                                col_names = ['year', 'month', 'day', 'hour', 'lat', 'lon', 'ave(frp)', 'max(frp)', 'ave(confidence)', 'N']
                                df = pd.read_csv(io.StringIO(content), names=col_names, header=None, comment='#')
                                all_hotspots_df.append(df)

                        except FileNotFoundError:
                            logging.warning(f"Hourly directory not found, skipping: {remote_dir}")
                            continue
                        except Exception as e:
                            logging.error(f"Error processing directory {remote_dir}: {e}")

        except Exception as e:
            logging.error(f"An unexpected error occurred during JAXA SFTP operation: {e}", exc_info=True)
            return pd.DataFrame()

        if not all_hotspots_df:
            logging.warning("No JAXA hotspots found for the entire Indonesian day.")
            return pd.DataFrame()
            
        final_df = pd.concat(all_hotspots_df, ignore_index=True)
        logging.info(f"Successfully fetched and processed {len(final_df)} hotspots from JAXA for the full Indonesian day.")
        return final_df
