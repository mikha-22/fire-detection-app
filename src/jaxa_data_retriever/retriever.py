# src/jaxa_data_retriever/retriever.py

import os
import io
import logging
from datetime import datetime
import pandas as pd
import paramiko

# --- JAXA Configuration ---
JAXA_SFTP_HOST = 'ftp.ptree.jaxa.jp'
JAXA_SFTP_PORT = 2051
JAXA_L3_HOURLY_BASE_PATH = '/pub/himawari/L3/WLF/010/'
USERNAME = "nathaniell.wijaya_gmail.com"
PASSWORD = "SP+wari8"

class JaxaDataRetriever:
    """
    A dedicated class for retrieving wildfire data from the JAXA P-Tree SFTP server.
    """
    def __init__(self):
        """Initializes the JAXA Data Retriever."""
        logging.info("JaxaDataRetriever initialized.")
        self.host = JAXA_SFTP_HOST
        self.port = JAXA_SFTP_PORT
        self.base_path = JAXA_L3_HOURLY_BASE_PATH
        self.username = USERNAME
        self.password = PASSWORD

    def get_l3_hourly_data(self, target_date: datetime) -> pd.DataFrame:
        """
        Connects to the JAXA SFTP server and downloads Level 3 hourly wildfire data
        for a specified date.
        """
        logging.info(f"Fetching JAXA L3 wildfire data via SFTP for target date: {target_date.strftime('%Y-%m-%d')}")
        all_hotspots = []
        try:
            with paramiko.Transport((self.host, self.port)) as transport:
                transport.connect(username=self.username, password=self.password)
                with paramiko.SFTPClient.from_transport(transport) as sftp:
                    logging.info(f"Successfully connected to JAXA SFTP server: {self.host}")

                    remote_dir = f"{self.base_path}{target_date.strftime('%Y%m')}/{target_date.strftime('%d')}/"

                    try:
                        for filename in sftp.listdir(remote_dir):
                            if "L3WLF" in filename and filename.endswith('.csv'):
                                remote_filepath = remote_dir + filename
                                logging.info(f"Processing JAXA file: {remote_filepath}")
                                with sftp.open(remote_filepath, 'r') as f:
                                    content = f.read().decode('utf-8')
                                    col_names = ['year', 'month', 'day', 'hour', 'latitude', 'longitude', 'frp_mean', 'frp_max', 'confidence', 'n_detections']
                                    df = pd.read_csv(io.StringIO(content), names=col_names, comment='#')
                                    all_hotspots.append(df)
                    except FileNotFoundError:
                        logging.info(f"No JAXA data directory found for {target_date.strftime('%Y-%m-%d')}.")
                    except Exception as e:
                        logging.warning(f"Could not process JAXA directory {remote_dir}: {e}")

        except Exception as e:
            logging.error(f"An unexpected error occurred during JAXA SFTP operation: {e}", exc_info=True)

        if not all_hotspots:
            logging.info("No JAXA wildfire hotspots were processed.")
            return pd.DataFrame()

        combined_df = pd.concat(all_hotspots, ignore_index=True)
        logging.info(f"Successfully fetched and combined {len(combined_df)} hotspots from JAXA.")
        return combined_df
