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
# --- UPDATED: Path to the Daily L3 product ---
JAXA_L3_DAILY_BASE_PATH = '/pub/himawari/L3/WLF/010/'
USERNAME = "nathaniell.wijaya_gmail.com"
PASSWORD = "SP+wari8"

class JaxaDataRetriever:
    """
    A dedicated class for retrieving daily aggregated wildfire data from the JAXA P-Tree SFTP server.
    """
    def __init__(self):
        """Initializes the JAXA Data Retriever."""
        logging.info("JaxaDataRetriever initialized for Daily L3 Product.")
        self.host = JAXA_SFTP_HOST
        self.port = JAXA_SFTP_PORT
        self.base_path = JAXA_L3_DAILY_BASE_PATH
        self.username = USERNAME
        self.password = PASSWORD

    def get_l3_daily_data(self, target_date: datetime) -> pd.DataFrame:
        """
        Connects to the JAXA SFTP server and downloads the single daily summary
        wildfire CSV file for a specified date.
        """
        logging.info(f"Fetching JAXA L3 Daily wildfire data for target date: {target_date.strftime('%Y-%m-%d')}")
        
        try:
            with paramiko.Transport((self.host, self.port)) as transport:
                transport.connect(username=self.username, password=self.password)
                with paramiko.SFTPClient.from_transport(transport) as sftp:
                    logging.info(f"Successfully connected to JAXA SFTP server: {self.host}")

                    # Construct the path to the specific daily file
                    date_str_nodash = target_date.strftime('%Y%m%d')
                    remote_dir = f"{self.base_path}{target_date.strftime('%Y%m')}/daily/"
                    # The filename is predictable, e.g., H09_20250616_0000_1DWLF010_FLDK.06001_06001.csv
                    # We will search for the key parts to be safe.
                    
                    files_in_dir = sftp.listdir(remote_dir)
                    target_filename = None
                    for filename in files_in_dir:
                        if date_str_nodash in filename and "1DWLF" in filename:
                            target_filename = filename
                            break
                    
                    if not target_filename:
                        logging.warning(f"No daily file found in {remote_dir} for {date_str_nodash}")
                        return pd.DataFrame()

                    remote_filepath = remote_dir + target_filename
                    logging.info(f"Processing JAXA daily file: {remote_filepath}")
                    with sftp.open(remote_filepath, 'r') as f:
                        content = f.read().decode('utf-8')
                        # Format from README: year, month, day, lat, lon, FRP (mean), FRP (max), N
                        col_names = ['year', 'month', 'day', 'latitude', 'longitude', 'frp_mean', 'frp_max', 'n_detections']
                        df = pd.read_csv(io.StringIO(content), names=col_names, comment='#')
                        logging.info(f"Successfully fetched and processed {len(df)} hotspots from JAXA daily file.")
                        return df

        except FileNotFoundError:
            logging.info(f"No JAXA daily data directory found for {target_date.strftime('%Y-%m-%d')}.")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"An unexpected error occurred during JAXA SFTP operation: {e}", exc_info=True)
            return pd.DataFrame()
        
        return pd.DataFrame() # Should not be reached, but as a fallback
