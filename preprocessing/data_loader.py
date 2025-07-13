"""
Data loader for GWOSC gravitational wave data.
Handles downloading and initial processing of strain data.
"""

import numpy as np
import h5py
from gwpy.timeseries import TimeSeries
from gwpy.segments import DataQualityFlag
import requests
from urllib.parse import urljoin
import os
import json
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GWOSCDataLoader:
    """
    Loads gravitational wave strain data from GWOSC (Gravitational Wave Open Science Center).
    """
    
    def __init__(self, cache_dir: str = "./data/cache"):
        """
        Initialize the data loader.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        self.base_url = "https://www.gw-openscience.org/eventapi/json/GWTC/"
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_event_list(self, catalog: str = "GWTC-3-confident") -> List[Dict]:
        """
        Get list of gravitational wave events from GWOSC.
        
        Args:
            catalog: Event catalog name
            
        Returns:
            List of event dictionaries
        """
        url = urljoin(self.base_url, f"{catalog}/")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            events = response.json()
            logger.info(f"Found {len(events)} events in {catalog}")
            return events
        except requests.RequestException as e:
            logger.error(f"Failed to fetch event list: {e}")
            return []
    
    def load_strain_data(self, 
                        detector: str, 
                        gps_time: float, 
                        duration: float = 4.0,
                        sample_rate: int = 4096) -> Optional[TimeSeries]:
        """
        Load strain data for a specific detector around an event time.
        
        Args:
            detector: Detector name (e.g., 'H1', 'L1', 'V1')
            gps_time: GPS time of the event
            duration: Duration of data to load in seconds
            sample_rate: Desired sample rate in Hz
            
        Returns:
            TimeSeries object or None if failed
        """
        start_time = gps_time - duration / 2
        end_time = gps_time + duration / 2
        
        try:
            # Try to load from GWOSC
            strain = TimeSeries.fetch_open_data(
                detector, 
                start_time, 
                end_time, 
                sample_rate=sample_rate,
                cache=True
            )
            
            logger.info(f"Loaded {duration}s of {detector} data around GPS {gps_time}")
            return strain
            
        except Exception as e:
            logger.error(f"Failed to load strain data for {detector} at GPS {gps_time}: {e}")
            return None
    
    def get_noise_segments(self, 
                          detector: str, 
                          start_gps: float, 
                          end_gps: float, 
                          duration: float = 4.0,
                          num_segments: int = 100) -> List[float]:
        """
        Get random noise segments from times without known events.
        
        Args:
            detector: Detector name
            start_gps: Start GPS time for search
            end_gps: End GPS time for search
            duration: Duration of each segment
            num_segments: Number of noise segments to generate
            
        Returns:
            List of GPS times for noise segments
        """
        # Generate random times within the range
        noise_times = []
        time_range = end_gps - start_gps
        
        for _ in range(num_segments):
            # Generate random time, ensuring we have enough buffer
            random_time = start_gps + np.random.uniform(duration, time_range - duration)
            noise_times.append(random_time)
        
        logger.info(f"Generated {len(noise_times)} noise segment times for {detector}")
        return noise_times
    
    def save_metadata(self, metadata: Dict, filepath: str):
        """
        Save metadata to JSON file.
        
        Args:
            metadata: Metadata dictionary
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to {filepath}")
