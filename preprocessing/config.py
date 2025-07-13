"""
Configuration file for gravitational wave preprocessing pipeline.
"""

import os
from pathlib import Path

# Data directories
DATA_DIR = Path("./data")
CACHE_DIR = DATA_DIR / "cache"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Create directories if they don't exist
for directory in [DATA_DIR, CACHE_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Signal processing parameters
SIGNAL_PROCESSING_CONFIG = {
    'sample_rate': 4096,           # Original sample rate (Hz)
    'target_sample_rate': 2048,    # Target sample rate after processing (Hz)
    'duration': 4.0,               # Duration of data segments (seconds)
    'low_freq': 20.0,              # Low frequency cutoff (Hz)
    'high_freq': 500.0,            # High frequency cutoff (Hz)
    'filter_order': 4,             # Butterworth filter order
    'nperseg': 512,                # STFT window length
    'spectrogram_size': (224, 224), # Target spectrogram size
    'normalize': True              # Whether to normalize data
}

# Data augmentation parameters
AUGMENTATION_CONFIG = {
    'num_noise_segments': 100,     # Number of noise segments per detector
    'num_bns_injections': 200,     # Number of BNS injections
    'num_bbh_injections': 100,     # Number of BBH injections
    'snr_range': (8.0, 20.0),      # SNR range for injections
    'mass_range_bns': (1.0, 2.0),  # Mass range for BNS (solar masses)
    'mass_range_bbh': (10.0, 80.0), # Mass range for BBH (solar masses)
    'distance_range_bns': (50, 200), # Distance range for BNS (Mpc)
    'distance_range_bbh': (200, 1000) # Distance range for BBH (Mpc)
}

# Detector configuration
DETECTOR_CONFIG = {
    'detectors': ['H1', 'L1', 'V1'], # List of detectors to process
    'noise_time_ranges': {
        'O1': (1126051217, 1137254417),  # O1 run
        'O2': (1164556817, 1187733618),  # O2 run  
        'O3': (1238166018, 1269363618)   # O3 run
    }
}

# Event catalogs
EVENT_CATALOGS = [
    'GWTC-1-confident',
    'GWTC-2-confident', 
    'GWTC-3-confident'
]

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': DATA_DIR / 'preprocessing.log'
}

# GWOSC API configuration
GWOSC_CONFIG = {
    'base_url': 'https://www.gw-openscience.org/',
    'api_url': 'https://www.gw-openscience.org/eventapi/json/GWTC/',
    'timeout': 30,  # Request timeout in seconds
    'retries': 3    # Number of retry attempts
}

# Model training configuration (for future use)
MODEL_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'validation_split': 0.2,
    'test_split': 0.1,
    'random_seed': 42
}

# File naming conventions
NAMING_CONVENTIONS = {
    'timeseries_suffix': '_timeseries.npy',
    'spectrogram_suffix': '_spectrogram.npy',
    'metadata_file': 'metadata.json',
    'processed_data_format': '{event_name}_{detector}_{gps_time}_{event_type}'
}
