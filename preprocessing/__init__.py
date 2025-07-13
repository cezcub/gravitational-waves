"""
Preprocessing module for gravitational wave data.

This module provides tools for:
- Loading gravitational wave data from GWOSC
- Signal processing (filtering, whitening, resampling)
- Waveform simulation for data augmentation
- Complete preprocessing pipeline

Main classes:
- GWOSCDataLoader: Downloads and loads strain data
- SignalProcessor: Handles signal processing operations
- WaveformSimulator: Generates synthetic waveforms
- GWPreprocessingPipeline: Main preprocessing pipeline
"""

from .data_loader import GWOSCDataLoader
from .signal_processing import SignalProcessor
from .waveform_simulation import WaveformSimulator
from .preprocessing_pipeline import GWPreprocessingPipeline

__all__ = [
    'GWOSCDataLoader',
    'SignalProcessor', 
    'WaveformSimulator',
    'GWPreprocessingPipeline'
]

__version__ = '1.0.0'
