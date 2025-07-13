"""
Main preprocessing pipeline for gravitational wave data.
Orchestrates data loading, processing, and augmentation.
"""

import os
import numpy as np
import pickle
import json
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

# Import our custom modules
from .data_loader import GWOSCDataLoader
from .signal_processing import SignalProcessor
from .waveform_simulation import WaveformSimulator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GWPreprocessingPipeline:
    """
    Main preprocessing pipeline for gravitational wave data.
    """
    
    def __init__(self, 
                 output_dir: str = "./data/processed",
                 sample_rate: int = 4096,
                 target_sample_rate: int = 2048,
                 duration: float = 4.0):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            output_dir: Directory to save processed data
            sample_rate: Original sample rate
            target_sample_rate: Target sample rate after processing
            duration: Duration of data segments in seconds
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        self.duration = duration
        
        # Initialize components
        self.data_loader = GWOSCDataLoader()
        self.signal_processor = SignalProcessor(sample_rate)
        self.waveform_simulator = WaveformSimulator(sample_rate)
        
        # Data storage
        self.processed_data = []
        self.metadata = []
        
    def process_event_data(self, 
                          events: List[Dict], 
                          detectors: List[str] = ['H1', 'L1', 'V1']) -> None:
        """
        Process event data for all detectors.
        
        Args:
            events: List of event dictionaries from GWOSC
            detectors: List of detector names to process
        """
        logger.info(f"Processing {len(events)} events for detectors: {detectors}")
        
        for event in events:
            event_name = event.get('commonName', 'Unknown')
            gps_time = event.get('GPS', None)
            
            if gps_time is None:
                logger.warning(f"No GPS time found for event {event_name}")
                continue
                
            logger.info(f"Processing event: {event_name} at GPS {gps_time}")
            
            for detector in detectors:
                try:
                    # Load strain data
                    strain_data = self.data_loader.load_strain_data(
                        detector, gps_time, self.duration, self.sample_rate
                    )
                    
                    if strain_data is None:
                        logger.warning(f"Failed to load data for {detector} at {gps_time}")
                        continue
                    
                    # Process the strain data
                    time_series, spectrogram = self.signal_processor.process_strain_data(
                        strain_data.value,
                        target_sample_rate=self.target_sample_rate
                    )
                    
                    # Determine event type (simplified classification)
                    event_type = self._classify_event(event)
                    
                    # Save processed data
                    self._save_processed_sample(
                        time_series, spectrogram, event_type, 
                        detector, gps_time, event_name
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing {detector} data for event {event_name}: {e}")
                    continue
    
    def generate_noise_data(self, 
                           detectors: List[str] = ['H1', 'L1', 'V1'],
                           num_noise_segments: int = 100,
                           start_gps: float = 1126051217,  # Start of O1
                           end_gps: float = 1137254417) -> List[np.ndarray]:   # End of O1
        """
        Generate noise data segments.
        
        Args:
            detectors: List of detector names
            num_noise_segments: Number of noise segments per detector
            start_gps: Start GPS time for noise generation
            end_gps: End GPS time for noise generation
            
        Returns:
            List of noise segments
        """
        logger.info(f"Generating {num_noise_segments} noise segments per detector")
        
        noise_segments = []
        
        for detector in detectors:
            # Get random noise times
            noise_times = self.data_loader.get_noise_segments(
                detector, start_gps, end_gps, self.duration, num_noise_segments
            )
            
            for gps_time in noise_times:
                try:
                    # Load noise data
                    noise_data = self.data_loader.load_strain_data(
                        detector, gps_time, self.duration, self.sample_rate
                    )
                    
                    if noise_data is None:
                        continue
                    
                    # Process noise data
                    time_series, spectrogram = self.signal_processor.process_strain_data(
                        noise_data.value,
                        target_sample_rate=self.target_sample_rate
                    )
                    
                    # Save as noise class
                    self._save_processed_sample(
                        time_series, spectrogram, "NOISE", 
                        detector, gps_time, "noise_segment"
                    )
                    
                    # Store raw noise for augmentation
                    noise_segments.append(time_series)
                    
                except Exception as e:
                    logger.error(f"Error processing noise data for {detector} at {gps_time}: {e}")
                    continue
        
        return noise_segments
    
    def augment_minority_classes(self, 
                               noise_segments: List[np.ndarray],
                               num_bns_injections: int = 200,
                               num_bbh_injections: int = 100) -> None:
        """
        Augment minority classes by injecting synthetic signals.
        
        Args:
            noise_segments: List of noise segments for injection
            num_bns_injections: Number of BNS injections
            num_bbh_injections: Number of BBH injections
        """
        logger.info("Starting data augmentation with synthetic injections")
        
        # Generate BNS injections
        bns_data = self.waveform_simulator.generate_augmented_dataset(
            noise_segments, "BNS", num_bns_injections
        )
        
        # Generate BBH injections
        bbh_data = self.waveform_simulator.generate_augmented_dataset(
            noise_segments, "BBH", num_bbh_injections
        )
        
        # Process and save augmented data
        for injected_data, signal_type in bns_data + bbh_data:
            # Generate spectrogram
            spectrogram = self.signal_processor.generate_spectrogram(injected_data)
            
            # Save augmented sample
            self._save_processed_sample(
                injected_data, spectrogram, signal_type,
                "synthetic", 0.0, f"augmented_{signal_type}"
            )
    
    def _classify_event(self, event: Dict) -> str:
        """
        Classify event type based on metadata.
        
        Args:
            event: Event dictionary from GWOSC
            
        Returns:
            Event type string
        """
        # Simple classification based on event name patterns
        event_name = event.get('commonName', '').upper()
        
        if 'GW' in event_name:
            # Try to determine if it's BBH or BNS based on naming convention
            # This is a simplified heuristic
            if any(x in event_name for x in ['170817', 'BNS']):
                return 'BNS'
            else:
                return 'BBH'
        else:
            return 'UNKNOWN'
    
    def _save_processed_sample(self, 
                              time_series: np.ndarray,
                              spectrogram: np.ndarray,
                              event_type: str,
                              detector: str,
                              gps_time: float,
                              event_name: str) -> None:
        """
        Save processed sample to disk.
        
        Args:
            time_series: Processed time series data
            spectrogram: Generated spectrogram
            event_type: Type of event (BBH, BNS, NOISE)
            detector: Detector name
            gps_time: GPS time
            event_name: Event name
        """
        # Create unique filename
        filename = f"{event_name}_{detector}_{gps_time}_{event_type}"
        
        # Save time series
        ts_path = self.output_dir / f"{filename}_timeseries.npy"
        np.save(ts_path, time_series)
        
        # Save spectrogram
        spec_path = self.output_dir / f"{filename}_spectrogram.npy"
        np.save(spec_path, spectrogram)
        
        # Store metadata
        metadata = {
            'filename': filename,
            'event_type': event_type,
            'detector': detector,
            'gps_time': gps_time,
            'event_name': event_name,
            'sample_rate': self.target_sample_rate,
            'duration': self.duration,
            'timeseries_path': str(ts_path),
            'spectrogram_path': str(spec_path)
        }
        
        self.metadata.append(metadata)
        
        logger.info(f"Saved processed sample: {filename}")
    
    def save_metadata(self) -> None:
        """
        Save metadata to disk.
        """
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        logger.info(f"Saved metadata for {len(self.metadata)} samples to {metadata_path}")
    
    def run_full_pipeline(self, 
                         num_noise_segments: int = 100,
                         num_bns_injections: int = 200,
                         num_bbh_injections: int = 100) -> None:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            num_noise_segments: Number of noise segments to generate
            num_bns_injections: Number of BNS injections for augmentation
            num_bbh_injections: Number of BBH injections for augmentation
        """
        logger.info("Starting full preprocessing pipeline")
        
        # Step 1: Get event list
        events = self.data_loader.get_event_list()
        
        if not events:
            logger.error("No events found. Exiting pipeline.")
            return
        
        # Step 2: Process event data
        self.process_event_data(events)
        
        # Step 3: Generate noise data
        noise_segments = self.generate_noise_data(
            num_noise_segments=num_noise_segments
        )
        
        # Step 4: Augment minority classes
        if noise_segments:
            self.augment_minority_classes(
                noise_segments, num_bns_injections, num_bbh_injections
            )
        
        # Step 5: Save metadata
        self.save_metadata()
        
        logger.info("Preprocessing pipeline completed successfully")
        logger.info(f"Total samples processed: {len(self.metadata)}")
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self) -> None:
        """
        Print summary of processed data.
        """
        if not self.metadata:
            return
            
        # Count samples by type
        type_counts = {}
        detector_counts = {}
        
        for sample in self.metadata:
            event_type = sample['event_type']
            detector = sample['detector']
            
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
            detector_counts[detector] = detector_counts.get(detector, 0) + 1
        
        logger.info("=== PREPROCESSING SUMMARY ===")
        logger.info(f"Total samples: {len(self.metadata)}")
        logger.info("Samples by type:")
        for event_type, count in type_counts.items():
            logger.info(f"  {event_type}: {count}")
        logger.info("Samples by detector:")
        for detector, count in detector_counts.items():
            logger.info(f"  {detector}: {count}")
        logger.info("=============================")


if __name__ == "__main__":
    # Example usage
    pipeline = GWPreprocessingPipeline()
    pipeline.run_full_pipeline(
        num_noise_segments=50,
        num_bns_injections=100,
        num_bbh_injections=50
    )
