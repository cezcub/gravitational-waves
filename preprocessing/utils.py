"""
Utility functions for data visualization and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataVisualizer:
    """
    Utility class for visualizing gravitational wave data.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    def plot_timeseries(self, 
                       time_series: np.ndarray, 
                       sample_rate: int = 2048,
                       title: str = "Gravitational Wave Time Series",
                       save_path: Optional[Path] = None) -> None:
        """
        Plot time series data.
        
        Args:
            time_series: Time series data
            sample_rate: Sample rate in Hz
            title: Plot title
            save_path: Optional path to save plot
        """
        time = np.arange(len(time_series)) / sample_rate
        
        plt.figure(figsize=self.figsize)
        plt.plot(time, time_series, 'b-', linewidth=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Strain')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_spectrogram(self, 
                        spectrogram: np.ndarray,
                        title: str = "Gravitational Wave Spectrogram",
                        save_path: Optional[Path] = None) -> None:
        """
        Plot spectrogram data.
        
        Args:
            spectrogram: Spectrogram data
            title: Plot title
            save_path: Optional path to save plot
        """
        plt.figure(figsize=self.figsize)
        plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Log Amplitude')
        plt.xlabel('Time Bins')
        plt.ylabel('Frequency Bins')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_comparison(self, 
                       data_dict: Dict[str, np.ndarray],
                       plot_type: str = "timeseries",
                       sample_rate: int = 2048) -> None:
        """
        Plot comparison of multiple data samples.
        
        Args:
            data_dict: Dictionary of {label: data} pairs
            plot_type: Type of plot ("timeseries" or "spectrogram")
            sample_rate: Sample rate for time series plots
        """
        n_samples = len(data_dict)
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
        
        if n_samples == 1:
            axes = [axes]
        
        for i, (label, data) in enumerate(data_dict.items()):
            ax = axes[i]
            
            if plot_type == "timeseries":
                time = np.arange(len(data)) / sample_rate
                ax.plot(time, data, 'b-', linewidth=0.5)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Strain')
            
            elif plot_type == "spectrogram":
                im = ax.imshow(data, aspect='auto', origin='lower', cmap='viridis')
                ax.set_xlabel('Time Bins')
                ax.set_ylabel('Frequency Bins')
                plt.colorbar(im, ax=ax, label='Log Amplitude')
            
            ax.set_title(f"{label}")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_dataset_statistics(self, metadata_path: Path) -> None:
        """
        Plot statistics of the processed dataset.
        
        Args:
            metadata_path: Path to metadata.json file
        """
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Count samples by type
        type_counts = {}
        detector_counts = {}
        
        for sample in metadata:
            event_type = sample['event_type']
            detector = sample['detector']
            
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
            detector_counts[detector] = detector_counts.get(detector, 0) + 1
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot event types
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
        
        ax1.bar(types, counts, color=colors)
        ax1.set_title('Samples by Event Type')
        ax1.set_xlabel('Event Type')
        ax1.set_ylabel('Number of Samples')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot detector distribution
        detectors = list(detector_counts.keys())
        det_counts = list(detector_counts.values())
        
        ax2.bar(detectors, det_counts, color='skyblue')
        ax2.set_title('Samples by Detector')
        ax2.set_xlabel('Detector')
        ax2.set_ylabel('Number of Samples')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"Dataset Summary:")
        print(f"Total samples: {len(metadata)}")
        print(f"Event types: {dict(type_counts)}")
        print(f"Detectors: {dict(detector_counts)}")


class DataAnalyzer:
    """
    Utility class for analyzing processed gravitational wave data.
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize the analyzer.
        
        Args:
            data_dir: Directory containing processed data
        """
        self.data_dir = Path(data_dir)
        self.metadata_path = self.data_dir / "metadata.json"
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
    
    def get_sample_statistics(self) -> Dict:
        """
        Get statistics about the processed dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            'total_samples': len(self.metadata),
            'event_types': {},
            'detectors': {},
            'sample_rates': {},
            'durations': {}
        }
        
        for sample in self.metadata:
            # Event types
            event_type = sample['event_type']
            stats['event_types'][event_type] = stats['event_types'].get(event_type, 0) + 1
            
            # Detectors
            detector = sample['detector']
            stats['detectors'][detector] = stats['detectors'].get(detector, 0) + 1
            
            # Sample rates
            sample_rate = sample['sample_rate']
            stats['sample_rates'][sample_rate] = stats['sample_rates'].get(sample_rate, 0) + 1
            
            # Durations
            duration = sample['duration']
            stats['durations'][duration] = stats['durations'].get(duration, 0) + 1
        
        return stats
    
    def load_sample(self, sample_id: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load a specific sample by ID.
        
        Args:
            sample_id: Index of sample to load
            
        Returns:
            Tuple of (time_series, spectrogram, metadata)
        """
        if sample_id >= len(self.metadata):
            raise IndexError(f"Sample ID {sample_id} out of range")
        
        sample_meta = self.metadata[sample_id]
        
        # Load time series
        ts_path = Path(sample_meta['timeseries_path'])
        time_series = np.load(ts_path)
        
        # Load spectrogram
        spec_path = Path(sample_meta['spectrogram_path'])
        spectrogram = np.load(spec_path)
        
        return time_series, spectrogram, sample_meta
    
    def get_samples_by_type(self, event_type: str) -> List[int]:
        """
        Get sample IDs for a specific event type.
        
        Args:
            event_type: Event type to filter by
            
        Returns:
            List of sample IDs
        """
        sample_ids = []
        for i, sample in enumerate(self.metadata):
            if sample['event_type'] == event_type:
                sample_ids.append(i)
        
        return sample_ids
    
    def calculate_snr_estimates(self, sample_ids: List[int]) -> List[float]:
        """
        Calculate rough SNR estimates for given samples.
        
        Args:
            sample_ids: List of sample IDs to analyze
            
        Returns:
            List of SNR estimates
        """
        snr_estimates = []
        
        for sample_id in sample_ids:
            time_series, _, _ = self.load_sample(sample_id)
            
            # Simple SNR estimation based on signal power
            signal_power = np.mean(time_series**2)
            
            # Estimate noise power from beginning and end of signal
            edge_length = len(time_series) // 10
            noise_power = np.mean([
                np.mean(time_series[:edge_length]**2),
                np.mean(time_series[-edge_length:]**2)
            ])
            
            if noise_power > 0:
                snr = np.sqrt(signal_power / noise_power)
            else:
                snr = 0.0
            
            snr_estimates.append(snr)
        
        return snr_estimates


def create_visualization_report(data_dir: Path, output_dir: Path) -> None:
    """
    Create a comprehensive visualization report of the processed data.
    
    Args:
        data_dir: Directory containing processed data
        output_dir: Directory to save visualization report
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer and visualizer
    analyzer = DataAnalyzer(data_dir)
    visualizer = DataVisualizer()
    
    # Get statistics
    stats = analyzer.get_sample_statistics()
    
    # Create dataset statistics plot
    visualizer.plot_dataset_statistics(analyzer.metadata_path)
    
    # Create sample plots for each event type
    for event_type in stats['event_types'].keys():
        sample_ids = analyzer.get_samples_by_type(event_type)
        
        if sample_ids:
            # Plot first few samples of each type
            samples_to_plot = min(3, len(sample_ids))
            
            for i in range(samples_to_plot):
                sample_id = sample_ids[i]
                time_series, spectrogram, metadata = analyzer.load_sample(sample_id)
                
                # Plot time series
                visualizer.plot_timeseries(
                    time_series, 
                    metadata['sample_rate'],
                    f"{event_type} - {metadata['detector']} - Sample {i+1}",
                    output_dir / f"{event_type}_timeseries_{i+1}.png"
                )
                
                # Plot spectrogram
                visualizer.plot_spectrogram(
                    spectrogram,
                    f"{event_type} - {metadata['detector']} - Sample {i+1}",
                    output_dir / f"{event_type}_spectrogram_{i+1}.png"
                )
    
    logger.info(f"Visualization report created in {output_dir}")


if __name__ == "__main__":
    # Example usage
    data_dir = Path("./data/processed")
    output_dir = Path("./data/visualizations")
    
    if data_dir.exists():
        create_visualization_report(data_dir, output_dir)
    else:
        print(f"Data directory not found: {data_dir}")
        print("Please run the preprocessing pipeline first.")
