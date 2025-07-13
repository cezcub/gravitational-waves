"""
Signal processing utilities for gravitational wave data preprocessing.
Handles filtering, whitening, and spectrogram generation.
"""

import numpy as np
import scipy.signal
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, spectrogram
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SignalProcessor:
    """
    Handles signal processing operations for gravitational wave data.
    """
    
    def __init__(self, sample_rate: int = 4096):
        """
        Initialize the signal processor.
        
        Args:
            sample_rate: Sample rate of the data in Hz
        """
        self.sample_rate = sample_rate
        
    def bandpass_filter(self, 
                       data: np.ndarray, 
                       low_freq: float = 20.0, 
                       high_freq: float = 500.0, 
                       order: int = 4) -> np.ndarray:
        """
        Apply bandpass filter to the data.
        
        Args:
            data: Input time series data
            low_freq: Lower cutoff frequency in Hz
            high_freq: Upper cutoff frequency in Hz
            order: Filter order
            
        Returns:
            Filtered data
        """
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design Butterworth bandpass filter
        b, a = butter(order, [low, high], btype='band')
        
        # Apply filter
        filtered_data = filtfilt(b, a, data)
        
        logger.info(f"Applied bandpass filter: {low_freq}-{high_freq} Hz")
        return filtered_data
    
    def whiten_data(self, data: np.ndarray, window_length: int = 4096) -> np.ndarray:
        """
        Whiten the data by dividing by the noise PSD.
        
        Args:
            data: Input time series data
            window_length: Length of window for PSD estimation
            
        Returns:
            Whitened data
        """
        # Calculate power spectral density
        freqs, psd = scipy.signal.welch(
            data, 
            fs=self.sample_rate, 
            nperseg=window_length,
            noverlap=window_length//2
        )
        
        # Avoid division by zero
        psd[psd == 0] = np.finfo(float).eps
        
        # Take FFT of data
        data_fft = fft(data)
        freqs_fft = fftfreq(len(data), 1/self.sample_rate)
        
        # Interpolate PSD to match FFT frequencies
        psd_interp = np.interp(np.abs(freqs_fft), freqs, psd)
        
        # Whiten by dividing by sqrt(PSD)
        whitened_fft = data_fft / np.sqrt(psd_interp)
        
        # Take inverse FFT
        whitened_data = np.real(np.fft.ifft(whitened_fft))
        
        logger.info("Applied whitening to data")
        return whitened_data
    
    def resample_data(self, data: np.ndarray, target_rate: int) -> np.ndarray:
        """
        Resample data to target sample rate.
        
        Args:
            data: Input time series data
            target_rate: Target sample rate in Hz
            
        Returns:
            Resampled data
        """
        if target_rate == self.sample_rate:
            return data
            
        # Calculate resampling factor
        resample_factor = target_rate / self.sample_rate
        new_length = int(len(data) * resample_factor)
        
        # Use scipy's resample function
        resampled_data = scipy.signal.resample(data, new_length)
        
        logger.info(f"Resampled data from {self.sample_rate} Hz to {target_rate} Hz")
        return resampled_data
    
    def generate_spectrogram(self, 
                           data: np.ndarray, 
                           nperseg: int = 512, 
                           noverlap: Optional[int] = None,
                           target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Generate log-scaled spectrogram from time series data.
        
        Args:
            data: Input time series data
            nperseg: Length of each segment for STFT
            noverlap: Number of points to overlap between segments
            target_size: Target size for output spectrogram (height, width)
            
        Returns:
            Log-scaled spectrogram of shape target_size
        """
        if noverlap is None:
            noverlap = nperseg // 2
            
        # Calculate spectrogram
        frequencies, times, Sxx = spectrogram(
            data, 
            fs=self.sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann'
        )
        
        # Apply log scaling (add small epsilon to avoid log(0))
        log_spectrogram = np.log10(Sxx + 1e-10)
        
        # Resize to target size
        from scipy.ndimage import zoom
        
        height_factor = target_size[0] / log_spectrogram.shape[0]
        width_factor = target_size[1] / log_spectrogram.shape[1]
        
        resized_spectrogram = zoom(log_spectrogram, (height_factor, width_factor))
        
        logger.info(f"Generated spectrogram of size {target_size}")
        return resized_spectrogram
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to zero mean and unit variance.
        
        Args:
            data: Input data
            
        Returns:
            Normalized data
        """
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            logger.warning("Data has zero standard deviation, returning original data")
            return data
            
        normalized_data = (data - mean) / std
        
        logger.info("Normalized data to zero mean and unit variance")
        return normalized_data
    
    def process_strain_data(self, 
                           data: np.ndarray, 
                           low_freq: float = 20.0, 
                           high_freq: float = 500.0,
                           target_sample_rate: int = 2048,
                           normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete processing pipeline for strain data.
        
        Args:
            data: Raw strain data
            low_freq: Lower cutoff frequency
            high_freq: Upper cutoff frequency
            target_sample_rate: Target sample rate after processing
            normalize: Whether to normalize the data
            
        Returns:
            Tuple of (processed_time_series, spectrogram)
        """
        # Step 1: Bandpass filter
        filtered_data = self.bandpass_filter(data, low_freq, high_freq)
        
        # Step 2: Whiten the data
        whitened_data = self.whiten_data(filtered_data)
        
        # Step 3: Resample to target rate
        resampled_data = self.resample_data(whitened_data, target_sample_rate)
        
        # Step 4: Normalize if requested
        if normalize:
            time_series = self.normalize_data(resampled_data)
        else:
            time_series = resampled_data
            
        # Step 5: Generate spectrogram
        spectrogram = self.generate_spectrogram(time_series)
        
        logger.info("Completed full strain data processing pipeline")
        return time_series, spectrogram
