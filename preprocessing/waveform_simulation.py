"""
Waveform simulation utilities for data augmentation.
Generates synthetic gravitational wave signals for training.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class WaveformSimulator:
    """
    Generates synthetic gravitational wave signals for data augmentation.
    """
    
    def __init__(self, sample_rate: int = 4096):
        """
        Initialize the waveform simulator.
        
        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.G = 6.67430e-11  # Gravitational constant
        self.c = 299792458    # Speed of light
        self.M_sun = 1.989e30 # Solar mass
        
    def chirp_mass(self, m1: float, m2: float) -> float:
        """
        Calculate chirp mass from component masses.
        
        Args:
            m1: Mass of first object in solar masses
            m2: Mass of second object in solar masses
            
        Returns:
            Chirp mass in solar masses
        """
        return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    
    def generate_inspiral_waveform(self, 
                                  m1: float, 
                                  m2: float, 
                                  duration: float = 4.0,
                                  f_low: float = 20.0,
                                  distance: float = 100.0,
                                  phase: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a simplified inspiral waveform using post-Newtonian approximation.
        
        Args:
            m1: Mass of first object in solar masses
            m2: Mass of second object in solar masses
            duration: Duration of waveform in seconds
            f_low: Starting frequency in Hz
            distance: Distance to source in Mpc
            phase: Initial phase
            
        Returns:
            Tuple of (time_array, strain_array)
        """
        # Time array
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        
        # Convert masses to kg
        m1_kg = m1 * self.M_sun
        m2_kg = m2 * self.M_sun
        M_total = m1_kg + m2_kg
        
        # Chirp mass
        M_c = self.chirp_mass(m1, m2) * self.M_sun
        
        # Reduced mass
        mu = m1_kg * m2_kg / M_total
        
        # Time to coalescence (simplified)
        tau = t[-1] - t
        
        # Frequency evolution (leading order)
        f_gw = f_low * (1 + (tau / self._tau_0(M_c, f_low))**(-3/8))
        
        # Phase evolution
        phi = phase + 2 * np.pi * np.cumsum(f_gw) / self.sample_rate
        
        # Amplitude (simplified)
        r_meters = distance * 3.086e22  # Convert Mpc to meters
        h0 = (4 * (self.G * M_c / self.c**2)**(5/3) * 
              (np.pi * f_gw / self.c)**(2/3) * self.c / r_meters)
        
        # Apply amplitude taper to avoid discontinuities
        taper_samples = int(0.1 * self.sample_rate)  # 0.1 second taper
        taper = np.ones_like(h0)
        taper[:taper_samples] = np.sin(np.linspace(0, np.pi/2, taper_samples))**2
        taper[-taper_samples:] = np.cos(np.linspace(0, np.pi/2, taper_samples))**2
        
        # Generate plus and cross polarizations
        h_plus = h0 * np.cos(phi) * taper
        h_cross = h0 * np.sin(phi) * taper
        
        # For simplicity, return only plus polarization
        strain = h_plus
        
        logger.info(f"Generated inspiral waveform: m1={m1}, m2={m2}, duration={duration}s")
        return t, strain
    
    def _tau_0(self, M_c: float, f_low: float) -> float:
        """
        Calculate time to coalescence parameter.
        
        Args:
            M_c: Chirp mass in kg
            f_low: Starting frequency in Hz
            
        Returns:
            Time parameter in seconds
        """
        return (5 * self.c**5) / (256 * np.pi * (self.G * M_c)**(5/3) * 
                                 (np.pi * f_low)**(8/3))
    
    def generate_bns_waveform(self, duration: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a typical Binary Neutron Star (BNS) waveform.
        
        Args:
            duration: Duration of waveform in seconds
            
        Returns:
            Tuple of (time_array, strain_array)
        """
        # Typical BNS parameters
        m1 = np.random.uniform(1.0, 2.0)  # Solar masses
        m2 = np.random.uniform(1.0, 2.0)  # Solar masses
        distance = np.random.uniform(50, 200)  # Mpc
        
        return self.generate_inspiral_waveform(m1, m2, duration, distance=distance)
    
    def generate_bbh_waveform(self, duration: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a typical Binary Black Hole (BBH) waveform.
        
        Args:
            duration: Duration of waveform in seconds
            
        Returns:
            Tuple of (time_array, strain_array)
        """
        # Typical BBH parameters
        m1 = np.random.uniform(10, 80)  # Solar masses
        m2 = np.random.uniform(10, 80)  # Solar masses
        distance = np.random.uniform(200, 1000)  # Mpc
        
        return self.generate_inspiral_waveform(m1, m2, duration, distance=distance)
    
    def inject_signal_into_noise(self, 
                                noise: np.ndarray, 
                                signal: np.ndarray, 
                                snr_target: float = 10.0) -> np.ndarray:
        """
        Inject a gravitational wave signal into noise at a specific SNR.
        
        Args:
            noise: Background noise array
            signal: Signal to inject
            snr_target: Target signal-to-noise ratio
            
        Returns:
            Noise + signal data
        """
        # Ensure signal and noise have same length
        min_length = min(len(noise), len(signal))
        noise = noise[:min_length]
        signal = signal[:min_length]
        
        # Calculate current SNR
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)
        current_snr = np.sqrt(signal_power / noise_power)
        
        # Scale signal to achieve target SNR
        scale_factor = snr_target / current_snr
        scaled_signal = signal * scale_factor
        
        # Add signal to noise
        injected_data = noise + scaled_signal
        
        logger.info(f"Injected signal with SNR = {snr_target:.1f}")
        return injected_data
    
    def generate_augmented_dataset(self, 
                                  noise_segments: List[np.ndarray], 
                                  signal_type: str = "BNS",
                                  num_injections: int = 100,
                                  snr_range: Tuple[float, float] = (8.0, 20.0)) -> List[Tuple[np.ndarray, str]]:
        """
        Generate augmented dataset by injecting signals into noise.
        
        Args:
            noise_segments: List of noise segments
            signal_type: Type of signal to inject ("BNS" or "BBH")
            num_injections: Number of injections to perform
            snr_range: Range of SNR values to use
            
        Returns:
            List of (injected_data, label) tuples
        """
        augmented_data = []
        
        for i in range(num_injections):
            # Select random noise segment
            noise = np.random.choice(noise_segments)
            
            # Generate signal
            if signal_type.upper() == "BNS":
                _, signal = self.generate_bns_waveform(duration=len(noise)/self.sample_rate)
            elif signal_type.upper() == "BBH":
                _, signal = self.generate_bbh_waveform(duration=len(noise)/self.sample_rate)
            else:
                raise ValueError(f"Unknown signal type: {signal_type}")
            
            # Random SNR
            snr = np.random.uniform(*snr_range)
            
            # Inject signal
            injected_data = self.inject_signal_into_noise(noise, signal, snr)
            
            augmented_data.append((injected_data, signal_type))
            
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{num_injections} {signal_type} injections")
        
        return augmented_data
