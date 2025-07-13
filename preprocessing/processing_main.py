import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging

# Import our preprocessing pipeline
try:
    # Try relative imports first (when imported as a module)
    from . import GWPreprocessingPipeline
    from .config import (
        SIGNAL_PROCESSING_CONFIG,
        AUGMENTATION_CONFIG,
        DETECTOR_CONFIG,
        PROCESSED_DATA_DIR,
        LOGGING_CONFIG
    )
except ImportError:
    # Fallback to absolute imports (when run directly)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from preprocessing import GWPreprocessingPipeline
    from preprocessing.config import (
        SIGNAL_PROCESSING_CONFIG,
        AUGMENTATION_CONFIG,
        DETECTOR_CONFIG,
        PROCESSED_DATA_DIR,
        LOGGING_CONFIG
    )

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['log_file']),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the gravitational wave preprocessing pipeline.
    """
    logger.info("Starting Gravitational Wave Deep Learning Project")
    logger.info("=" * 60)
    
    # Initialize preprocessing pipeline
    logger.info("Initializing preprocessing pipeline...")
    
    pipeline = GWPreprocessingPipeline(
        output_dir=str(PROCESSED_DATA_DIR),
        sample_rate=SIGNAL_PROCESSING_CONFIG['sample_rate'],
        target_sample_rate=SIGNAL_PROCESSING_CONFIG['target_sample_rate'],
        duration=SIGNAL_PROCESSING_CONFIG['duration']
    )
    
    logger.info("Pipeline configuration:")
    logger.info(f"  • Output directory: {PROCESSED_DATA_DIR}")
    logger.info(f"  • Sample rate: {SIGNAL_PROCESSING_CONFIG['sample_rate']} Hz")
    logger.info(f"  • Target sample rate: {SIGNAL_PROCESSING_CONFIG['target_sample_rate']} Hz")
    logger.info(f"  • Segment duration: {SIGNAL_PROCESSING_CONFIG['duration']} seconds")
    logger.info(f"  • Detectors: {DETECTOR_CONFIG['detectors']}")
    logger.info(f"  • Frequency band: {SIGNAL_PROCESSING_CONFIG['low_freq']}-{SIGNAL_PROCESSING_CONFIG['high_freq']} Hz")
    
    # Run the full preprocessing pipeline
    logger.info("Running full preprocessing pipeline...")
    
    try:
        pipeline.run_full_pipeline(
            num_noise_segments=AUGMENTATION_CONFIG['num_noise_segments'],
            num_bns_injections=AUGMENTATION_CONFIG['num_bns_injections'],
            num_bbh_injections=AUGMENTATION_CONFIG['num_bbh_injections']
        )
        
        logger.info("✓ Preprocessing pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"✗ Error in preprocessing pipeline: {e}")
        raise
    
    # Future: Add model training code here
    logger.info("Preprocessing complete. Ready for model training.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()