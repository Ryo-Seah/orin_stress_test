"""
Backward compatibility wrapper for the refactored data processing classes.
This file maintains compatibility with existing code while using the new modular structure.
"""

# Import the new modular classes
from data_processing import (
    BOLDDataset,
    PoseFormatConverter, 
    PoseFeatureExtractor,
    StressScoreCalculator
)

# For backward compatibility, alias the new class to the old name
BOLDDataLoader = BOLDDataset

# Expose all classes for flexible usage
__all__ = [
    'BOLDDataLoader',  # Backward compatibility
    'BOLDDataset',     # New name
    'PoseFormatConverter',
    'PoseFeatureExtractor', 
    'StressScoreCalculator'
]

# Optional: Add deprecation warning for old usage
import warnings

def create_bold_data_loader(*args, **kwargs):
    """Factory function with deprecation warning."""
    warnings.warn(
        "Direct instantiation of BOLDDataLoader is deprecated. "
        "Consider using 'from data_processing import BOLDDataset' for new code.",
        DeprecationWarning,
        stacklevel=2
    )
    return BOLDDataset(*args, **kwargs)

# You can uncomment this line if you want deprecation warnings:
# BOLDDataLoader = create_bold_data_loader