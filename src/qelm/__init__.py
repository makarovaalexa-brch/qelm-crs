"""
QELM-CRS: Question Embedding Learning with RL for Conversational Recommendation

A novel approach to conversational recommendation using reinforcement learning
with continuous embeddings in semantic space for question generation.
"""

__version__ = "0.1.0"
__author__ = "Aleksandra Makarova"
__email__ = "a.makarova@pgr.reading.ac.uk"

# Core modules
from qelm import models, data, training, utils

__all__ = [
    "models",
    "data",
    "training",
    "utils"
]