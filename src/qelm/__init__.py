"""
QELM-CRS: Question Embedding Learning for Conversational Recommendation Systems

A novel approach to conversational recommendation using reinforcement learning
with continuous action spaces for question generation.
"""

__version__ = "0.1.0"
__author__ = "Aleksandra Makarova"
__email__ = "a.makarova@pgr.reading.ac.uk"

from qelm import models, agents, environment, data, utils, evaluation

__all__ = [
    "models",
    "agents",
    "environment",
    "data",
    "utils",
    "evaluation"
]