"""
Neural network models for QELM-CRS system.

Includes:
- RL actor-critic for continuous embedding prediction
- Two-tower recommender system
- SentenceBERT embedding space
"""

from .embedding_qelm import (
    SentenceBERTEmbeddingSpace,
    EmbeddingActorCritic,
    QuestionGenerator,
    EmbeddingQLEM
)
from .two_tower_recommender import (
    TwoTowerRecommender,
    MovieCatalog,
    RecommenderTrainer,
    UserTower,
    ItemTower
)

__all__ = [
    # Embedding QELM
    "SentenceBERTEmbeddingSpace",
    "EmbeddingActorCritic",
    "QuestionGenerator",
    "EmbeddingQLEM",
    # Two-Tower Recommender
    "TwoTowerRecommender",
    "MovieCatalog",
    "RecommenderTrainer",
    "UserTower",
    "ItemTower"
]
