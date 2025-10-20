"""
Two-Tower Recommender for QELM.

Takes the same dialogue state as RL model (384-dim SentenceBERT) and recommends movies.

Architecture:
    User Tower: Dialogue State → User Embedding (128-dim)
    Item Tower: Movie Metadata → Movie Embedding (128-dim)
    Score: Dot Product(user_emb, movie_emb)

Training: BPR loss on (positive, negative) pairs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import pandas as pd
from pathlib import Path
import random


class UserTower(nn.Module):
    """
    User tower: Maps dialogue state to user embedding.

    Input: Conversation state (384-dim from SentenceBERT)
    Output: User embedding (128-dim)
    """

    def __init__(self, state_dim: int = 384, user_emb_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, user_emb_dim),
            nn.LayerNorm(user_emb_dim)  # Normalize for dot product
        )

    def forward(self, state):
        """
        Args:
            state: Dialogue state tensor [batch_size, 384]

        Returns:
            User embedding [batch_size, 128]
        """
        return self.network(state)


class ItemTower(nn.Module):
    """
    Item tower: Maps movie metadata to item embedding.

    Input: Movie title + genres encoded with SentenceBERT (384-dim)
    Output: Item embedding (128-dim)
    """

    def __init__(self, item_dim: int = 384, item_emb_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(item_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, item_emb_dim),
            nn.LayerNorm(item_emb_dim)  # Normalize for dot product
        )

    def forward(self, item_features):
        """
        Args:
            item_features: Movie feature tensor [batch_size, 384]

        Returns:
            Item embedding [batch_size, 128]
        """
        return self.network(item_features)


class TwoTowerRecommender(nn.Module):
    """
    Two-Tower recommendation model.

    Shares the same dialogue state representation as RL model.
    """

    def __init__(
        self,
        state_dim: int = 384,
        embedding_dim: int = 128,
        learning_rate: float = 0.001
    ):
        super().__init__()

        self.user_tower = UserTower(state_dim, embedding_dim)
        self.item_tower = ItemTower(state_dim, embedding_dim)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

    def forward(self, user_states, item_features):
        """
        Compute scores for user-item pairs.

        Args:
            user_states: [batch_size, 384] dialogue states
            item_features: [batch_size, 384] movie features

        Returns:
            scores: [batch_size] dot product scores
        """
        user_emb = self.user_tower(user_states)
        item_emb = self.item_tower(item_features)

        # Dot product for scoring
        scores = (user_emb * item_emb).sum(dim=1)

        return scores

    def predict_scores(self, user_state, candidate_items):
        """
        Predict scores for all candidate items.

        Args:
            user_state: [384] single dialogue state
            candidate_items: [num_items, 384] candidate movie features

        Returns:
            scores: [num_items] predicted scores
        """
        with torch.no_grad():
            # Expand user state to match batch size
            user_states = user_state.unsqueeze(0).expand(len(candidate_items), -1)

            # Get scores
            scores = self.forward(user_states, candidate_items)

        return scores

    def train_bpr_step(
        self,
        user_states,
        positive_items,
        negative_items
    ):
        """
        Training step with BPR (Bayesian Personalized Ranking) loss.

        BPR assumes positive items should be ranked higher than negative items.

        Args:
            user_states: [batch_size, 384] dialogue states
            positive_items: [batch_size, 384] liked movie features
            negative_items: [batch_size, 384] disliked/random movie features

        Returns:
            loss value
        """
        # Get scores
        pos_scores = self.forward(user_states, positive_items)
        neg_scores = self.forward(user_states, negative_items)

        # BPR loss: encourage pos_score > neg_score
        # loss = -log(sigmoid(pos_score - neg_score))
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()


class MovieCatalog:
    """
    Handles movie metadata and feature encoding.

    Encodes movies using SentenceBERT for the item tower.
    """

    def __init__(self, movielens_data_path: Optional[str] = None):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Load MovieLens data
        self.movies = self._load_movies(movielens_data_path)

        # Pre-compute movie embeddings
        self._encode_movies()

        print(f"✓ Loaded {len(self.movies)} movies")

    def _load_movies(self, data_path: Optional[str]) -> pd.DataFrame:
        """Load MovieLens movies."""
        if data_path:
            try:
                # Try filtered dataset first (subset of top movies)
                movies_path = Path(data_path) / "movies_filtered.csv"
                if movies_path.exists():
                    movies = pd.read_csv(movies_path)
                    print(f"Using filtered dataset: {len(movies)} movies")
                    return movies

                # Fall back to full dataset
                movies_path = Path(data_path) / "movies.csv"
                if movies_path.exists():
                    movies = pd.read_csv(movies_path)
                    return movies
            except Exception as e:
                print(f"Could not load MovieLens: {e}")

        # Sample data for development
        return pd.DataFrame({
            'movieId': [1, 2, 3, 4, 5],
            'title': [
                'The Dark Knight (2008)',
                'Inception (2010)',
                'Pulp Fiction (1994)',
                'The Matrix (1999)',
                'Interstellar (2014)'
            ],
            'genres': [
                'Action|Crime|Drama',
                'Action|Sci-Fi|Thriller',
                'Crime|Drama',
                'Action|Sci-Fi',
                'Adventure|Drama|Sci-Fi'
            ]
        })

    def _encode_movies(self):
        """Pre-encode all movies with SentenceBERT."""
        movie_texts = []

        for idx, row in self.movies.iterrows():
            # Create text representation: title + genres
            title = row['title'].split('(')[0].strip()  # Remove year
            genres = row['genres'].replace('|', ', ')
            text = f"{title}: {genres}"
            movie_texts.append(text)

        # Encode all movies
        self.movie_embeddings = self.encoder.encode(
            movie_texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        # Convert to tensor
        self.movie_embeddings_tensor = torch.FloatTensor(self.movie_embeddings)

    def get_movie_features(self, movie_ids: List[int]) -> torch.Tensor:
        """Get movie features by IDs."""
        indices = [self.movies[self.movies['movieId'] == mid].index[0] for mid in movie_ids]
        return self.movie_embeddings_tensor[indices]

    def get_all_movie_features(self) -> torch.Tensor:
        """Get all movie features."""
        return self.movie_embeddings_tensor

    def get_movie_ids(self) -> List[int]:
        """Get all movie IDs."""
        return self.movies['movieId'].tolist()


class RecommenderTrainer:
    """
    Trainer for two-tower recommender.

    Trains on conversation → liked movies pairs.
    """

    def __init__(
        self,
        recommender: TwoTowerRecommender,
        movie_catalog: MovieCatalog,
        encoder: SentenceTransformer
    ):
        self.recommender = recommender
        self.movie_catalog = movie_catalog
        self.encoder = encoder

    def create_training_batch(
        self,
        conversations: List[str],
        liked_movies: List[List[int]],
        batch_size: int = 32
    ):
        """
        Create training batches from conversation-movie pairs.

        Args:
            conversations: List of conversation texts
            liked_movies: List of lists of liked movie IDs
            batch_size: Batch size

        Yields:
            (user_states, positive_items, negative_items) tensors
        """
        all_movie_ids = self.movie_catalog.get_movie_ids()

        for i in range(0, len(conversations), batch_size):
            batch_convs = conversations[i:i+batch_size]
            batch_likes = liked_movies[i:i+batch_size]

            # Encode conversations
            user_states = self.encoder.encode(
                batch_convs,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            user_states = torch.FloatTensor(user_states)

            # Get positive and negative items
            positive_ids = []
            negative_ids = []

            for likes in batch_likes:
                # Sample one positive
                pos_id = random.choice(likes) if likes else random.choice(all_movie_ids)
                positive_ids.append(pos_id)

                # Sample one negative (not in likes)
                neg_candidates = [mid for mid in all_movie_ids if mid not in likes]
                neg_id = random.choice(neg_candidates) if neg_candidates else random.choice(all_movie_ids)
                negative_ids.append(neg_id)

            # Get movie features
            positive_items = self.movie_catalog.get_movie_features(positive_ids)
            negative_items = self.movie_catalog.get_movie_features(negative_ids)

            yield user_states, positive_items, negative_items

    def train(
        self,
        conversations: List[str],
        liked_movies: List[List[int]],
        epochs: int = 10,
        batch_size: int = 32
    ):
        """
        Train recommender on conversation data.

        Args:
            conversations: List of conversation texts
            liked_movies: List of lists of liked movie IDs per conversation
            epochs: Number of training epochs
            batch_size: Batch size
        """
        print(f"\n{'='*60}")
        print("TRAINING TWO-TOWER RECOMMENDER")
        print(f"{'='*60}")
        print(f"Training examples: {len(conversations)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}\n")

        for epoch in range(epochs):
            epoch_losses = []

            # Create batches
            for user_states, pos_items, neg_items in self.create_training_batch(
                conversations, liked_movies, batch_size
            ):
                loss = self.recommender.train_bpr_step(
                    user_states, pos_items, neg_items
                )
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}\n")

    def recommend(
        self,
        conversation_state: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[int, str, float]]:
        """
        Get top-k movie recommendations for a conversation state.

        Args:
            conversation_state: [384] dialogue state from SentenceBERT
            top_k: Number of recommendations

        Returns:
            List of (movie_id, title, score) tuples
        """
        # Convert to tensor
        state_tensor = torch.FloatTensor(conversation_state)

        # Get all movie features
        all_movie_features = self.movie_catalog.get_all_movie_features()

        # Predict scores
        scores = self.recommender.predict_scores(state_tensor, all_movie_features)

        # Get top-k
        top_indices = torch.argsort(scores, descending=True)[:top_k]

        # Get movie details
        recommendations = []
        for idx in top_indices:
            idx = idx.item()
            movie_id = self.movie_catalog.movies.iloc[idx]['movieId']
            title = self.movie_catalog.movies.iloc[idx]['title']
            score = scores[idx].item()
            recommendations.append((movie_id, title, score))

        return recommendations


# Example usage
if __name__ == "__main__":
    print("Testing Two-Tower Recommender...")

    # Initialize components
    movie_catalog = MovieCatalog(movielens_data_path=None)  # Use sample data

    recommender = TwoTowerRecommender(
        state_dim=384,
        embedding_dim=128
    )

    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    trainer = RecommenderTrainer(recommender, movie_catalog, encoder)

    # Sample training data
    sample_conversations = [
        "I love action movies with great cinematography like The Dark Knight",
        "I enjoy mind-bending sci-fi films like Inception",
        "I prefer dark crime dramas with great dialogue",
    ]

    sample_liked_movies = [
        [1],  # The Dark Knight
        [2],  # Inception
        [3],  # Pulp Fiction
    ]

    # Train
    trainer.train(
        conversations=sample_conversations,
        liked_movies=sample_liked_movies,
        epochs=5,
        batch_size=2
    )

    # Test recommendation
    test_conversation = "I want something like Inception with sci-fi elements"
    test_state = encoder.encode(test_conversation, convert_to_numpy=True)

    recommendations = trainer.recommend(test_state, top_k=3)

    print("\nRecommendations for:", test_conversation)
    for movie_id, title, score in recommendations:
        print(f"  {title}: {score:.3f}")
