"""
Stage 1: Supervised Pretraining

Train RL actor to predict SentenceBERT embeddings from conversation context.

Goal: Learn the mapping from conversation context → relevant concept embeddings
BEFORE reward-based RL training.

Training data: Reddit question-concept pairs
Loss: MSE between predicted embedding and target concept embeddings
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import openai

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.embedding_qelm import SentenceBERTEmbeddingSpace, EmbeddingActorCritic


class ConceptExtractor:
    """
    Extracts movie concepts from Reddit posts using GPT.

    Converts post text → list of relevant movie concepts.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.cache = {}

    def extract_concepts(self, text: str) -> List[str]:
        """
        Extract movie concepts from text.

        Args:
            text: Reddit post text

        Returns:
            List of concept strings (movies, directors, genres, themes)
        """
        # Check cache
        if text in self.cache:
            return self.cache[text]

        prompt = f"""
Extract movie-related concepts from this text.
Return ONLY a JSON list of concepts (movies, directors, actors, genres, themes).

Text: {text[:500]}

Example output: ["Inception", "Nolan", "sci-fi", "mind-bending", "thriller"]

JSON list:"""

        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You extract movie concepts. Return JSON lists only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            concepts = json.loads(content)

            # Cache result
            self.cache[text] = concepts

            return concepts

        except Exception as e:
            print(f"Concept extraction error: {e}")
            # Fallback: extract capitalized words
            words = text.split()
            concepts = [w.strip('.,!?') for w in words if len(w) > 2 and w[0].isupper()]
            return concepts[:5]  # Limit to 5


class Stage1Dataset:
    """
    Dataset for Stage 1 supervised training.

    Loads Reddit posts and converts to (context, target_embedding) pairs.
    """

    def __init__(
        self,
        reddit_data_path: str,
        embedding_space: SentenceBERTEmbeddingSpace,
        concept_extractor: ConceptExtractor,
        encoder: SentenceTransformer
    ):
        self.embedding_space = embedding_space
        self.concept_extractor = concept_extractor
        self.encoder = encoder

        # Load Reddit data
        self.data = self._load_reddit_data(reddit_data_path)

        # Process into training examples
        self.examples = self._create_training_examples()

        print(f"Created {len(self.examples)} training examples")

    def _load_reddit_data(self, data_path: str) -> List[Dict]:
        """Load Reddit posts from JSON files."""
        data_path = Path(data_path)

        if not data_path.exists():
            print(f"Reddit data not found at {data_path}")
            print("Creating sample data for development...")
            return self._create_sample_data()

        # Load all JSON files in directory
        all_posts = []
        for json_file in data_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    posts = json.load(f)
                    all_posts.extend(posts)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        return all_posts

    def _create_sample_data(self) -> List[Dict]:
        """Create sample training data for development."""
        return [
            {
                "title": "Looking for movies like Inception",
                "text": "I loved Inception's mind-bending plot. Any recommendations?",
                "concepts": ["Inception", "mind-bending", "sci-fi", "Nolan"]
            },
            {
                "title": "Best Tarantino films?",
                "text": "I've seen Pulp Fiction. What other Tarantino movies should I watch?",
                "concepts": ["Tarantino", "Pulp Fiction", "crime", "dialogue"]
            },
            {
                "title": "Dark psychological thrillers",
                "text": "Something dark and psychological like Shutter Island",
                "concepts": ["psychological", "thriller", "dark", "suspenseful"]
            },
            {
                "title": "Great action movies",
                "text": "Looking for intense action with good plot. Like Mad Max Fury Road.",
                "concepts": ["action", "intense", "Mad Max", "explosions"]
            },
            {
                "title": "Emotional dramas",
                "text": "Need a good cry. Something like Schindler's List or Room.",
                "concepts": ["drama", "emotional", "sad", "powerful"]
            },
        ]

    def _create_training_examples(self) -> List[Dict]:
        """
        Convert Reddit conversation pairs to training examples.

        NEW FORMAT (conversation pairs):
        Input: User post ("I love Inception")
        Target: Concepts from RESPONSE ("Have you seen Interstellar?")
        → ["Interstellar", "Nolan", "sci-fi"]

        This teaches: User preference → What to ask NEXT
        """
        examples = []

        for post in tqdm(self.data, desc="Processing Reddit conversations"):
            # NEW: Handle conversation pair format
            if "user_post" in post and "response" in post:
                # Input context = user's statement
                context = post["user_post"]

                # Target concepts = from the RESPONSE (what to ask next)
                if "response_concepts" in post:
                    concepts = post["response_concepts"]
                else:
                    # Extract concepts from the response
                    concepts = self.concept_extractor.extract_concepts(post["response"])

            # LEGACY: Old format (single posts) - for backwards compatibility
            else:
                text = post.get("title", "") + " " + post.get("text", "")
                context = text

                if "concepts" in post:
                    concepts = post["concepts"]
                else:
                    concepts = self.concept_extractor.extract_concepts(context)

            if not context.strip():
                continue

            if not concepts:
                continue

            # Get SentenceBERT embeddings for concepts
            concept_embeddings = []
            valid_concepts = []

            for concept in concepts:
                emb = self.embedding_space.get_embedding(concept)
                if emb is not None:
                    concept_embeddings.append(emb)
                    valid_concepts.append(concept)

            if not concept_embeddings:
                continue

            # Target: average of concept embeddings
            target_embedding = np.mean(concept_embeddings, axis=0)

            # Create training example
            examples.append({
                "context": context[:500],  # Truncate long posts
                "concepts": valid_concepts,
                "target_embedding": target_embedding
            })

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class Stage1Trainer:
    """
    Trains RL actor with supervised learning on question-concept pairs.

    After training, RL will predict embeddings in the right semantic space.
    """

    def __init__(
        self,
        rl_agent: EmbeddingActorCritic,
        embedding_space: SentenceBERTEmbeddingSpace,
        reddit_data_path: str,
        encoder: SentenceTransformer
    ):
        self.rl_agent = rl_agent
        self.embedding_space = embedding_space
        self.encoder = encoder

        # Create dataset
        concept_extractor = ConceptExtractor()
        self.dataset = Stage1Dataset(
            reddit_data_path,
            embedding_space,
            concept_extractor,
            encoder
        )

        # Training metrics
        self.train_losses = []

    def train(
        self,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Train RL actor with supervised learning.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        print(f"\n{'='*60}")
        print("STAGE 1: SUPERVISED PRETRAINING")
        print(f"{'='*60}")
        print(f"Dataset size: {len(self.dataset)} examples")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}\n")

        optimizer = torch.optim.Adam(
            self.rl_agent.actor.parameters(),
            lr=learning_rate
        )

        for epoch in range(epochs):
            epoch_losses = []

            # Shuffle dataset
            indices = np.random.permutation(len(self.dataset))

            # Train in batches
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_examples = [self.dataset[idx] for idx in batch_indices]

                # Encode contexts
                contexts = [ex["context"] for ex in batch_examples]
                state_embeddings = self.encoder.encode(
                    contexts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )

                # Get target embeddings
                target_embeddings = np.array([
                    ex["target_embedding"] for ex in batch_examples
                ])

                # Convert to tensors
                states = torch.FloatTensor(state_embeddings)
                targets = torch.FloatTensor(target_embeddings)

                # Forward pass
                predicted_embeddings = self.rl_agent.actor(states)

                # Normalize predictions (for cosine similarity later)
                predicted_normalized = F.normalize(predicted_embeddings, dim=1)
                targets_normalized = F.normalize(targets, dim=1)

                # Loss: MSE in normalized space
                loss = F.mse_loss(predicted_normalized, targets_normalized)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rl_agent.actor.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())

            # Epoch summary
            avg_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_loss)

            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"stage1_epoch{epoch+1}.pt")

        print(f"\n{'='*60}")
        print("STAGE 1 TRAINING COMPLETE")
        print(f"Final loss: {self.train_losses[-1]:.4f}")
        print(f"{'='*60}\n")

    def evaluate(self, num_samples: int = 5):
        """
        Evaluate trained model on sample examples.

        Shows what concepts RL predicts for different contexts.
        """
        print(f"\n{'='*60}")
        print("STAGE 1 EVALUATION")
        print(f"{'='*60}\n")

        # Sample random examples
        indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)

        for idx in indices:
            example = self.dataset[idx]

            # Encode context
            state = self.encoder.encode(example["context"], convert_to_numpy=True)

            # Predict embedding
            predicted_embedding = self.rl_agent.predict_embedding(state, explore=False)

            # Find nearest concepts
            nearest = self.embedding_space.find_nearest_entities(predicted_embedding, top_k=5)

            print(f"Context: {example['context'][:100]}...")
            print(f"Target concepts: {example['concepts']}")
            print(f"Predicted concepts: {[c for c, s in nearest]}")
            print(f"Similarities: {[f'{s:.3f}' for c, s in nearest]}")
            print()

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint_path = Path("checkpoints") / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'actor_state_dict': self.rl_agent.actor.state_dict(),
            'train_losses': self.train_losses,
        }, checkpoint_path)

        print(f"Saved checkpoint: {checkpoint_path}")


# Main training script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1: Supervised Pretraining")
    parser.add_argument("--reddit-data", type=str, default="data/reddit", help="Reddit data directory")
    parser.add_argument("--movielens-data", type=str, default=None, help="MovieLens data directory")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    # Initialize components
    print("Initializing SentenceBERT embedding space...")
    embedding_space = SentenceBERTEmbeddingSpace(movielens_data_path=args.movielens_data)

    print("Initializing RL actor...")
    rl_agent = EmbeddingActorCritic(
        state_dim=384,  # SentenceBERT
        embedding_dim=384  # SentenceBERT
    )

    print("Initializing encoder...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    # Create trainer
    trainer = Stage1Trainer(
        rl_agent=rl_agent,
        embedding_space=embedding_space,
        reddit_data_path=args.reddit_data,
        encoder=encoder
    )

    # Train
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    # Evaluate
    trainer.evaluate(num_samples=5)

    # Save final model
    trainer.save_checkpoint("stage1_final.pt")
