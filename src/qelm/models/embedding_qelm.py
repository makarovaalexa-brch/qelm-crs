"""
Embedding-Based QELM: RL outputs continuous embeddings in SentenceBERT space.

Key innovation: RL predicts embeddings in SentenceBERT space (384-dim) that map to
movie entities (titles, actors, directors, genres). No vocabulary limitations.

Architecture:
Conversation → RL Actor → SentenceBERT Embedding → Nearest Entities → GPT → Question

Advantages over Word2Vec:
- Handles multi-word entities: "The Dark Knight", "Leonardo DiCaprio"
- No vocabulary limitations (embeds any text)
- Better semantic similarity for movie domain
- Consistent 384-dim space throughout system
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
import openai
from sentence_transformers import SentenceTransformer
import random
from pathlib import Path
import pandas as pd


class SentenceBERTEmbeddingSpace:
    """
    SentenceBERT embedding space for movie entities.

    Handles multi-word entities: "The Dark Knight", "Christopher Nolan"
    No vocabulary limitations - can embed any text on-the-fly.
    """

    def __init__(self, movielens_data_path: Optional[str] = None):
        """
        Initialize SentenceBERT embedding space.

        Args:
            movielens_data_path: Path to MovieLens data for entity extraction
        """
        print("Initializing SentenceBERT embedding space...")

        # Load SentenceBERT
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384

        print(f"✓ Loaded SentenceBERT: {self.embedding_dim}-dim")

        # Extract MovieLens entities (movies, actors, directors)
        self.movie_entities = self._extract_movielens_entities(movielens_data_path)
        print(f"✓ Extracted {len(self.movie_entities)} MovieLens entities")

        # Pre-compute entity embeddings for fast lookup
        self._cache_entity_embeddings()

    def _extract_movielens_entities(self, data_path: Optional[str]) -> List[str]:
        """
        Extract movie titles, genres, and other entities from MovieLens data.

        Returns list of entity strings that can be used as concepts.
        """
        entities = []

        # Add common movie-related concepts
        entities.extend([
            # Genres
            "action", "adventure", "animation", "comedy", "crime", "documentary",
            "drama", "fantasy", "horror", "mystery", "romance", "sci-fi",
            "thriller", "western", "musical", "war", "biographical",

            # Moods & Themes
            "dark", "uplifting", "emotional", "intense", "light-hearted",
            "cerebral", "violent", "peaceful", "suspenseful", "funny",
            "sad", "inspiring", "disturbing", "thought-provoking",

            # Directors
            "Christopher Nolan", "Quentin Tarantino", "Steven Spielberg",
            "Martin Scorsese", "Stanley Kubrick", "Alfred Hitchcock",
            "David Fincher", "Wes Anderson", "David Lynch", "Francis Ford Coppola",

            # Actors
            "Leonardo DiCaprio", "Tom Hanks", "Meryl Streep", "Denzel Washington",
            "Brad Pitt", "Scarlett Johansson", "Samuel L. Jackson", "Morgan Freeman",

            # Technical
            "cinematography", "soundtrack", "dialogue", "visual effects",
            "acting", "plot twist", "character development", "pacing", "atmosphere",

            # Time periods
            "classic film", "modern film", "recent release", "vintage cinema",

            # Specific interests
            "based on true story", "book adaptation", "superhero movie",
            "space exploration", "war film", "crime thriller", "family movie",
            "friendship", "redemption story", "revenge plot"
        ])

        # Try to load actual MovieLens movie titles
        if data_path:
            try:
                movies_path = Path(data_path) / "movies.csv"
                if movies_path.exists():
                    movies_df = pd.read_csv(movies_path)

                    # Extract movie titles (clean them)
                    for title in movies_df['title']:
                        # Remove year (e.g., "Inception (2010)" -> "Inception")
                        clean_title = title.split('(')[0].strip()
                        entities.append(clean_title)

                    print(f"✓ Loaded {len(movies_df)} movie titles from MovieLens")
            except Exception as e:
                print(f"Note: Could not load MovieLens movies: {e}")

        return entities

    def _cache_entity_embeddings(self):
        """Pre-compute embeddings for all known entities."""
        print("Pre-computing entity embeddings...")

        # Encode all entities at once (batch encoding is faster)
        self.entity_list = self.movie_entities
        self.entity_embeddings = self.encoder.encode(
            self.entity_list,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        print(f"✓ Cached {len(self.entity_list)} entity embeddings")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get SentenceBERT embedding for any text.

        Works for:
        - Single words: "action"
        - Multi-word entities: "The Dark Knight"
        - Phrases: "intense psychological thriller"
        """
        return self.encoder.encode(text, convert_to_numpy=True)

    def find_nearest_entities(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Find nearest movie entities to a given embedding.

        Args:
            embedding: Query embedding (384-dim)
            top_k: How many entities to return
            threshold: Minimum similarity threshold

        Returns:
            List of (entity, similarity) tuples
        """
        if len(self.entity_list) == 0:
            return []

        # Normalize query embedding
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        # Normalize entity embeddings
        entity_norms = np.linalg.norm(self.entity_embeddings, axis=1, keepdims=True)
        entity_matrix_norm = self.entity_embeddings / (entity_norms + 1e-8)

        # Compute cosine similarities
        similarities = np.dot(entity_matrix_norm, embedding_norm)

        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= threshold:
                entity = self.entity_list[idx]
                results.append((entity, float(similarity)))

        # If no entities above threshold, return top entity anyway
        if not results and len(top_indices) > 0:
            top_idx = top_indices[0]
            entity = self.entity_list[top_idx]
            results.append((entity, float(similarities[top_idx])))

        return results


class EmbeddingActorCritic:
    """
    Actor-Critic RL agent that predicts continuous embeddings.

    Actor: State → SentenceBERT embedding (384-dim)
    Critic: State + Embedding → Q-value

    Uses DDPG-style training for continuous action space.
    """

    def __init__(
        self,
        state_dim: int = 384,  # SentenceBERT dimension
        embedding_dim: int = 384,  # SentenceBERT dimension (changed from 300)
        hidden_dim: int = 512,
        learning_rate: float = 0.001,
        exploration_noise: float = 0.2
    ):
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.exploration_noise = exploration_noise

        # Actor network: state → embedding
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh()  # Output in [-1, 1]
        )

        # Critic network: state + embedding → Q-value
        self.critic = nn.Sequential(
            nn.Linear(state_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate * 2)

        # Experience replay
        self.memory = []

        # Exploration
        self.noise_scale = exploration_noise
        self.noise_decay = 0.995
        self.noise_min = 0.05

    def predict_embedding(
        self,
        state: np.ndarray,
        explore: bool = True
    ) -> np.ndarray:
        """
        Predict SentenceBERT embedding from conversation state.

        Args:
            state: Conversation state embedding (384-dim)
            explore: Whether to add exploration noise

        Returns:
            Predicted embedding (384-dim)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            embedding = self.actor(state_tensor).squeeze().numpy()

        # Add exploration noise
        if explore:
            noise = np.random.normal(0, self.noise_scale, size=embedding.shape)
            embedding = embedding + noise
            embedding = np.clip(embedding, -1, 1)

        # Normalize to unit vector for cosine similarity
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding

    def store_experience(self, state, embedding, reward, next_state, done):
        """Store experience for replay."""
        self.memory.append((state, embedding, reward, next_state, done))

        if len(self.memory) > 10000:
            self.memory.pop(0)

    def train_step(self, batch_size: int = 32, gamma: float = 0.99):
        """Actor-critic training step."""
        if len(self.memory) < batch_size:
            return None

        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states, embeddings, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        embeddings = torch.FloatTensor(np.array(embeddings))
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)

        # --- Update Critic ---
        with torch.no_grad():
            next_embeddings = self.actor(next_states)
            next_q = self.critic(torch.cat([next_states, next_embeddings], dim=1)).squeeze()
            next_q[dones] = 0.0
            target_q = rewards + gamma * next_q

        current_q = self.critic(torch.cat([states, embeddings], dim=1)).squeeze()
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # --- Update Actor ---
        pred_embeddings = self.actor(states)
        actor_q = self.critic(torch.cat([states, pred_embeddings], dim=1)).squeeze()
        actor_loss = -actor_q.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Decay exploration
        if self.noise_scale > self.noise_min:
            self.noise_scale *= self.noise_decay

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'avg_q': current_q.mean().item(),
            'noise': self.noise_scale
        }


class QuestionGenerator:
    """
    Converts selected entities/concepts into natural language questions using GPT.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.cache = {}

    def generate_question(
        self,
        entities: List[str],
        conversation_context: str,
        discovered_preferences: Dict
    ) -> str:
        """
        Generate question from RL-selected entities.

        Args:
            entities: Movie entities selected by RL (e.g., ["Inception", "Nolan", "sci-fi"])
            conversation_context: Recent conversation turns
            discovered_preferences: What we know about user

        Returns:
            Natural language question
        """
        # Build prompt
        prompt = self._build_prompt(entities, conversation_context, discovered_preferences)

        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at asking questions to discover movie preferences. Generate natural, engaging questions."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )

            question = response.choices[0].message.content.strip()
            return question

        except Exception as e:
            print(f"GPT error: {e}, using fallback")
            return self._fallback_question(entities)

    def _build_prompt(self, entities, context, preferences):
        """Build GPT prompt for question generation."""
        parts = [
            "Generate a natural question to discover movie preferences.",
            f"\nFocus on these concepts: {', '.join(entities)}",
        ]

        if context:
            parts.append(f"\nConversation so far: {context[-300:]}")

        if preferences:
            pref_str = ", ".join([f"{k}: {v}" for k, v in preferences.items()])
            parts.append(f"\nAlready discovered: {pref_str}")

        parts.append("\nGenerate a specific, engaging question that explores the given concepts.")
        parts.append("Question:")

        return "\n".join(parts)

    def _fallback_question(self, entities):
        """Simple fallback when GPT fails."""
        if not entities:
            return "What kind of movies do you enjoy?"

        entity = entities[0]
        templates = [
            f"How do you feel about {entity}?",
            f"What's your opinion on {entity}?",
            f"Are you interested in {entity}?",
        ]
        return random.choice(templates)


class EmbeddingQLEM:
    """
    Main QELM system with embedding-based RL and SentenceBERT.

    Pipeline:
    1. Encode conversation state (SentenceBERT)
    2. RL predicts SentenceBERT embedding (actor-critic)
    3. Map embedding to nearest movie entities
    4. GPT generates natural question from entities
    5. User responds, update state, train RL
    """

    def __init__(
        self,
        movielens_data_path: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo"
    ):
        # Initialize SentenceBERT embedding space
        self.embedding_space = SentenceBERTEmbeddingSpace(movielens_data_path)

        # Initialize RL agent (actor-critic)
        self.rl_agent = EmbeddingActorCritic(
            state_dim=384,  # SentenceBERT
            embedding_dim=384  # SentenceBERT (changed from 300)
        )

        # Question generator
        self.question_generator = QuestionGenerator(model_name)

        # Conversation state encoder (same as embedding space)
        self.encoder = self.embedding_space.encoder

        # Conversation state
        self.conversation_history = []
        self.discovered_preferences = {}
        self.embedding_history = []

        print(f"✓ Initialized EmbeddingQLEM with SentenceBERT embedding space")

    def encode_conversation_state(self) -> np.ndarray:
        """Encode conversation history into state vector."""
        if not self.conversation_history:
            return np.zeros(384)

        # Recent conversation + discovered preferences
        components = []
        components.append(" ".join(self.conversation_history[-6:]))

        if self.discovered_preferences:
            pref_text = " ".join([f"{k}: {v}" for k, v in self.discovered_preferences.items()])
            components.append(pref_text)

        state_text = " | ".join(components)
        return self.encoder.encode(state_text, convert_to_numpy=True)

    def select_next_question(self, explore: bool = True, verbose: bool = True) -> str:
        """
        Main QELM pipeline: select and generate next question.
        """
        # 1. Encode state
        state = self.encode_conversation_state()

        # 2. RL predicts SentenceBERT embedding
        predicted_embedding = self.rl_agent.predict_embedding(state, explore=explore)
        self.embedding_history.append(predicted_embedding)

        # 3. Find nearest movie entities
        nearest_entities = self.embedding_space.find_nearest_entities(
            predicted_embedding,
            top_k=3,
            threshold=0.3
        )

        entity_names = [entity for entity, score in nearest_entities]

        if verbose:
            print(f"\n🧠 RL predicted embedding → Nearest entities:")
            for entity, score in nearest_entities:
                print(f"   • {entity}: {score:.3f}")

        # 4. Generate question from entities
        context = " | ".join(self.conversation_history[-4:])
        question = self.question_generator.generate_question(
            entity_names,
            context,
            self.discovered_preferences
        )

        # 5. Update history
        self.conversation_history.append(f"Agent: {question}")

        return question

    def process_user_response(self, response: str):
        """Process user response and update state."""
        self.conversation_history.append(f"User: {response}")
        self._extract_preferences(response)

    def _extract_preferences(self, response: str):
        """Simple preference extraction from response."""
        response_lower = response.lower()

        # Check against known entities
        for entity in self.embedding_space.entity_list:
            if entity.lower() in response_lower:
                if any(word in response_lower for word in ['love', 'like', 'enjoy', 'great']):
                    self.discovered_preferences[entity] = 'positive'
                elif any(word in response_lower for word in ['hate', 'dislike', 'terrible']):
                    self.discovered_preferences[entity] = 'negative'

    def train_from_episode(self, reward: float):
        """Train RL from completed episode."""
        if len(self.embedding_history) < 2:
            return None

        states = []
        for i in range(len(self.embedding_history)):
            temp_history = self.conversation_history[:i*2]
            if temp_history:
                state_text = " | ".join(temp_history)
                state = self.encoder.encode(state_text, convert_to_numpy=True)
            else:
                state = np.zeros(384)
            states.append(state)

        # Store experiences
        for i in range(len(states)):
            next_state = states[i+1] if i+1 < len(states) else states[i]
            done = (i == len(states) - 1)

            self.rl_agent.store_experience(
                states[i],
                self.embedding_history[i],
                reward,
                next_state,
                done
            )

        # Train
        return self.rl_agent.train_step()

    def reset_conversation(self):
        """Reset for new episode."""
        self.conversation_history = []
        self.discovered_preferences = {}
        self.embedding_history = []


# Test/demo code
if __name__ == "__main__":
    print("="*60)
    print("EMBEDDING-BASED QELM WITH SENTENCEBERT")
    print("="*60)

    # Initialize system
    qelm = EmbeddingQLEM(movielens_data_path=None)

    # First question
    question1 = qelm.select_next_question(explore=True, verbose=True)
    print(f"\n❓ QELM: {question1}")

    # Simulate response
    response1 = "I really love Christopher Nolan films, especially Inception and Interstellar"
    qelm.process_user_response(response1)
    print(f"👤 User: {response1}")

    # Second question
    question2 = qelm.select_next_question(explore=True, verbose=True)
    print(f"\n❓ QELM: {question2}")

    # Show discovered preferences
    print(f"\n✅ Discovered preferences: {qelm.discovered_preferences}")
