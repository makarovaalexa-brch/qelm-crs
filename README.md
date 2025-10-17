# QELM-CRS

**Question Embedding Learning with RL for Conversational Recommendation**

## Overview

QELM is a conversational movie recommendation system that uses reinforcement learning to optimize question-asking strategies. The RL agent predicts continuous embeddings in semantic space that guide natural language question generation.

## Key Innovation

Instead of selecting from a fixed set of questions, QELM:
1. Predicts **continuous embeddings** (384-dim SentenceBERT space)
2. Maps embeddings to movie entities (titles, actors, directors, genres)
3. Generates natural questions via GPT based on predicted entities
4. Learns from recommendation success using actor-critic RL

## Architecture

```
User Response
    ↓
Preference Extractor (GPT) → Structured facts
    ↓
State Encoder (SentenceBERT) → 384-dim vector
    ↓
RL Actor (trained) → 384-dim embedding
    ↓
Entity Mapper → ["Nolan", "Inception", "sci-fi"]
    ↓
GPT Question Generator → "Since you enjoy Nolan, have you seen Tenet?"
    ↓
Two-Tower Recommender → Movie recommendations
```

## Core Components

- **Embedding Space**: SentenceBERT (384-dim) - handles multi-word entities
- **RL Agent**: Actor-critic (DDPG-style) for continuous action space
- **Recommender**: Two-tower architecture with BPR loss
- **Training**: 3-stage curriculum learning (supervised → quality → end-to-end)

## Quick Start

### 1. Install Dependencies
```bash
poetry install
```

### 2. Train on Google Colab (GPU)
Upload `train_colab.ipynb` to Google Colab and follow the notebook.

### 3. Local Training
```bash
# Generate sample data
poetry run python src/qelm/data/reddit_scraper.py --sample --output-dir data/reddit

# Train Stage 1 (supervised pretraining)
poetry run python src/qelm/training/stage1_supervised.py --reddit-data data/reddit --epochs 10

# Test the system
poetry run python src/qelm/models/embedding_qelm.py
```

## Training Stages

1. **Stage 1 (Supervised)**: Learn to predict embeddings in correct semantic space
2. **Stage 2 (Quality)**: Optimize for question diversity and naturalness
3. **Stage 3 (End-to-End)**: Maximize recommendation success with held-out evaluation

## Project Structure

```
qelm-crs/
├── src/qelm/
│   ├── models/
│   │   ├── embedding_qelm.py           # Main RL system
│   │   └── two_tower_recommender.py    # Recommendation model
│   ├── utils/
│   │   └── preference_extractor.py     # LLM preference extraction
│   ├── data/
│   │   └── reddit_scraper.py           # Reddit data collection
│   └── training/
│       └── stage1_supervised.py        # Stage 1 training loop
├── train_colab.ipynb                   # Colab training notebook
└── docs/                               # Additional documentation
```

## Research Contribution

First system to:
- Use continuous embeddings for conversational recommendation
- Bootstrap RL with pretrained SentenceBERT semantic knowledge
- Train on real question-asking data from Reddit
- Combine actor-critic RL + GPT generation in shared semantic space
