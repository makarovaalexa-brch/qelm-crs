"""
Agent implementations for conversational recommendation.

Includes:
- QBot (question asking agent)
- User simulators (MovieLens-based, rule-based)
- Question generators (LLM-based)
- Multi-agent coordinators
"""

from .qbot import QBot
from .user_simulator import MovieLensLLMSimulator, FixedUserSimulator
from .question_generator import QuestionGenerator

__all__ = [
    "QBot",
    "MovieLensLLMSimulator",
    "FixedUserSimulator",
    "QuestionGenerator"
]