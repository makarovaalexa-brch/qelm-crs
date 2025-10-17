"""
LLM-based preference extractor.

Extracts structured user preferences from conversational text.
Uses GPT to parse natural language responses into clean preference facts.
"""

import openai
from typing import Dict, List
import json


class PreferenceExtractor:
    """
    Extracts structured preferences from conversation using LLM.

    Converts messy conversation:
    "I love Nolan films and action, but hate romance"

    Into clean structured data:
    {"loved": ["Nolan", "action"], "hated": ["romance"]}
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name

    def extract_from_conversation(
        self,
        conversation: List[str],
        existing_preferences: Dict = None
    ) -> Dict[str, List[str]]:
        """
        Extract preferences from conversation history.

        Args:
            conversation: List of conversation turns
            existing_preferences: Previously extracted preferences to update

        Returns:
            Dict with keys: loved, liked, neutral, disliked, hated
        """
        # Build prompt
        prompt = self._build_extraction_prompt(conversation, existing_preferences)

        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at extracting user preferences from conversations about movies.
Extract entities (movies, directors, actors, genres, themes) and classify sentiment.
Return valid JSON only."""
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3  # Lower temperature for structured output
            )

            # Parse JSON response
            content = response.choices[0].message.content.strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            preferences = json.loads(content)

            # Ensure all keys exist
            for key in ["loved", "liked", "neutral", "disliked", "hated"]:
                if key not in preferences:
                    preferences[key] = []

            return preferences

        except Exception as e:
            print(f"Preference extraction error: {e}")
            # Fallback to simple keyword extraction
            return self._simple_keyword_extraction(conversation, existing_preferences)

    def _build_extraction_prompt(
        self,
        conversation: List[str],
        existing_preferences: Dict
    ) -> str:
        """Build prompt for GPT extraction."""

        parts = [
            "Extract user movie preferences from this conversation.",
            "",
            "Conversation:",
        ]

        # Add recent conversation (last 10 turns)
        recent_conv = conversation[-10:] if len(conversation) > 10 else conversation
        for turn in recent_conv:
            parts.append(f"  {turn}")

        parts.extend([
            "",
            "Extract movie-related entities (movies, directors, actors, genres, themes) and classify by sentiment.",
            "",
            "Return JSON in this exact format:",
            "{",
            '  "loved": ["entity1", "entity2"],   // User explicitly loves',
            '  "liked": ["entity3"],              // User likes or enjoys',
            '  "neutral": ["entity4"],            // User mentioned neutrally',
            '  "disliked": ["entity5"],           // User doesn\'t like',
            '  "hated": ["entity6"]               // User explicitly hates',
            "}",
            "",
            "Rules:",
            "- Only include entities explicitly mentioned by the user",
            "- Use exact names (e.g., 'Nolan' not 'Christopher Nolan')",
            "- Be conservative - only extract clear preferences",
            "- Return valid JSON only, no explanations",
        ])

        if existing_preferences:
            parts.extend([
                "",
                f"Previously extracted: {json.dumps(existing_preferences)}",
                "Update or add to these preferences based on new information."
            ])

        return "\n".join(parts)

    def _simple_keyword_extraction(
        self,
        conversation: List[str],
        existing_preferences: Dict
    ) -> Dict[str, List[str]]:
        """
        Simple fallback: keyword-based extraction.

        Used when LLM fails or is unavailable.
        """
        if existing_preferences:
            preferences = existing_preferences.copy()
        else:
            preferences = {
                "loved": [],
                "liked": [],
                "neutral": [],
                "disliked": [],
                "hated": []
            }

        # Combine conversation text
        text = " ".join(conversation).lower()

        # Simple sentiment keywords
        love_keywords = ["love", "adore", "favorite", "amazing", "incredible"]
        like_keywords = ["like", "enjoy", "good", "nice", "appreciate"]
        dislike_keywords = ["don't like", "dislike", "not a fan", "meh"]
        hate_keywords = ["hate", "terrible", "awful", "worst", "can't stand"]

        # Extract entities (very basic - just looks for capitalized words)
        words = " ".join(conversation).split()

        for i, word in enumerate(words):
            # Check if word is capitalized (potential entity)
            if word[0].isupper() and len(word) > 2:
                entity = word.strip('.,!?')

                # Check surrounding context for sentiment
                context_window = " ".join(words[max(0, i-5):min(len(words), i+5)]).lower()

                if any(kw in context_window for kw in love_keywords):
                    if entity not in preferences["loved"]:
                        preferences["loved"].append(entity)
                elif any(kw in context_window for kw in hate_keywords):
                    if entity not in preferences["hated"]:
                        preferences["hated"].append(entity)
                elif any(kw in context_window for kw in like_keywords):
                    if entity not in preferences["liked"]:
                        preferences["liked"].append(entity)
                elif any(kw in context_window for kw in dislike_keywords):
                    if entity not in preferences["disliked"]:
                        preferences["disliked"].append(entity)

        return preferences

    def preferences_to_text(self, preferences: Dict[str, List[str]]) -> str:
        """
        Convert structured preferences to text for state encoding.

        Output format optimized for SentenceBERT encoding.
        """
        parts = []

        if preferences.get("loved"):
            parts.append(f"loves: {', '.join(preferences['loved'])}")
        if preferences.get("liked"):
            parts.append(f"likes: {', '.join(preferences['liked'])}")
        if preferences.get("disliked"):
            parts.append(f"dislikes: {', '.join(preferences['disliked'])}")
        if preferences.get("hated"):
            parts.append(f"hates: {', '.join(preferences['hated'])}")

        return " | ".join(parts) if parts else ""


# Test code
if __name__ == "__main__":
    extractor = PreferenceExtractor()

    # Test conversation
    conversation = [
        "Agent: What movies do you enjoy?",
        "User: I love Christopher Nolan films, especially Inception and Interstellar",
        "Agent: What about other genres?",
        "User: I like action movies but hate romantic comedies"
    ]

    # Extract preferences
    preferences = extractor.extract_from_conversation(conversation)

    print("Extracted preferences:")
    print(json.dumps(preferences, indent=2))

    # Convert to text for encoding
    pref_text = extractor.preferences_to_text(preferences)
    print(f"\nPreference text for encoding:\n{pref_text}")
