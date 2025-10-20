"""
Reddit conversation data scraper.

Scrapes conversation pairs from movie-related subreddits for Stage 1 training.

KEY DESIGN:
- Scrapes USER POST + RESPONSE pairs (not just posts)
- Extracts concepts from the RESPONSE (what to ask next)
- Trains model: User preference → What concepts to explore

Example:
    User: "I love Inception"
    Response: "Have you seen other Nolan films like Interstellar?"
    → Concepts: ["Nolan", "Interstellar", "director"]

Subreddits:
- r/MovieSuggestions
- r/movies
- r/TrueFilm
- r/criterion
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import requests


class RedditScraper:
    """
    Scrapes movie conversation data from Reddit using Pushshift API.

    No authentication needed - uses public Pushshift archive.
    """

    def __init__(self, output_dir: str = "data/reddit"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Pushshift API endpoint (public archive)
        self.api_base = "https://api.pullpush.io/reddit/search"

    def scrape_subreddit_questions(
        self,
        subreddit: str,
        max_posts: int = 100,
        keywords: List[str] = None
    ) -> List[Dict]:
        """
        Scrape question posts from a subreddit.

        Args:
            subreddit: Subreddit name (without r/)
            max_posts: Maximum number of posts to scrape
            keywords: Filter posts containing these keywords

        Returns:
            List of post dictionaries with questions and responses
        """
        if keywords is None:
            keywords = [
                "recommend", "suggestion", "looking for", "similar to",
                "what should", "any recommendations", "help me find"
            ]

        print(f"Scraping r/{subreddit}...")

        posts = []
        after = None

        while len(posts) < max_posts:
            # Build query
            params = {
                "subreddit": subreddit,
                "size": min(100, max_posts - len(posts)),
                "sort": "desc",
                "sort_type": "created_utc",
            }

            if after:
                params["after"] = after

            try:
                # Get submissions
                response = requests.get(
                    f"{self.api_base}/submission",
                    params=params,
                    timeout=30
                )

                if response.status_code != 200:
                    print(f"API error: {response.status_code}")
                    break

                data = response.json()

                if "data" not in data or not data["data"]:
                    print("No more posts found")
                    break

                # Filter for question posts
                for post in data["data"]:
                    # Check if post contains question keywords
                    text = (post.get("title", "") + " " + post.get("selftext", "")).lower()

                    if any(keyword in text for keyword in keywords):
                        posts.append({
                            "id": post.get("id"),
                            "title": post.get("title"),
                            "text": post.get("selftext"),
                            "author": post.get("author"),
                            "created_utc": post.get("created_utc"),
                            "score": post.get("score"),
                            "num_comments": post.get("num_comments"),
                            "url": f"https://reddit.com{post.get('permalink')}"
                        })

                        if len(posts) >= max_posts:
                            break

                # Update pagination
                if data["data"]:
                    after = data["data"][-1]["created_utc"]

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                print(f"Error scraping: {e}")
                break

        print(f"Scraped {len(posts)} question posts")
        return posts

    def get_post_comments(self, post_id: str, subreddit: str) -> List[Dict]:
        """
        Get comments for a specific post.

        Args:
            post_id: Reddit post ID
            subreddit: Subreddit name

        Returns:
            List of comment dictionaries
        """
        params = {
            "link_id": post_id,
            "subreddit": subreddit,
            "size": 50,
            "sort": "asc"
        }

        try:
            response = requests.get(
                f"{self.api_base}/comment",
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])

        except Exception as e:
            print(f"Error getting comments: {e}")

        return []

    def scrape_conversation_pairs(
        self,
        subreddit: str,
        max_pairs: int = 100
    ) -> List[Dict]:
        """
        Scrape conversation pairs: user post + helpful response.

        Args:
            subreddit: Subreddit name
            max_pairs: Maximum conversation pairs to collect

        Returns:
            List of conversation pairs with structure:
            {
                "user_post": "I love Inception",
                "response": "Have you seen Interstellar?",
                "response_concepts": ["Interstellar", "Nolan", "sci-fi"]
            }
        """
        print(f"Scraping conversation pairs from r/{subreddit}...")

        # First get posts
        posts = self.scrape_subreddit_questions(subreddit, max_posts=max_pairs * 2)

        conversation_pairs = []

        for post in posts:
            if len(conversation_pairs) >= max_pairs:
                break

            # Get comments for this post
            comments = self.get_post_comments(post["id"], subreddit)

            if not comments:
                continue

            # Find good responses (non-bot, substantive, upvoted)
            good_responses = []
            for comment in comments:
                body = comment.get("body", "")
                score = comment.get("score", 0)
                author = comment.get("author", "")

                # Filter criteria
                if (
                    len(body) > 50 and  # Substantive response
                    score >= 1 and  # At least 1 upvote
                    author != "AutoModerator" and
                    "[deleted]" not in body and
                    "[removed]" not in body
                ):
                    good_responses.append({
                        "body": body,
                        "score": score
                    })

            if not good_responses:
                continue

            # Take top-scored response
            best_response = max(good_responses, key=lambda x: x["score"])

            # Create conversation pair
            user_text = post["title"] + " " + post.get("text", "")

            conversation_pairs.append({
                "id": post["id"],
                "user_post": user_text.strip(),
                "response": best_response["body"],
                "subreddit": subreddit,
                "post_url": post.get("url", "")
            })

            # Rate limiting
            time.sleep(0.5)

        print(f"Collected {len(conversation_pairs)} conversation pairs")
        return conversation_pairs

    def scrape_movie_subreddits(
        self,
        max_posts_per_subreddit: int = 100
    ) -> Dict[str, List[Dict]]:
        """
        Scrape multiple movie-related subreddits for conversation pairs.

        Returns:
            Dictionary mapping subreddit name to list of conversation pairs
        """
        subreddits = [
            "MovieSuggestions",
            "movies",
            "TrueFilm",
            "criterion",
            "Letterboxd"
        ]

        all_data = {}

        for subreddit in subreddits:
            pairs = self.scrape_conversation_pairs(
                subreddit,
                max_pairs=max_posts_per_subreddit
            )
            all_data[subreddit] = pairs

            # Save intermediate results
            self.save_data(pairs, f"{subreddit}_conversations.json")

            print(f"Completed r/{subreddit}: {len(pairs)} conversation pairs")
            time.sleep(2)  # Be nice to the API

        return all_data

    def save_data(self, data: List[Dict], filename: str):
        """Save scraped data to JSON file."""
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved to {output_path}")

    def load_data(self, filename: str) -> List[Dict]:
        """Load previously scraped data."""
        input_path = self.output_dir / filename

        if not input_path.exists():
            print(f"File not found: {input_path}")
            return []

        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def create_sample_reddit_data():
    """
    Create sample conversation pairs for development.

    Each pair shows: User preference → Follow-up question/suggestion
    Concepts are extracted from the RESPONSE (what to ask next).
    """
    sample_conversations = [
        {
            "id": "sample1",
            "user_post": "I loved Inception's mind-bending plot and great cinematography.",
            "response": "Since you enjoyed Inception, have you seen other Christopher Nolan films like Interstellar or The Prestige? They have similar themes of complex narratives.",
            "response_concepts": ["Christopher Nolan", "Interstellar", "The Prestige", "director", "complex narratives"],
            "subreddit": "MovieSuggestions"
        },
        {
            "id": "sample2",
            "user_post": "I've seen Pulp Fiction and loved it. What other Tarantino movies should I watch?",
            "response": "If you loved Pulp Fiction, definitely check out Reservoir Dogs and Kill Bill. Both have Tarantino's signature style with great dialogue and non-linear storytelling.",
            "response_concepts": ["Reservoir Dogs", "Kill Bill", "non-linear storytelling", "dialogue", "Tarantino style"],
            "subreddit": "movies"
        },
        {
            "id": "sample3",
            "user_post": "I'm in the mood for something dark and psychological. Think Shutter Island or Black Swan.",
            "response": "Based on your taste, I'd recommend Requiem for a Dream or The Machinist. Both are intense psychological dramas with similar dark atmosphere.",
            "response_concepts": ["Requiem for a Dream", "The Machinist", "psychological drama", "dark atmosphere", "intense"],
            "subreddit": "MovieSuggestions"
        },
        {
            "id": "sample4",
            "user_post": "Looking for films where the music really enhances the experience. I loved Interstellar's score.",
            "response": "Hans Zimmer did the score for Interstellar. You might enjoy other Zimmer soundtracks like Dune, Blade Runner 2049, or The Dark Knight trilogy.",
            "response_concepts": ["Hans Zimmer", "Dune", "Blade Runner 2049", "The Dark Knight", "composer", "soundtrack"],
            "subreddit": "TrueFilm"
        },
        {
            "id": "sample5",
            "user_post": "Need something light and funny after a rough week. Preferably not too stupid.",
            "response": "For smart comedies, try The Grand Budapest Hotel or Knives Out. Both are clever, funny, and beautifully crafted films.",
            "response_concepts": ["The Grand Budapest Hotel", "Knives Out", "clever comedy", "smart humor", "well-crafted"],
            "subreddit": "MovieSuggestions"
        }
    ]

    return sample_conversations


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Reddit movie conversations")
    parser.add_argument(
        "--max-posts",
        type=int,
        default=100,
        help="Max posts per subreddit"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Create sample data instead of scraping"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/reddit",
        help="Output directory"
    )

    args = parser.parse_args()

    scraper = RedditScraper(output_dir=args.output_dir)

    if args.sample:
        print("Creating sample conversation pairs...")
        sample_data = create_sample_reddit_data()
        scraper.save_data(sample_data, "sample_conversations.json")
        print(f"Created {len(sample_data)} sample conversation pairs")
    else:
        print("Scraping Reddit movie subreddits...")
        all_data = scraper.scrape_movie_subreddits(
            max_posts_per_subreddit=args.max_posts
        )

        total_posts = sum(len(posts) for posts in all_data.values())
        print(f"\nTotal posts scraped: {total_posts}")
        print("Data saved to:", scraper.output_dir)
