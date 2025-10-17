"""
Reddit conversation data scraper.

Scrapes question-asking conversations from movie-related subreddits.
Collects data for Stage 1 supervised training.

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

    def scrape_movie_subreddits(
        self,
        max_posts_per_subreddit: int = 100
    ) -> Dict[str, List[Dict]]:
        """
        Scrape multiple movie-related subreddits.

        Returns:
            Dictionary mapping subreddit name to list of posts
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
            posts = self.scrape_subreddit_questions(
                subreddit,
                max_posts=max_posts_per_subreddit
            )
            all_data[subreddit] = posts

            # Save intermediate results
            self.save_data(posts, f"{subreddit}_questions.json")

            print(f"Completed r/{subreddit}: {len(posts)} posts")
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
    Create sample Reddit-style data for development.

    Use when API is unavailable or for testing.
    """
    sample_posts = [
        {
            "id": "sample1",
            "title": "Looking for movies like Inception",
            "text": "I loved Inception's mind-bending plot and great cinematography. Any recommendations for similar movies?",
            "concepts": ["Inception", "mind-bending", "cinematography", "sci-fi"],
            "subreddit": "MovieSuggestions"
        },
        {
            "id": "sample2",
            "title": "Best Tarantino films?",
            "text": "I've seen Pulp Fiction and loved it. What other Tarantino movies should I watch?",
            "concepts": ["Tarantino", "Pulp Fiction", "crime", "dialogue"],
            "subreddit": "movies"
        },
        {
            "id": "sample3",
            "title": "Dark psychological thrillers",
            "text": "I'm in the mood for something dark and psychological. Think Shutter Island or Black Swan.",
            "concepts": ["psychological", "thriller", "dark", "Shutter Island", "Black Swan"],
            "subreddit": "MovieSuggestions"
        },
        {
            "id": "sample4",
            "title": "Movies with great soundtracks",
            "text": "Looking for films where the music really enhances the experience. I loved Interstellar's score.",
            "concepts": ["soundtrack", "music", "Interstellar", "score"],
            "subreddit": "TrueFilm"
        },
        {
            "id": "sample5",
            "title": "Feel-good comedies for weekend",
            "text": "Need something light and funny after a rough week. Preferably not too stupid.",
            "concepts": ["comedy", "feel-good", "light-hearted", "funny"],
            "subreddit": "MovieSuggestions"
        }
    ]

    return sample_posts


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
        print("Creating sample Reddit data...")
        sample_data = create_sample_reddit_data()
        scraper.save_data(sample_data, "sample_questions.json")
        print(f"Created {len(sample_data)} sample posts")
    else:
        print("Scraping Reddit movie subreddits...")
        all_data = scraper.scrape_movie_subreddits(
            max_posts_per_subreddit=args.max_posts
        )

        total_posts = sum(len(posts) for posts in all_data.values())
        print(f"\nTotal posts scraped: {total_posts}")
        print("Data saved to:", scraper.output_dir)
