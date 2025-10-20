"""
Data collection modules for QELM-CRS.

Includes:
- Reddit conversation scraper
"""

from .reddit_scraper import RedditScraper, create_sample_reddit_data

__all__ = [
    "RedditScraper",
    "create_sample_reddit_data"
]
