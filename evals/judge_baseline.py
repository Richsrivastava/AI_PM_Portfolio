import re, json
from typing import List

def contains_keyphrases(text: str, phrases: List[str]) -> float:
    text_low = text.lower()
    hits = sum(1 for p in phrases if p.lower() in text_low)
    return hits / max(1, len(phrases))
