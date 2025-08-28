import re
from typing import List

def contains_keyphrases(text: str, phrases: List[str]) -> float:
    """
    Calculate the fraction of keyphrases found in text.
    
    Args:
        text: Text to search in
        phrases: List of keyphrases to look for
        
    Returns:
        Float between 0 and 1 representing coverage ratio
    """
    text_low = text.lower()
    hits = sum(1 for p in phrases if p.lower() in text_low)
    return hits / max(1, len(phrases))

def score_example(response_text: str, expected_keyphrases: List[str]) -> float:
    """
    Score an LLM response based on keyphrase coverage.
    
    Args:
        response_text: The LLM's response to evaluate
        expected_keyphrases: List of expected keyphrases
        
    Returns:
        Float between 0 and 1 representing quality score
    """
    if not expected_keyphrases:
        return 1.0  # If no keyphrases expected, consider it a perfect match
    
    return contains_keyphrases(response_text, expected_keyphrases)

def evaluate_holiday_response(response_text: str, task_type: str) -> dict:
    """
    Comprehensive evaluation of holiday planning responses.
    
    Args:
        response_text: The LLM response
        task_type: Type of holiday planning task
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Define task-specific keyphrases
    task_keyphrases = {
        "wizard_holiday_planning": [
            "itinerary", "day", "hotel", "restaurant", "activity", 
            "transportation", "schedule", "morning", "afternoon", "evening"
        ],
        "direct_chat_planning": [
            "destination", "travel", "accommodation", "activities", 
            "budget", "duration", "attractions", "food", "culture"
        ],
        "sample_itinerary_planning": [
            "itinerary", "schedule", "attractions", "accommodation",
            "dining", "activities", "transportation", "recommendations"
        ]
    }
    
    keyphrases = task_keyphrases.get(task_type, [])
    keyphrase_score = score_example(response_text, keyphrases)
    
    # Additional quality checks
    response_length = len(response_text.split())
    has_structure = bool(re.search(r'(Day \d+|Morning|Afternoon|Evening|\d+\.|•|-)'), response_text)
    has_specifics = bool(re.search(r'(\$\d+|€\d+|\d+:\d+|\d+ hours?|\d+ days?)'), response_text)
    
    return {
        'keyphrase_coverage': keyphrase_score,
        'response_length': response_length,
        'has_structure': has_structure,
        'has_specifics': has_specifics,
        'overall_score': (keyphrase_score + (0.1 if has_structure else 0) + (0.1 if has_specifics else 0)) / 1.2
    }