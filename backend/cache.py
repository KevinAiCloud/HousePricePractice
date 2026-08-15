import re
from typing import List, Optional, Dict, Any

# In-memory query cache dictionary: { cache_key: final_answer_string }
_query_cache: Dict[str, str] = {}

def normalize_question(question: str) -> str:
    """
    Normalizes a user question by stripping leading/trailing whitespace,
    converting to lowercase, and collapsing multiple internal whitespace characters.
    
    Args:
        question: The raw user question string.
        
    Returns:
        The normalized question string.
    """
    if not question:
        return ""
    # Strip leading/trailing whitespace, convert to lowercase
    normalized = question.strip().lower()
    # Normalize multiple spaces/tabs/newlines into a single space
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized

def make_cache_key(document_names: List[str], question: str) -> str:
    """
    Constructs a deterministic cache key using sorted document names and the normalized question.
    
    Format:
        Doc1.pdf|Doc2.pdf::normalized_question
        
    Args:
        document_names: A list of active document filenames (e.g. ['Annual_Report.pdf']).
        question: The user's question.
        
    Returns:
        A unique string key representing this query in the context of these documents.
    """
    normalized_q = normalize_question(question)
    # Deduplicate and sort document names so ordering does not change the key
    sorted_docs = sorted(list(set(document_names)))
    docs_prefix = "|".join(sorted_docs)
    return f"{docs_prefix}::{normalized_q}"

def get_cached_answer(cache_key: str) -> Optional[str]:
    """
    Retrieves a cached final response for a given cache key if present.
    
    Args:
        cache_key: The string cache key produced by make_cache_key.
        
    Returns:
        The cached final response string, or None if it's a cache miss.
    """
    return _query_cache.get(cache_key, None)

def store_answer(cache_key: str, answer: str) -> None:
    """
    Stores a final response string in the in-memory cache dictionary.
    
    Args:
        cache_key: The string cache key produced by make_cache_key.
        answer: The final formatted response string (including citations/refusals).
    """
    _query_cache[cache_key] = answer

def clear_cache() -> None:
    """
    Clears all entries from the in-memory query cache.
    """
    _query_cache.clear()

def get_cache_size() -> int:
    """
    Returns the current number of cached responses.
    """
    return len(_query_cache)
