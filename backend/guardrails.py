from typing import List, Dict, Any

# Safe refusal response to return when retrieved evidence is too weak
REFUSAL_MESSAGE = (
    "I couldn't find enough relevant information in the "
    "uploaded documents to answer that question."
)

def check_retrieval_relevance(
    retrieved_chunks: List[Dict[str, Any]], 
    threshold: float = 0.40
) -> Dict[str, Any]:
    """
    Evaluates whether the retrieved chunks contain sufficiently relevant evidence
    by testing the best retrieval similarity score against a configurable threshold.
    
    Args:
        retrieved_chunks: List of retrieved chunk dicts (produced by retriever).
        threshold: Configurable minimum required similarity score (default 0.40).
        
    Returns:
        A dictionary with structured decision metadata:
        {
            "allowed": bool,
            "best_score": float,
            "threshold": float,
            "reason": str
        }
    """
    if not retrieved_chunks:
        return {
            "allowed": False,
            "best_score": 0.0,
            "threshold": threshold,
            "reason": "No chunks retrieved."
        }
        
    # Find the highest similarity score in the retrieved set
    best_score = max(chunk.get("score", 0.0) for chunk in retrieved_chunks)
    
    # Borderline evaluation: exactly at threshold (>=) is ALLOWED, below (<) is REFUSED.
    if best_score >= threshold:
        return {
            "allowed": True,
            "best_score": best_score,
            "threshold": threshold,
            "reason": f"Relevant evidence found (best score {best_score:.4f} >= threshold {threshold:.4f})."
        }
    else:
        return {
            "allowed": False,
            "best_score": best_score,
            "threshold": threshold,
            "reason": f"No sufficiently relevant evidence found (best score {best_score:.4f} < threshold {threshold:.4f})."
        }

def verify_citation_consistency(
    unique_sources: List[Dict[str, Any]], 
    retrieved_chunks: List[Dict[str, Any]]
) -> bool:
    """
    Enforces bibliography integrity by verifying that every source listed in the
    citations bibliography corresponds to a document/page that was actually retrieved.
    
    Args:
        unique_sources: Deduplicated citations list (from citations.py).
        retrieved_chunks: Raw retrieved chunks list (from retriever.py).
        
    Returns:
        True if all cited documents and pages exist in the retrieved set, False otherwise.
    """
    # Build set of valid retrieved (document, page) keys
    retrieved_set = set()
    for chunk in retrieved_chunks:
        doc = chunk.get("document")
        page = chunk.get("page")
        if doc is not None and page is not None:
            retrieved_set.add((doc, page))
            
    # Check that every unique source is inside the retrieved set
    for src in unique_sources:
        doc = src.get("document")
        page = src.get("page")
        if (doc, page) not in retrieved_set:
            return False
            
    return True
