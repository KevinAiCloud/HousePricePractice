import faiss
import numpy as np
from typing import List, Dict, Any
from embeddings import SimpleEmbedder

def build_faiss_index(chunks: List[Dict[str, Any]], embedder: SimpleEmbedder) -> faiss.IndexFlatIP:
    """
    Creates a simple FAISS Flat Inner-Product index from a list of chunks.
    
    Args:
        chunks: A list of text chunk dictionaries (produced by pdf_processor).
        embedder: An instance of SimpleEmbedder.
        
    Returns:
        A faiss.IndexFlatIP containing the embeddings of the text chunks.
    """
    if not chunks:
        raise ValueError("Cannot build index for an empty list of chunks.")
        
    # Extract the text from each chunk
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate normalized embeddings (shape: [num_chunks, 384])
    embeddings = embedder.embed_texts(texts)
    
    # Create a standard FAISS Inner Product index (ideal for normalized cosine similarity search)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # Add the embeddings to the FAISS index
    index.add(embeddings)
    
    return index

def retrieve(
    question: str,
    chunks: List[Dict[str, Any]],
    index: faiss.IndexFlatIP,
    embedder: SimpleEmbedder,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Performs semantic retrieval by embedding the user's question, searching the
    FAISS index, and mapping the matching vectors back to the original chunks.
    
    Args:
        question: The user's query string.
        chunks: The original list of chunks (must be in the exact order they were indexed).
        index: The FAISS index containing the chunk embeddings.
        embedder: The SimpleEmbedder wrapper.
        top_k: The number of top relevant chunks to retrieve (default is 5).
        
    Returns:
        A list of matching chunk dictionaries, with an added "score" field.
    """
    # 1. Embed the query question (shape: [1, 384])
    query_vector = embedder.embed_texts([question])
    
    # 2. Query the FAISS index
    # search() returns (distances/scores, indices)
    scores, indices = index.search(query_vector, top_k)
    
    # 3. Map retrieved index numbers back to the original chunks list
    results = []
    for score, idx in zip(scores[0], indices[0]):
        # FAISS returns -1 as filler if fewer vectors exist than top_k
        if idx == -1 or idx >= len(chunks):
            continue
            
        # Copy the original chunk dictionary to avoid modifying the input list
        retrieved_chunk = chunks[idx].copy()
        retrieved_chunk["score"] = float(score)  # Convert float32 to standard float
        
        results.append(retrieved_chunk)
        
    return results
