import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union

class SimpleEmbedder:
    """
    A simple, beginner-friendly wrapper around the all-MiniLM-L6-v2 model.
    It generates 384-dimensional dense vectors for semantic text search.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load the sentence transformer model
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
    def embed_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generates L2-normalized embeddings for the input text(s).
        
        Args:
            texts: A single string or a list of strings to embed.
            
        Returns:
            A numpy array of shape (num_texts, 384) containing the embeddings,
            where each embedding is L2-normalized (unit length).
        """
        # Ensure we always deal with a list for model encoding consistency
        if isinstance(texts, str):
            texts = [texts]
            
        # encode returns float32 numpy array by default
        # normalize_embeddings=True ensures that vectors have unit length (magnitude of 1),
        # which means inner product search (IP) in FAISS will return cosine similarity.
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        return np.asarray(embeddings, dtype="float32")
