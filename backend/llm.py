import requests
from typing import List, Dict, Any

def generate_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    model_name: str = "gemma2:2b"
) -> str:
    """
    Constructs a grounded context prompt and requests a natural-language answer
    from the local Ollama LLM service.
    
    Args:
        question: The user's query string.
        retrieved_chunks: A list of retrieved chunk dictionaries (retaining metadata).
        model_name: The name of the Ollama model to use (default is "gemma2:2b").
        
    Returns:
        The generated answer string from the local LLM.
    """
    # 1. Format the retrieved chunks into structured context blocks
    context_blocks = []
    for chunk in retrieved_chunks:
        doc = chunk.get("document", "Unknown Document")
        page = chunk.get("page", "Unknown Page")
        text = chunk.get("text", "")
        # Format matching prompt metadata specifications
        block = f"[Document: {doc} | Page: {page}]\n{text}"
        context_blocks.append(block)
        
    context_str = "\n\n".join(context_blocks)
    
    # 2. Build the instruction-guided prompt (grounding instructions)
    prompt = f"""You are a document question-answering assistant.

Answer the user's question using ONLY the information provided in the context below.

Do not use outside knowledge.
Do not invent facts.
Do not guess.

If the provided context does not contain enough information to answer the question, clearly say that the information was not found in the uploaded documents.

CONTEXT:

{context_str}

QUESTION:

{question}

ANSWER:"""

    # 3. Make HTTP request to Ollama local REST API endpoint
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False  # Non-streaming for simpler parsing
    }
    
    try:
        # Increase timeout to 180s to allow for CPU inference and model loading
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        response_json = response.json()
        return response_json.get("response", "").strip()
        
    except requests.exceptions.ConnectionError:
        return (
            "Error: Could not connect to the local Ollama service. "
            "Please ensure Ollama is running (`ollama serve`)."
        )
    except requests.exceptions.Timeout:
        return "Error: The request to the local Ollama service timed out."
    except Exception as e:
        return f"Error occurred during generation: {str(e)}"
