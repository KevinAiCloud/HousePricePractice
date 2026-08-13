import os
import PyPDF2
from typing import List, Dict, Any

def extract_pdf_pages(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF file.
    
    Args:
        file_path: The path to the PDF file on disk.
        
    Returns:
        A list of dictionaries representing each page:
        [
            {"page": 1, "text": "extracted text from page 1..."},
            {"page": 2, "text": "extracted text from page 2..."}
        ]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
        
    pages = []
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for idx, page in enumerate(pdf_reader.pages, start=1):
            # Extract text from the page. If text is None (e.g. empty/scanned page), fallback to empty string.
            text = page.extract_text() or ""
            pages.append({
                "page": idx,
                "text": text
            })
            
    return pages

def chunk_pages(
    pages: List[Dict[str, Any]], 
    document_name: str, 
    chunk_size: int = 256, 
    overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Chunks the text of each page individually. Chunks do not span across page boundaries,
    which ensures each chunk maps to exactly one page.
    
    Args:
        pages: The output list from extract_pdf_pages.
        document_name: The filename of the source document (used in chunk metadata).
        chunk_size: Number of words per chunk (default is 256).
        overlap: Number of overlapping words between consecutive chunks (default is 50).
        
    Returns:
        A list of chunks, where each chunk has the structure:
        {
            "chunk_id": "document_name_p{page}_c{counter}",
            "document": "document_name",
            "page": page_number,
            "text": "chunk text contents..."
        }
    """
    chunks = []
    global_chunk_idx = 1
    
    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"].strip()
        
        # Safe handling of empty pages: they are skipped without crashing
        if not text:
            continue
            
        # Split text by whitespace into words
        words = text.split()
        if not words:
            continue
            
        i = 0
        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size  # Prevent infinite loops if overlap >= chunk_size
            
        while i < len(words):
            # Get slice of words for current chunk
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Simple readable chunk ID format
            chunk_id = f"{document_name}_p{page_num}_c{global_chunk_idx}"
            
            chunks.append({
                "chunk_id": chunk_id,
                "document": document_name,
                "page": page_num,
                "text": chunk_text
            })
            
            global_chunk_idx += 1
            
            # Break if we have reached or exceeded the end of words
            if i + chunk_size >= len(words):
                break
            i += step
            
    return chunks
