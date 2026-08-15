import os
import numpy as np
import faiss
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

from pdf_processor import extract_pdf_pages, chunk_pages
from embeddings import SimpleEmbedder
from retriever import build_faiss_index, retrieve

def generate_retrieval_pdf(filename: str):
    """
    Generates a 3-page test PDF for retrieval verification:
    - Page 1: Revenue increase and reasons
    - Page 2: Market expansion and expenses
    - Page 3: Product launch details
    """
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Page 1
    p1_text = (
        "The company's revenue increased by 15 percent in 2025. "
        "The increase was primarily caused by stronger product sales."
    )
    story.append(Paragraph(p1_text, styles["Normal"]))
    story.append(PageBreak())
    
    # Page 2
    p2_text = (
        "The company expanded into three new markets. "
        "Operating expenses increased by 5 percent."
    )
    story.append(Paragraph(p2_text, styles["Normal"]))
    story.append(PageBreak())
    
    # Page 3
    p3_text = "The company launched a new product in June."
    story.append(Paragraph(p3_text, styles["Normal"]))
    
    doc.build(story)

def run_retrieval_tests():
    pdf_filename = "test_retrieval_sample.pdf"
    
    print("Generating retrieval test PDF...")
    generate_retrieval_pdf(pdf_filename)
    
    try:
        # Step 1: Extract PDF pages (Part 1 logic)
        print("Extracting pages...")
        pages = extract_pdf_pages(pdf_filename)
        
        # Step 2: Chunk pages (Part 1 logic)
        print("Chunking text...")
        # Small chunk size so each page becomes exactly one chunk for clear testing
        chunks = chunk_pages(pages, pdf_filename, chunk_size=100, overlap=10)
        
        print(f"Generated {len(chunks)} chunks:")
        for ch in chunks:
            print(f"  Chunk ID: {ch['chunk_id']} (Page {ch['page']}): '{ch['text']}'")
            
        assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
        
        # Step 3: Load embedder (Part 2 logic)
        print("\nInitializing SimpleEmbedder (all-MiniLM-L6-v2)...")
        embedder = SimpleEmbedder()
        
        # Step 4: Build FAISS index (Part 2 logic)
        print("Building FAISS index...")
        index = build_faiss_index(chunks, embedder)
        
        # Verify index type and count
        assert isinstance(index, faiss.IndexFlatIP), "Index should be an inner-product index."
        assert index.ntotal == 3, f"Expected 3 indexed vectors, got {index.ntotal}"
        print("FAISS index successfully built with inner-product distance metric.")
        
        # Step 5: Test Semantic Queries
        test_queries = [
            {
                "query": "What was the company's revenue growth?",
                "expected_page": 1,
                "description": "Direct match query"
            },
            {
                "query": "Why did revenue increase?",
                "expected_page": 1,
                "description": "Causal query"
            },
            {
                "query": "How much did the company earn more compared with the previous year?",
                "expected_page": 1,
                "description": "Semantic query (no exact keywords 'revenue', 'growth', 'increased')"
            },
            {
                "query": "Tell me about the new market expansions.",
                "expected_page": 2,
                "description": "Market expansion query"
            },
            {
                "query": "When did the company release a new item?",
                "expected_page": 3,
                "description": "Product launch query"
            }
        ]
        
        print("\n" + "="*60)
        print("RUNNING SEMANTIC RETRIEVAL TESTS")
        print("="*60)
        
        for tq in test_queries:
            q = tq["query"]
            print(f"\nQuery: '{q}' ({tq['description']})")
            print("-" * 50)
            
            # Retrieve top 2 chunks
            retrieved = retrieve(q, chunks, index, embedder, top_k=2)
            
            for rank, item in enumerate(retrieved, start=1):
                print(f"  Rank {rank} (Score: {item['score']:.4f}):")
                print(f"    Chunk ID: {item['chunk_id']}")
                print(f"    Page:     {item['page']}")
                print(f"    Text:     {item['text']}")
            
            # Assertions
            top_match = retrieved[0]
            assert top_match["page"] == tq["expected_page"], (
                f"Failed for query '{q}': expected top chunk from page {tq['expected_page']}, "
                f"but got page {top_match['page']}"
            )
            # Scores should be valid cosine similarities [-1.0, 1.0]
            assert -1.0 <= top_match["score"] <= 1.0, f"Invalid similarity score: {top_match['score']}"
            
        print("\n[OK] Semantic retrieval tests passed successfully!")
        
    finally:
        if os.path.exists(pdf_filename):
            try:
                os.remove(pdf_filename)
                print("Cleaned up temporary retrieval PDF.")
            except Exception as e:
                print(f"Could not clean up temporary file: {e}")

if __name__ == "__main__":
    run_retrieval_tests()
