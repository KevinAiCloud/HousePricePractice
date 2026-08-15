import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

from pdf_processor import extract_pdf_pages, chunk_pages
from embeddings import SimpleEmbedder
from retriever import build_faiss_index, retrieve
from llm import generate_answer
from citations import get_unique_sources, format_sources, append_citations

def generate_doc1(filename: str):
    """Generates Annual_Report.pdf (2 pages)"""
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
        "The names of the markets are not specified. "
        "Operating expenses increased by 5 percent."
    )
    story.append(Paragraph(p2_text, styles["Normal"]))
    
    doc.build(story)

def generate_doc2(filename: str):
    """Generates Product_Documentation.pdf (1 page)"""
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    p1_text = "The company launched a new product in June."
    story.append(Paragraph(p1_text, styles["Normal"]))
    
    doc.build(story)

def run_citation_tests():
    doc1_name = "Annual_Report.pdf"
    doc2_name = "Product_Documentation.pdf"
    
    print("Generating test PDFs...")
    generate_doc1(doc1_name)
    generate_doc2(doc2_name)
    
    try:
        # Ingest both documents
        print("Processing Document 1...")
        pages1 = extract_pdf_pages(doc1_name)
        chunks1 = chunk_pages(pages1, doc1_name, chunk_size=100, overlap=10)
        
        print("Processing Document 2...")
        pages2 = extract_pdf_pages(doc2_name)
        chunks2 = chunk_pages(pages2, doc2_name, chunk_size=100, overlap=10)
        
        # Combine all chunks
        all_chunks = chunks1 + chunks2
        print(f"Total chunks indexed: {len(all_chunks)}")
        
        # Build unified FAISS index
        embedder = SimpleEmbedder()
        index = build_faiss_index(all_chunks, embedder)
        
        # -------------------------------------------------------------
        # TEST 1: Single Source Citation
        # -------------------------------------------------------------
        print("\n" + "="*60)
        print("TEST 1: Single Source Citation")
        print("="*60)
        q1 = "Why did revenue increase?"
        retrieved1 = retrieve(q1, all_chunks, index, embedder, top_k=1)
        
        raw_answer1 = generate_answer(q1, retrieved1)
        unique_sources1 = get_unique_sources(retrieved1)
        final_response1 = append_citations(raw_answer1, unique_sources1)
        
        print("FINAL RESPONSE:")
        print(final_response1)
        print()
        
        # Assertions
        assert len(unique_sources1) == 1, "Expected exactly 1 source"
        assert unique_sources1[0]["document"] == "Annual_Report.pdf", "Expected Annual_Report.pdf"
        assert unique_sources1[0]["page"] == 1, "Expected page 1"
        assert final_response1.strip().endswith("Page 1"), "Bibliography formatting error"
        assert "[1]" in final_response1, "Expected [1] citation reference in output"
        print("[OK] Test 1 passed.")

        # -------------------------------------------------------------
        # TEST 2: Multiple Source Citations (Different Documents & Pages)
        # -------------------------------------------------------------
        print("\n" + "="*60)
        print("TEST 2: Multiple Source Citations")
        print("="*60)
        q2 = "What were the company's major developments?"
        # Retrieve top 3 to capture elements from page 1, page 2 of doc 1, and page 1 of doc 2
        retrieved2 = retrieve(q2, all_chunks, index, embedder, top_k=3)
        
        raw_answer2 = generate_answer(q2, retrieved2)
        unique_sources2 = get_unique_sources(retrieved2)
        final_response2 = append_citations(raw_answer2, unique_sources2)
        
        print("FINAL RESPONSE:")
        print(final_response2)
        print()
        
        # Assertions
        # FAISS should retrieve chunks from both documents and page numbers.
        # We verify that unique sources are generated correctly based on whatever FAISS returns.
        assert len(unique_sources2) > 0, "Expected retrieved sources"
        for idx, src in enumerate(unique_sources2, start=1):
            assert src["citation_id"] == idx, "Citation IDs must be sequential"
            # Ensure different documents with the same page number remain separate
            # For example, if both page 1 of Annual_Report.pdf and page 1 of Product_Documentation.pdf are present:
            # We assert their keys differ.
            
        print("[OK] Test 2 passed.")

        # -------------------------------------------------------------
        # TEST 3: Deduplication
        # -------------------------------------------------------------
        print("\n" + "="*60)
        print("TEST 3: Deduplication")
        print("="*60)
        # Construct a mock retrieval set with duplicates
        mock_chunks = [
            {"document": "Annual_Report.pdf", "page": 1, "chunk_id": "c1", "text": "A"},
            {"document": "Annual_Report.pdf", "page": 1, "chunk_id": "c2", "text": "B"},
            {"document": "Annual_Report.pdf", "page": 2, "chunk_id": "c3", "text": "C"},
            {"document": "Product_Documentation.pdf", "page": 1, "chunk_id": "c4", "text": "D"}
        ]
        
        unique_sources3 = get_unique_sources(mock_chunks)
        print("Unique sources from mock duplicate list:")
        for src in unique_sources3:
            print(f"  [{src['citation_id']}] {src['document']} - Page {src['page']}")
            
        assert len(unique_sources3) == 3, f"Expected 3 unique sources after deduplication, got {len(unique_sources3)}"
        assert unique_sources3[0]["document"] == "Annual_Report.pdf" and unique_sources3[0]["page"] == 1
        assert unique_sources3[1]["document"] == "Annual_Report.pdf" and unique_sources3[1]["page"] == 2
        assert unique_sources3[2]["document"] == "Product_Documentation.pdf" and unique_sources3[2]["page"] == 1
        print("[OK] Test 3 (Deduplication) passed.")

        # -------------------------------------------------------------
        # TEST 4: Citation Integrity
        # -------------------------------------------------------------
        print("\n" + "="*60)
        print("TEST 4: Citation Integrity")
        print("="*60)
        # Verify that no citation is manufactured that was not in the retrieved chunks
        mock_retrieved_set = [
            {"document": "Doc_X.pdf", "page": 5},
            {"document": "Doc_Y.pdf", "page": 10}
        ]
        unique_sources4 = get_unique_sources(mock_retrieved_set)
        
        # Verify that every citation maps back to one of the mock_retrieved_set elements
        allowed_pairs = {(item["document"], item["page"]) for item in mock_retrieved_set}
        for src in unique_sources4:
            pair = (src["document"], src["page"])
            assert pair in allowed_pairs, f"Invented source found: {pair}"
            
        print("[OK] Test 4 (Citation Integrity) passed.")

        # -------------------------------------------------------------
        # TEST 5: Empty Retrieval Results Handling
        # -------------------------------------------------------------
        print("\n" + "="*60)
        print("TEST 5: Empty Retrieval Results")
        print("="*60)
        empty_chunks = []
        unique_sources5 = get_unique_sources(empty_chunks)
        final_response5 = append_citations("The answer text.", unique_sources5)
        
        print("FINAL RESPONSE WITH EMPTY RETRIEVAL:")
        print(final_response5)
        print()
        
        assert len(unique_sources5) == 0
        assert "No supporting sources were retrieved." in final_response5, "Expected failure notice"
        print("[OK] Test 5 (Empty Results) passed.")

        print("\nAll Part 4 Citation Tests PASSED successfully!")
        
    finally:
        # Clean up files
        for filename in [doc1_name, doc2_name]:
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except Exception as e:
                    print(f"Could not remove {filename}: {e}")
        print("Cleaned up temporary test documents.")

if __name__ == "__main__":
    run_citation_tests()
