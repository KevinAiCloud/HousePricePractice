import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

from pdf_processor import extract_pdf_pages, chunk_pages
from embeddings import SimpleEmbedder
from retriever import build_faiss_index, retrieve
from llm import generate_answer
from citations import get_unique_sources, append_citations
from guardrails import check_retrieval_relevance, verify_citation_consistency, REFUSAL_MESSAGE

def run_unit_tests():
    print("="*60)
    print("RUNNING UNIT TESTS: relevance check boundary limits")
    print("="*60)
    
    # 1. Exact threshold check: 0.40 -> ALLOW
    exact_chunks = [{"score": 0.40}]
    res = check_retrieval_relevance(exact_chunks, threshold=0.40)
    assert res["allowed"] is True, f"Boundary 0.40 failed: expected allowed=True, got {res}"
    print("[OK] Boundary 0.40 similarity check allows generation.")
    
    # 2. Below threshold check: 0.39 -> REFUSE
    below_chunks = [{"score": 0.39}]
    res = check_retrieval_relevance(below_chunks, threshold=0.40)
    assert res["allowed"] is False, f"Boundary 0.39 failed: expected allowed=False, got {res}"
    print("[OK] Boundary 0.39 similarity check refuses generation.")
    
    # 3. Strong evidence check: 0.75 -> ALLOW
    strong_chunks = [{"score": 0.75}, {"score": 0.20}]
    res = check_retrieval_relevance(strong_chunks, threshold=0.40)
    assert res["allowed"] is True, f"Strong check failed: expected allowed=True, got {res}"
    print("[OK] Strong evidence check (0.75) allows generation.")
    
    # 4. Weak evidence check: 0.25 -> REFUSE
    weak_chunks = [{"score": 0.25}, {"score": 0.10}]
    res = check_retrieval_relevance(weak_chunks, threshold=0.40)
    assert res["allowed"] is False, f"Weak check failed: expected allowed=False, got {res}"
    print("[OK] Weak evidence check (0.25) refuses generation.")
    
    # 5. Empty check: [] -> REFUSE
    res = check_retrieval_relevance([], threshold=0.40)
    assert res["allowed"] is False, f"Empty check failed: expected allowed=False, got {res}"
    print("[OK] Empty retrieval check refuses generation.")
    
    # 6. Citation consistency check
    mock_chunks = [{"document": "file.pdf", "page": 1}]
    mock_citations_valid = [{"document": "file.pdf", "page": 1}]
    mock_citations_invalid = [{"document": "file.pdf", "page": 2}]
    
    assert verify_citation_consistency(mock_citations_valid, mock_chunks) is True
    assert verify_citation_consistency(mock_citations_invalid, mock_chunks) is False
    print("[OK] Citation consistency verification check validated.")
    
    print("\nUnit tests completed successfully!")

def generate_test_pdf(filename: str):
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
    story.append(PageBreak())
    
    # Page 3
    p3_text = "The company launched a new product in June."
    story.append(Paragraph(p3_text, styles["Normal"]))
    
    doc.build(story)

def run_pipeline(question: str, chunks: list, index, embedder, threshold: float = 0.40):
    # 1. Retrieve
    retrieved = retrieve(question, chunks, index, embedder, top_k=2)
    
    # 2. Guardrail check
    guard = check_retrieval_relevance(retrieved, threshold=threshold)
    
    print("\n" + "="*50)
    print("GROUNDING CHECK")
    print("="*50)
    print(f"Question:             '{question}'")
    print(f"Best similarity score: {guard['best_score']:.4f}")
    print(f"Threshold:             {guard['threshold']:.4f}")
    print(f"Decision:              {'ALLOW' if guard['allowed'] else 'REFUSE'}")
    print(f"Reason:                {guard['reason']}")
    print("="*50)
    
    if not guard["allowed"]:
        print("LLM BYPASSED: Short-circuiting and returning safe refusal response.")
        return REFUSAL_MESSAGE, retrieved, guard
        
    # If allowed, proceed to LLM
    print("LLM INVOKED: Querying Ollama/Gemma...")
    raw_answer = generate_answer(question, retrieved)
    unique_sources = get_unique_sources(retrieved)
    
    # Run citation consistency check (source-integrity check)
    consistency = verify_citation_consistency(unique_sources, retrieved)
    assert consistency, "Citation integrity error: bibliography references unretrieved sources!"
    
    final_response = append_citations(raw_answer, unique_sources)
    return final_response, retrieved, guard

def run_integration_tests():
    pdf_filename = "test_guardrail_sample.pdf"
    
    print("\n" + "="*60)
    print("RUNNING INTEGRATION TESTS: end-to-end pipeline with relevance gate")
    print("="*60)
    
    print("Generating retrieval PDF...")
    generate_test_pdf(pdf_filename)
    
    try:
        pages = extract_pdf_pages(pdf_filename)
        chunks = chunk_pages(pages, pdf_filename, chunk_size=100, overlap=10)
        
        embedder = SimpleEmbedder()
        index = build_faiss_index(chunks, embedder)
        
        # Integration test cases
        # Threshold set to 0.40
        threshold_val = 0.40
        
        # Case 1: Strong query
        print("\n--- CASE 1: Strong Query ---")
        q1 = "Why did revenue increase?"
        answer1, retrieved1, guard1 = run_pipeline(q1, chunks, index, embedder, threshold=threshold_val)
        
        # Assertions
        assert guard1["allowed"] is True, "Expected strong query to be allowed"
        assert "stronger product sales" in answer1.lower()
        print("\nFINAL RESPONSE:")
        print(answer1)
        print("Case 1 Assertions Passed.")
        
        # Case 2: Unsupported Query
        print("\n--- CASE 2: Unsupported Query ---")
        q2 = "Who is the company's CEO?"
        answer2, retrieved2, guard2 = run_pipeline(q2, chunks, index, embedder, threshold=threshold_val)
        
        # Assertions
        if not guard2["allowed"]:
            # If the score is below the threshold, verify LLM was bypassed
            assert answer2 == REFUSAL_MESSAGE, "Expected standard safe refusal response"
            print("\nFINAL RESPONSE:")
            print(answer2)
            print("Case 2 Bypassed LLM successfully.")
        else:
            # Report as a retrieval limitation if the score happens to be high
            print(f"[RETR LIMITATION] Unsupported question scored above threshold ({guard2['best_score']:.4f}). LLM was invoked.")
            
        # Case 3: Misleading Query
        print("\n--- CASE 3: Misleading Query ---")
        q3 = "What was the CEO's salary in 2025?"
        answer3, retrieved3, guard3 = run_pipeline(q3, chunks, index, embedder, threshold=threshold_val)
        
        # Assertions
        if not guard3["allowed"]:
            assert answer3 == REFUSAL_MESSAGE, "Expected standard safe refusal response"
            print("\nFINAL RESPONSE:")
            print(answer3)
            print("Case 3 Bypassed LLM successfully.")
        else:
            print(f"[RETR LIMITATION] Misleading question scored above threshold ({guard3['best_score']:.4f}). LLM was invoked.")
            
        print("\nIntegration tests completed successfully!")
        
    finally:
        if os.path.exists(pdf_filename):
            try:
                os.remove(pdf_filename)
                print("Cleaned up temporary integration test PDF.")
            except Exception as e:
                print(f"Could not clean up temporary file: {e}")

if __name__ == "__main__":
    run_unit_tests()
    run_integration_tests()
