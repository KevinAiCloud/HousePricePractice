import os
import time
from typing import List, Dict, Any, Tuple
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image, ImageDraw, ImageFont

# Import our modular pipeline components
from pdf_processor import extract_pdf_pages, chunk_pages
from visual_processor import SimpleVisualCaptioner, extract_pdf_images, create_visual_chunks
from embeddings import SimpleEmbedder
from retriever import build_faiss_index, retrieve
from llm import generate_answer
from citations import get_unique_sources, append_citations
from guardrails import check_retrieval_relevance, REFUSAL_MESSAGE
from cache import (
    normalize_question,
    make_cache_key,
    get_cached_answer,
    store_answer,
    clear_cache,
    get_cache_size
)

# Execution tracker to explicitly verify that downstream components are bypassed on cache hits
class PipelineTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.retrieval_calls = 0
        self.guardrail_calls = 0
        self.llm_calls = 0
        self.citation_calls = 0

tracker = PipelineTracker()

def answer_query_with_cache(
    question: str,
    document_names: List[str],
    all_chunks: List[Dict[str, Any]],
    index: Any,
    embedder: SimpleEmbedder,
    guardrail_threshold: float = 0.40,
    top_k: int = 5,
    tracker_obj: PipelineTracker = None
) -> Tuple[str, str, float]:
    """
    End-to-end question answering pipeline wrapped with query caching.
    
    Returns:
        Tuple of (final_response, cache_status, elapsed_time)
        where cache_status is "HIT" or "MISS".
    """
    start_time = time.time()
    
    # 1. Create cache key
    cache_key = make_cache_key(document_names, question)
    doc_set_str = "|".join(sorted(list(set(document_names))))
    
    # 2. Cache Lookup
    cached_response = get_cached_answer(cache_key)
    
    print("\n==================================================")
    print("CACHE CHECK")
    print("==================================================")
    print(f"Question:     '{question}'")
    print(f"Document Set: {doc_set_str}")
    
    if cached_response is not None:
        elapsed = time.time() - start_time
        print("Status:       HIT")
        print("Returning cached response.")
        print("Retrieval skipped.")
        print("LLM skipped.")
        print("Citation generation skipped.")
        return cached_response, "HIT", elapsed
        
    print("Status:       MISS")
    print("Executing complete RAG pipeline...")
    
    # 3. Cache Miss: Execute retrieval
    if tracker_obj:
        tracker_obj.retrieval_calls += 1
    retrieved = retrieve(question, all_chunks, index, embedder, top_k=top_k)
    
    # 4. Guardrail Relevance Check
    if tracker_obj:
        tracker_obj.guardrail_calls += 1
    relevance_decision = check_retrieval_relevance(retrieved, threshold=guardrail_threshold)
    
    if not relevance_decision["allowed"]:
        final_response = REFUSAL_MESSAGE
    else:
        # 5. Local LLM Grounded Answer Generation
        if tracker_obj:
            tracker_obj.llm_calls += 1
        raw_answer = generate_answer(question, retrieved)
        
        # 6. Programmatic Source Attribution
        if tracker_obj:
            tracker_obj.citation_calls += 1
        unique_sources = get_unique_sources(retrieved)
        final_response = append_citations(raw_answer, unique_sources)
        
    # 7. Store final response in cache
    store_answer(cache_key, final_response)
    
    elapsed = time.time() - start_time
    return final_response, "MISS", elapsed

def create_test_pdf(filename: str, pages_content: List[str]):
    c = canvas.Canvas(filename, pagesize=letter)
    for text in pages_content:
        c.drawString(100, 700, text)
        c.showPage()
    c.save()

def create_visual_chart_pdf(filename: str):
    img_path = "temp_cache_chart.png"
    img = Image.new('RGB', (400, 200), color='white')
    d = ImageDraw.Draw(img)
    d.text((20, 20), "Revenue Growth Chart\n2023: 10%  2024: 12%  2025: 15%", fill='black')
    d.rectangle([(50, 100), (90, 180)], fill='blue')
    d.rectangle([(120, 80), (160, 180)], fill='blue')
    d.rectangle([(190, 50), (230, 180)], fill='blue')
    img.save(img_path)
    
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawImage(img_path, 100, 500, width=300, height=150)
    c.showPage()
    c.save()
    
    if os.path.exists(img_path):
        os.remove(img_path)

def run_cache_tests():
    print("============================================================")
    print("RUNNING PART 7 CACHE UNIT & INTEGRATION TESTS")
    print("============================================================")
    
    # ------------------------------------------------------------
    # TEST 1: Question Normalization
    # ------------------------------------------------------------
    print("\n--- TEST 1: Question Normalization ---")
    q1 = "What was the revenue?"
    q2 = "   WHAT WAS THE REVENUE?   "
    q3 = "what   was   the   revenue?"
    norm1 = normalize_question(q1)
    norm2 = normalize_question(q2)
    norm3 = normalize_question(q3)
    
    assert norm1 == "what was the revenue?", f"Expected 'what was the revenue?', got '{norm1}'"
    assert norm1 == norm2 == norm3, f"Normalization mismatch: '{norm1}' vs '{norm2}' vs '{norm3}'"
    print(f"Raw: '{q1}' -> Normalized: '{norm1}'")
    print(f"Raw: '{q2}' -> Normalized: '{norm2}'")
    print(f"Raw: '{q3}' -> Normalized: '{norm3}'")
    print("[OK] Test 1: Question normalization passed.")
    
    # Setup test PDFs
    pdf_2025 = "Annual_Report_2025.pdf"
    pdf_2026 = "Annual_Report_2026.pdf"
    pdf_visual = "Visual_Report_2025.pdf"
    
    create_test_pdf(pdf_2025, [
        "The company's revenue increased by 15 percent in 2025. The increase was primarily caused by stronger product sales.",
        "Operating expenses increased by 5 percent in 2025."
    ])
    
    create_test_pdf(pdf_2026, [
        "The company's revenue grew by 30 percent in 2026 due to international expansion.",
        "Operating expenses remained flat in 2026."
    ])
    
    create_visual_chart_pdf(pdf_visual)
    
    try:
        # Ingest and index Document Set A (2025)
        pages_2025 = extract_pdf_pages(pdf_2025)
        chunks_2025 = chunk_pages(pages_2025, pdf_2025)
        
        embedder = SimpleEmbedder()
        index_2025 = build_faiss_index(chunks_2025, embedder)
        
        # Ingest and index Document Set B (2026)
        pages_2026 = extract_pdf_pages(pdf_2026)
        chunks_2026 = chunk_pages(pages_2026, pdf_2026)
        index_2026 = build_faiss_index(chunks_2026, embedder)
        
        # Ingest and index Visual Report
        captioner = SimpleVisualCaptioner()
        extracted_images = extract_pdf_images(pdf_visual)
        visual_chunks = create_visual_chunks(extracted_images, pdf_visual, captioner)
        index_visual = build_faiss_index(visual_chunks, embedder)
        
        # ------------------------------------------------------------
        # TEST 2: Empty Cache Produces MISS
        # ------------------------------------------------------------
        print("\n--- TEST 2: Empty Cache Produces MISS ---")
        clear_cache()
        tracker.reset()
        assert get_cache_size() == 0, "Cache should be empty"
        
        q_rev = "What was the revenue growth?"
        ans1, status1, t1 = answer_query_with_cache(
            q_rev, [pdf_2025], chunks_2025, index_2025, embedder, tracker_obj=tracker
        )
        assert status1 == "MISS", f"Expected MISS, got {status1}"
        assert tracker.retrieval_calls == 1, "Retrieval should have run"
        assert tracker.llm_calls == 1, "LLM should have run"
        assert tracker.citation_calls == 1, "Citation generation should have run"
        assert get_cache_size() == 1, "Cache should have 1 entry"
        print(f"Final Answer:\n{ans1}")
        print(f"[OK] Test 2: Empty cache produced MISS and cached response.")
        
        # ------------------------------------------------------------
        # TEST 3: Identical Question Produces HIT
        # ------------------------------------------------------------
        print("\n--- TEST 3: Identical Question Produces HIT ---")
        tracker.reset()
        ans2, status2, t2 = answer_query_with_cache(
            q_rev, [pdf_2025], chunks_2025, index_2025, embedder, tracker_obj=tracker
        )
        assert status2 == "HIT", f"Expected HIT, got {status2}"
        assert tracker.retrieval_calls == 0, "Retrieval should NOT run on HIT"
        assert tracker.llm_calls == 0, "LLM should NOT run on HIT"
        assert tracker.citation_calls == 0, "Citations should NOT run on HIT"
        assert ans2 == ans1, "Cached answer must be identical to original answer"
        print(f"[OK] Test 3: Identical question produced HIT and completely bypassed downstream pipeline.")
        
        # ------------------------------------------------------------
        # TEST 4: Different Question Produces MISS
        # ------------------------------------------------------------
        print("\n--- TEST 4: Different Question Produces MISS (No semantic cache) ---")
        tracker.reset()
        q_causal = "Why did revenue increase?"
        ans3, status3, t3 = answer_query_with_cache(
            q_causal, [pdf_2025], chunks_2025, index_2025, embedder, tracker_obj=tracker
        )
        assert status3 == "MISS", f"Expected MISS for different question, got {status3}"
        assert tracker.llm_calls == 1, "LLM should run on miss"
        print(f"[OK] Test 4: Different question produced MISS.")
        
        # ------------------------------------------------------------
        # TEST 5: Normalization Produces HIT
        # ------------------------------------------------------------
        print("\n--- TEST 5: Normalization Produces HIT ---")
        tracker.reset()
        q_norm_test = "   WHAT WAS THE REVENUE GROWTH?   "
        ans5, status5, t5 = answer_query_with_cache(
            q_norm_test, [pdf_2025], chunks_2025, index_2025, embedder, tracker_obj=tracker
        )
        assert status5 == "HIT", f"Expected HIT for normalized question, got {status5}"
        assert tracker.llm_calls == 0, "LLM should NOT run on HIT"
        assert ans5 == ans1, "Answer should match cached entry"
        print(f"[OK] Test 5: Normalized variant produced HIT.")
        
        # ------------------------------------------------------------
        # TEST 6: Different Document Set Produces MISS
        # ------------------------------------------------------------
        print("\n--- TEST 6: Different Document Set Produces MISS ---")
        tracker.reset()
        # Same question "What was the revenue growth?", but with 2026 document
        ans6, status6, t6 = answer_query_with_cache(
            q_rev, [pdf_2026], chunks_2026, index_2026, embedder, tracker_obj=tracker
        )
        assert status6 == "MISS", f"Expected MISS for new document set, got {status6}"
        assert tracker.llm_calls == 1, "LLM should run for new document context"
        assert ans6 != ans1, "2026 answer should differ from 2025 answer"
        print(f"2026 Answer:\n{ans6}")
        print(f"[OK] Test 6: Different document context prevented stale answer and produced MISS.")
        
        # ------------------------------------------------------------
        # TEST 7: Same Document Set Produces HIT
        # ------------------------------------------------------------
        print("\n--- TEST 7: Same Document Set Produces HIT ---")
        tracker.reset()
        ans7, status7, t7 = answer_query_with_cache(
            " what was the revenue growth? ", [pdf_2026], chunks_2026, index_2026, embedder, tracker_obj=tracker
        )
        assert status7 == "HIT", f"Expected HIT for 2026 document cache, got {status7}"
        assert tracker.llm_calls == 0, "LLM should NOT run on HIT"
        assert ans7 == ans6, "Answer should match 2026 cached answer"
        print(f"[OK] Test 7: Repeated query on 2026 document produced HIT.")
        
        # ------------------------------------------------------------
        # TEST 8: Refusal Can Be Cached
        # ------------------------------------------------------------
        print("\n--- TEST 8: Refusal Can Be Cached ---")
        tracker.reset()
        q_unsupported = "Who is the company's CEO?"
        refusal1, status8_1, _ = answer_query_with_cache(
            q_unsupported, [pdf_2025], chunks_2025, index_2025, embedder, tracker_obj=tracker
        )
        assert status8_1 == "MISS", "First query should be MISS"
        assert refusal1 == REFUSAL_MESSAGE, "Expected guardrail refusal"
        assert tracker.llm_calls == 0, "LLM should be bypassed on guardrail refusal"
        
        # Repeat unsupported question
        tracker.reset()
        refusal2, status8_2, _ = answer_query_with_cache(
            q_unsupported, [pdf_2025], chunks_2025, index_2025, embedder, tracker_obj=tracker
        )
        assert status8_2 == "HIT", f"Expected HIT on cached refusal, got {status8_2}"
        assert tracker.retrieval_calls == 0, "Retrieval bypassed on cached refusal"
        assert tracker.llm_calls == 0, "LLM bypassed on cached refusal"
        assert refusal2 == REFUSAL_MESSAGE, "Cached refusal returned successfully"
        print(f"[OK] Test 8: Safe refusal cached and served on repeated query.")
        
        # ------------------------------------------------------------
        # TEST 9: Visual Question Can Be Cached
        # ------------------------------------------------------------
        print("\n--- TEST 9: Visual Question Can Be Cached ---")
        tracker.reset()
        q_visual = "What does the revenue growth chart show?"
        v_ans1, v_status1, _ = answer_query_with_cache(
            q_visual, [pdf_visual], visual_chunks, index_visual, embedder, tracker_obj=tracker
        )
        assert v_status1 == "MISS", "First visual query should be MISS"
        assert tracker.llm_calls == 1, "LLM should run on first visual query"
        
        # Repeat visual question
        tracker.reset()
        v_ans2, v_status2, _ = answer_query_with_cache(
            q_visual, [pdf_visual], visual_chunks, index_visual, embedder, tracker_obj=tracker
        )
        assert v_status2 == "HIT", "Second visual query should be HIT"
        assert tracker.llm_calls == 0, "LLM should NOT run on visual HIT"
        assert v_ans2 == v_ans1, "Visual response and citations must match"
        print(f"Visual Answer:\n{v_ans2}")
        print(f"[OK] Test 9: Visual question cached with full citations.")
        
        # ------------------------------------------------------------
        # TEST 10: Clear Cache
        # ------------------------------------------------------------
        print("\n--- TEST 10: Clear Cache ---")
        clear_cache()
        assert get_cache_size() == 0, "Cache size should be 0 after clear_cache()"
        
        tracker.reset()
        ans10, status10, _ = answer_query_with_cache(
            q_rev, [pdf_2025], chunks_2025, index_2025, embedder, tracker_obj=tracker
        )
        assert status10 == "MISS", f"Expected MISS after clear_cache(), got {status10}"
        assert tracker.llm_calls == 1, "Pipeline should re-execute after cache clear"
        print(f"[OK] Test 10: clear_cache() successfully reset the cache.")
        
        # ------------------------------------------------------------
        # Performance / Timing Demonstration
        # ------------------------------------------------------------
        print("\n==================================================")
        print("CACHE PERFORMANCE DEMONSTRATION")
        print("==================================================")
        print(f"First request (MISS):  {t1:.4f} seconds")
        print(f"Second request (HIT):  {t2:.4f} seconds")
        if t2 > 0:
            speedup = t1 / max(t2, 0.0001)
            print(f"Speedup:               ~{speedup:.1f}x faster")
        print("==================================================")
        
        print("\n[ALL TESTS PASSED] Part 7 Query Caching verified successfully!")
        
    finally:
        # Cleanup test files
        for f in [pdf_2025, pdf_2026, pdf_visual]:
            if os.path.exists(f):
                os.remove(f)
        print("Cleaned up temporary test PDFs.")

if __name__ == "__main__":
    run_cache_tests()
