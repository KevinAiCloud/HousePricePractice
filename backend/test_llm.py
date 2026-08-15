import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

from pdf_processor import extract_pdf_pages, chunk_pages
from embeddings import SimpleEmbedder
from retriever import build_faiss_index, retrieve
from llm import generate_answer

def generate_llm_test_pdf(filename: str):
    """
    Generates a 3-page synthetic PDF containing target facts:
    - Page 1: Revenue growth details
    - Page 2: Expansion details (unnamed markets) and expenses
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
        "The names of the markets are not specified. "
        "Operating expenses increased by 5 percent."
    )
    story.append(Paragraph(p2_text, styles["Normal"]))
    story.append(PageBreak())
    
    # Page 3
    p3_text = "The company launched a new product in June."
    story.append(Paragraph(p3_text, styles["Normal"]))
    
    doc.build(story)

def run_llm_tests():
    pdf_filename = "test_llm_sample.pdf"
    
    print("Generating synthetic LLM test PDF...")
    generate_llm_test_pdf(pdf_filename)
    
    try:
        # Step 1: Extract pages
        print("Extracting pages...")
        pages = extract_pdf_pages(pdf_filename)
        
        # Step 2: Chunk pages (100 words per chunk to isolate pages)
        print("Chunking pages...")
        chunks = chunk_pages(pages, pdf_filename, chunk_size=100, overlap=10)
        
        # Step 3: Embed & Index (MiniLM + FAISS)
        print("Initializing Embedder...")
        embedder = SimpleEmbedder()
        print("Building FAISS index...")
        index = build_faiss_index(chunks, embedder)
        
        # Step 4: Define test queries and expected facts/assertions
        test_cases = [
            {
                "id": 1,
                "name": "Direct Factual Question",
                "query": "What was the company's revenue growth?",
                "top_k": 2,
                "verification": lambda ans: "15" in ans or "fifteen" in ans.lower(),
                "error_msg": "Expected the answer to mention the 15 percent revenue growth."
            },
            {
                "id": 2,
                "name": "Causal Question",
                "query": "Why did revenue increase?",
                "top_k": 2,
                "verification": lambda ans: any(w in ans.lower() for w in ["product", "sales", "stronger"]),
                "error_msg": "Expected the answer to attribute the increase to stronger product sales."
            },
            {
                "id": 3,
                "name": "Missing details (No hallucination)",
                "query": "What markets did the company expand into?",
                "top_k": 2,
                "verification": lambda ans: "not" in ans.lower() or "unspecified" in ans.lower() or "names" in ans.lower() or "three" in ans.lower(),
                "error_msg": "Expected the answer to specify that the names are not mentioned or that it was three markets."
            },
            {
                "id": 4,
                "name": "Completely Unsupported Question",
                "query": "Who is the company's CEO?",
                "top_k": 2,
                "verification": lambda ans: any(phrase in ans.lower() for phrase in [
                    "not found", "not mention", "no information", "insufficient", 
                    "don't have", "does not specify", "cannot answer", "unable", "not provide",
                    "not contain", "no mention"
                ]),
                "error_msg": "Expected a refusal response due to insufficient context."
            }
        ]
        
        # Step 5: Execute Queries
        print("\n" + "="*70)
        print("RUNNING GROUNDED LOCAL LLM TESTS (gemma2:2b)")
        print("="*70)
        
        failures = 0
        for tc in test_cases:
            print(f"\nTEST {tc['id']} — {tc['name']}")
            print(f"QUESTION: {tc['query']}")
            print("-" * 50)
            
            # Retrieve evidence
            retrieved = retrieve(tc['query'], chunks, index, embedder, top_k=tc['top_k'])
            
            print("RETRIEVED CONTEXT:")
            for idx, item in enumerate(retrieved, start=1):
                print(f"  [{idx}] {item['document']} | Page {item['page']} | Score: {item['score']:.4f}")
                print(f"      Text: {item['text']}")
            print()
            
            # Generate answer via Ollama
            answer = generate_answer(tc['query'], retrieved, model_name="gemma2:2b")
            
            print("LOCAL LLM ANSWER:")
            print(answer)
            print("-" * 50)
            
            # Verify response content
            passed = tc['verification'](answer)
            if passed:
                print(f"Result: [OK]")
            else:
                print(f"Result: [WARNING] {tc['error_msg']}")
                failures += 1
                
        print("\n" + "="*70)
        print("LLM TESTING COMPLETE")
        print("="*70)
        
        if failures == 0:
            print("[OK] All prompt-grounding test assertions passed successfully!")
        else:
            print(f"[WARNING] LLM tests finished with {failures} assertion warning(s). Please inspect responses manually.")
            
    finally:
        if os.path.exists(pdf_filename):
            try:
                os.remove(pdf_filename)
                print("Cleaned up temporary LLM test PDF.")
            except Exception as e:
                print(f"Could not clean up temporary file: {e}")

if __name__ == "__main__":
    run_llm_tests()
