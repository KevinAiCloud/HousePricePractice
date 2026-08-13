import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

from pdf_processor import extract_pdf_pages, chunk_pages

def generate_test_pdf(filename: str):
    """
    Generates a 3-page test PDF:
    - Page 1: contains 300 words (w1 to w300)
    - Page 2: is empty (contains only a Spacer)
    - Page 3: contains 100 words (p3w1 to p3w100)
    """
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Page 1 text
    page1_words = [f"w{i}" for i in range(1, 301)]
    story.append(Paragraph(" ".join(page1_words), styles["Normal"]))
    story.append(PageBreak())
    
    # Page 2: empty
    story.append(Spacer(1, 1))
    story.append(PageBreak())
    
    # Page 3 text
    page3_words = [f"p3w{i}" for i in range(1, 101)]
    story.append(Paragraph(" ".join(page3_words), styles["Normal"]))
    
    doc.build(story)

def run_tests():
    pdf_filename = "test_sample.pdf"
    
    print("Generating sample PDF...")
    generate_test_pdf(pdf_filename)
    
    try:
        # 1. Extraction Test
        print("Extracting pages from PDF...")
        pages = extract_pdf_pages(pdf_filename)
        
        print("\n" + "="*50)
        print(f"DOCUMENT STRUCTURE: {pdf_filename}")
        print("="*50)
        
        for p in pages:
            text_preview = p['text'].strip()
            if len(text_preview) > 150:
                text_preview = text_preview[:150] + "..."
            elif not text_preview:
                text_preview = "[EMPTY PAGE]"
            print(f"Page {p['page']}")
            print("-" * 40)
            print(text_preview)
            print()
            
        # Assertions for extraction
        assert len(pages) == 3, f"Expected 3 pages, got {len(pages)}"
        assert pages[0]['page'] == 1, "Page index mismatch"
        assert pages[1]['page'] == 2, "Page index mismatch"
        assert pages[2]['page'] == 3, "Page index mismatch"
        assert pages[1]['text'].strip() == "", "Page 2 should be empty"
        print("[OK] PDF Extraction verification passed successfully.")
        
        # 2. Chunking Test (256 words, 50 words overlap)
        print("\nChunking pages...")
        chunks = chunk_pages(pages, pdf_filename, chunk_size=256, overlap=50)
        
        print("\n" + "="*50)
        print("CHUNKS GENERATED:")
        print("="*50)
        
        for idx, ch in enumerate(chunks, start=1):
            words = ch['text'].split()
            print(f"Chunk {idx}")
            print(f"  ID:       {ch['chunk_id']}")
            print(f"  Doc:      {ch['document']}")
            print(f"  Page:     {ch['page']}")
            print(f"  Words:    {len(words)}")
            text_preview = ch['text']
            if len(text_preview) > 150:
                text_preview = text_preview[:150] + "..."
            print(f"  Content:  {text_preview}")
            print()
            
        # Assertions for chunking
        assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
        
        # Chunk 1 validation
        c1_words = chunks[0]['text'].split()
        assert len(c1_words) == 256, f"Expected Chunk 1 to have 256 words, got {len(c1_words)}"
        assert chunks[0]['page'] == 1, "Chunk 1 should be from page 1"
        assert c1_words[0] == "w1"
        assert c1_words[-1] == "w256"
        
        # Chunk 2 validation
        c2_words = chunks[1]['text'].split()
        assert chunks[1]['page'] == 1, "Chunk 2 should be from page 1"
        assert c2_words[0] == "w207", f"Expected Chunk 2 to start at w207, got {c2_words[0]}"
        assert c2_words[-1] == "w300"
        
        # Overlap validation between Chunk 1 and Chunk 2
        overlap_words = set(c1_words).intersection(set(c2_words))
        assert len(overlap_words) == 50, f"Expected overlap of 50 words, got {len(overlap_words)}"
        assert "w207" in overlap_words
        assert "w256" in overlap_words
        
        # Chunk 3 validation (from Page 3)
        c3_words = chunks[2]['text'].split()
        assert len(c3_words) == 100, f"Expected Chunk 3 to have 100 words, got {len(c3_words)}"
        assert chunks[2]['page'] == 3, "Chunk 3 should be from page 3"
        assert c3_words[0] == "p3w1"
        assert c3_words[-1] == "p3w100"
        
        print("[OK] Chunking validation passed successfully.")
        print("\nAll tests PASSED!")
        
    finally:
        if os.path.exists(pdf_filename):
            try:
                os.remove(pdf_filename)
                print("Cleaned up temporary test PDF.")
            except Exception as e:
                print(f"Could not clean up temporary file: {e}")

if __name__ == "__main__":
    run_tests()
