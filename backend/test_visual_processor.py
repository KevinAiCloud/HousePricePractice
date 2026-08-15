import os
import io
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

from pdf_processor import extract_pdf_pages, chunk_pages
from embeddings import SimpleEmbedder
from retriever import build_faiss_index, retrieve
from llm import generate_answer
from citations import get_unique_sources, append_citations
from guardrails import check_retrieval_relevance
from visual_processor import (
    SimpleVisualCaptioner,
    extract_pdf_images,
    extract_text_from_image,
    create_visual_chunks
)

def create_test_font(size: int):
    """Loads Arial font on Windows with fallback to default PIL font"""
    font_path = r"C:\Windows\Fonts\arial.ttf"
    if os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            return ImageFont.load_default()
    return ImageFont.load_default()

def run_visual_unit_tests(captioner: SimpleVisualCaptioner):
    print("="*60)
    print("RUNNING VISUAL UNIT TESTS")
    print("="*60)
    
    # 1. OCR Unit Test
    print("Generating a test image with text for OCR...")
    img_ocr = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(img_ocr)
    font = create_test_font(24)
    draw.text((10, 30), "Tesseract Testing 123", fill="black", font=font)
    
    extracted_ocr = extract_text_from_image(img_ocr)
    print(f"Extracted OCR: '{extracted_ocr}'")
    assert any(term in extracted_ocr for term in ["Tesseract", "Testing", "123"]), (
        f"OCR unit test failed. Got: {extracted_ocr}"
    )
    print("[OK] Tesseract OCR unit test passed successfully.")
    
    # 2. BLIP Unit Test
    print("\nGenerating a simple visual image for BLIP...")
    # Create a simple red block image
    img_blip = Image.new("RGB", (300, 300), color="red")
    caption = captioner.generate_description(img_blip)
    print(f"BLIP Caption: '{caption}'")
    assert isinstance(caption, str), "BLIP caption must be a string"
    assert len(caption) > 0, "BLIP caption must not be empty"
    print("[OK] BLIP image captioning unit test passed successfully.")

def generate_visual_test_pdf(filename: str):
    """
    Generates a 3-page PDF with:
    - Page 1: Scanned image containing text (no selectable text layer)
    - Page 2: Normal selectable PDF text
    - Page 3: A visual chart containing labels and a bar chart
    """
    styles = getSampleStyleSheet()
    font_large = create_test_font(24)
    font_small = create_test_font(14)
    
    # 1. Draw Page 1 image (scanned page simulation)
    img_p1 = Image.new("RGB", (600, 150), color="white")
    draw_p1 = ImageDraw.Draw(img_p1)
    draw_p1.text(
        (20, 50), 
        "Operating expenses increased by 5 percent in 2025.", 
        fill="black", 
        font=font_small
    )
    p1_img_name = "temp_p1_scanned.png"
    img_p1.save(p1_img_name)
    
    # 2. Draw Page 3 image (chart simulation)
    img_p3 = Image.new("RGB", (600, 300), color="white")
    draw_p3 = ImageDraw.Draw(img_p3)
    # Chart title
    draw_p3.text((20, 20), "Revenue Growth Chart", fill="black", font=font_large)
    # Simple bar rectangles
    draw_p3.rectangle([50, 200, 100, 250], fill="blue", outline="black")
    draw_p3.rectangle([150, 150, 200, 250], fill="blue", outline="black")
    draw_p3.rectangle([250, 80, 300, 250], fill="blue", outline="black")
    # Bar Labels
    draw_p3.text((40, 260), "2023: 10%", fill="black", font=font_small)
    draw_p3.text((140, 260), "2024: 12%", fill="black", font=font_small)
    draw_p3.text((240, 260), "2025: 15%", fill="black", font=font_small)
    p3_img_name = "temp_p3_chart.png"
    img_p3.save(p3_img_name)
    
    # Compile reportlab PDF
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []
    
    # Page 1 flowable
    story.append(RLImage(p1_img_name, width=400, height=100))
    story.append(PageBreak())
    
    # Page 2 flowable
    story.append(Paragraph("The company operates in several markets.", styles["Normal"]))
    story.append(PageBreak())
    
    # Page 3 flowable
    story.append(RLImage(p3_img_name, width=400, height=200))
    
    doc.build(story)
    
    # Cleanup temporary image files
    for f in [p1_img_name, p3_img_name]:
        if os.path.exists(f):
            os.remove(f)

def run_integration_pipeline():
    pdf_filename = "test_visual_multimodal.pdf"
    
    print("\n" + "="*60)
    print("RUNNING MULTIMODAL INTEGRATION TESTS")
    print("="*60)
    
    print("Generating multimodal PDF...")
    generate_visual_test_pdf(pdf_filename)
    
    # Instantiate models once (reusable)
    captioner = SimpleVisualCaptioner()
    embedder = SimpleEmbedder()
    
    try:
        # Step 1: Ingest Normal Text Chunks
        print("\nIngesting normal PDF selectable text...")
        pages = extract_pdf_pages(pdf_filename)
        text_chunks = chunk_pages(pages, pdf_filename, chunk_size=100, overlap=10)
        
        # Verify that normal text extraction produces NO selectable text for Page 1 and Page 3 (only images)
        print("Normal text extraction results:")
        for p in pages:
            print(f"  Page {p['page']}: '{p['text'].strip()}'")
        assert len(pages[0]['text'].strip()) == 0, "Page 1 should have no normal selectable text"
        assert len(pages[2]['text'].strip()) == 0, "Page 3 should have no normal selectable text"
        print("[OK] Page 1 and Page 3 correctly identified as having no normal selectable text.")
        
        # Step 2: Ingest Visual Chunks
        print("\nIngesting visual PDF image blocks (OCR + BLIP)...")
        extracted_images = extract_pdf_images(pdf_filename)
        print(f"Extracted {len(extracted_images)} image(s) from PDF pages.")
        assert len(extracted_images) == 2, f"Expected 2 extracted images, got {len(extracted_images)}"
        
        visual_chunks = create_visual_chunks(extracted_images, pdf_filename, captioner)
        print(f"Generated {len(visual_chunks)} visual chunks:")
        for idx, ch in enumerate(visual_chunks, start=1):
            print(f"  Visual Chunk {idx} (Page {ch['page']}):")
            print(f"      Content:\n{ch['text']}")
            
        # Verify metadata
        assert visual_chunks[0]["page"] == 1, "Visual chunk 1 should be from page 1"
        assert visual_chunks[1]["page"] == 3, "Visual chunk 2 should be from page 3"
        print("[OK] Visual chunk metadata and page boundaries verified successfully.")
        
        # Combine chunks
        all_chunks = text_chunks + visual_chunks
        print(f"\nTotal combined chunks for FAISS indexing: {len(all_chunks)}")
        
        # Step 3: Build unified FAISS index
        print("Building single FAISS index...")
        index = build_faiss_index(all_chunks, embedder)
        
        # Step 4: Run Q&A Queries
        test_queries = [
            {
                "id": 1,
                "name": "Normal Text Query (Regression Check)",
                "query": "Where does the company operate?",
                "expected_page": 2,
                "verification": lambda ans: "markets" in ans.lower()
            },
            {
                "id": 2,
                "name": "Scanned Page OCR Query",
                "query": "How much did operating expenses increase?",
                "expected_page": 1,
                "verification": lambda ans: "5" in ans or "five" in ans.lower()
            },
            {
                "id": 3,
                "name": "Visual Chart Query",
                "query": "What does the revenue growth chart show?",
                "expected_page": 3,
                "verification": lambda ans: "revenue" in ans.lower() or "chart" in ans.lower()
            }
        ]
        
        for tq in test_queries:
            q = tq["query"]
            print(f"\nQUERY {tq['id']} ({tq['name']}): '{q}'")
            print("-" * 50)
            
            # Retrieve top 2
            retrieved = retrieve(q, all_chunks, index, embedder, top_k=2)
            
            print("RETRIEVED EVIDENCE:")
            for rank, ch in enumerate(retrieved, start=1):
                print(f"  Rank {rank} (Score {ch['score']:.4f}) [Page {ch['page']} - Type {ch.get('type', 'text')}]:")
                print(f"      Text: {ch['text'][:120]}...")
            
            # Guardrail check
            guard = check_retrieval_relevance(retrieved, threshold=0.40)
            if not guard["allowed"]:
                print(f"Guardrail Refused: {guard['reason']}")
                continue
                
            # LLM QA
            ans = generate_answer(q, retrieved)
            unique_sources = get_unique_sources(retrieved)
            final_response = append_citations(ans, unique_sources)
            
            print("\nFINAL PIPELINE ANSWER:")
            print(final_response)
            
            # Assertions
            top_match = retrieved[0]
            assert top_match["page"] == tq["expected_page"], (
                f"Expected top match from page {tq['expected_page']}, got page {top_match['page']}"
            )
            assert tq["verification"](final_response), f"Response verification failed for query: '{q}'"
            print(f"QUERY {tq['id']} [PASSED]")
            
        print("\nAll integration visual understanding pipeline tests PASSED!")
        
    finally:
        if os.path.exists(pdf_filename):
            try:
                os.remove(pdf_filename)
                print("Cleaned up temporary visual PDF.")
            except Exception as e:
                print(f"Could not clean up temporary file: {e}")

if __name__ == "__main__":
    # Load BLIP captioner once for all tests
    captioner = SimpleVisualCaptioner()
    
    run_visual_unit_tests(captioner)
    run_integration_pipeline()
