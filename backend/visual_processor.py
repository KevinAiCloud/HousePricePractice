import os
import io
import PyPDF2
from PIL import Image
import pytesseract
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import List, Dict, Any

# Configure pytesseract to use the local Windows executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class SimpleVisualCaptioner:
    """
    A simple, beginner-friendly wrapper around Salesforce/blip-image-captioning-base.
    Loads the captioning model once and provides reusable caption generation.
    """
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading BLIP model on device: {self.device}...")
        
        # Try loading from local directory first (avoids slow HF downloads)
        local_model_dir = os.path.join(os.path.dirname(__file__), "models", "blip-image-captioning-base")
        if os.path.exists(os.path.join(local_model_dir, "pytorch_model.bin")):
            print(f"Loading BLIP from local directory: {local_model_dir}")
            self.processor = BlipProcessor.from_pretrained(local_model_dir)
            self.model = BlipForConditionalGeneration.from_pretrained(local_model_dir).to(self.device)
        else:
            print(f"Loading BLIP from HuggingFace: {model_name}")
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        print("BLIP model loaded.")

    def generate_description(self, image: Image.Image) -> str:
        """
        Generates a text description for a PIL Image.
        """
        try:
            # Ensure the image is in RGB format for the processor
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_length=50)
                
            caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
            return caption.strip()
        except Exception as e:
            print(f"[WARNING] BLIP generation failed: {e}")
            return ""

def extract_pdf_images(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts embedded images from PDF pages using PyMuPDF (fitz).
    
    PyMuPDF reliably handles all PDF image encodings including those
    created by reportlab, scanned PDFs, and complex XObject streams.
    
    Args:
        file_path: Absolute path to the PDF document.
        
    Returns:
        A list of dictionaries representing extracted page images:
        [
            {"page": 1, "image": PIL.Image, "image_id": "document.pdf_p1_img1"},
            ...
        ]
    """
    import fitz  # PyMuPDF
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
        
    extracted_images = []
    doc_name = os.path.basename(file_path)
    
    pdf_doc = fitz.open(file_path)
    for page_idx in range(len(pdf_doc)):
        page = pdf_doc[page_idx]
        page_num = page_idx + 1  # 1-based page numbering
        
        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list, start=1):
            xref = img_info[0]  # Cross-reference number for the image
            try:
                base_image = pdf_doc.extract_image(xref)
                image_bytes = base_image["image"]
                img = Image.open(io.BytesIO(image_bytes))
                img_copy = img.copy()
                
                image_id = f"{doc_name}_p{page_num}_img{img_idx}"
                extracted_images.append({
                    "page": page_num,
                    "image": img_copy,
                    "image_id": image_id
                })
            except Exception as e:
                print(f"[WARNING] Failed to extract image xref={xref} on page {page_num}: {e}")
                
    pdf_doc.close()
    return extracted_images

def extract_text_from_image(image: Image.Image) -> str:
    """
    Runs Tesseract OCR on a PIL Image to recover text.
    """
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"[WARNING] Tesseract OCR failed: {e}")
        return ""

def create_visual_chunks(
    extracted_images: List[Dict[str, Any]], 
    document_name: str, 
    captioner: SimpleVisualCaptioner
) -> List[Dict[str, Any]]:
    """
    Combines Tesseract OCR text and BLIP descriptions into single searchable chunks.
    
    Args:
        extracted_images: The output from extract_pdf_images.
        document_name: Name of the source PDF.
        captioner: An instance of SimpleVisualCaptioner.
        
    Returns:
        A list of visual chunk dictionaries.
    """
    visual_chunks = []
    for img_data in extracted_images:
        page_num = img_data["page"]
        image = img_data["image"]
        image_id = img_data["image_id"]
        
        # 1. OCR text extraction
        ocr_text = extract_text_from_image(image)
        
        # 2. BLIP description generation
        blip_desc = captioner.generate_description(image)
        
        # 3. Format Combined Visual Text representation
        ocr_clean = ocr_text if ocr_text else "No text detected."
        blip_clean = blip_desc if blip_desc else "No description available."
        combined_text = f"VISUAL TEXT:\n{ocr_clean}\n\nIMAGE DESCRIPTION:\n{blip_clean}"
        
        visual_chunks.append({
            "chunk_id": image_id,
            "document": document_name,
            "page": page_num,
            "text": combined_text,
            "type": "image",
            "image_id": image_id
        })
        
    return visual_chunks
