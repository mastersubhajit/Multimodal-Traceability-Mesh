import fitz  # PyMuPDF
import pdfplumber
import os
import json
from typing import List, Dict, Any, Optional
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel
import layoutparser as lp
from paddleocr import PaddleOCR
import numpy as np

class DocParser:
    """
    Stage 1: Document Ingestion & Parsing
    Hybrid layered parsing strategy for extracting structural text, layout, and vision regions.
    """
    
    def __init__(self, output_dir: str = "data/processed", use_ocr: bool = True, use_vision: bool = True):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.use_ocr = use_ocr
        self.use_vision = use_vision
        self.doc_data = {}
        
        # Initialize Layout Parser (Detectron2 based LayoutLMv3 or standard models)
        try:
            self.layout_model = lp.Detectron2LayoutModel(
                'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
            )
        except Exception as e:
            print(f"Warning: Layout parser could not be initialized: {e}")
            self.layout_model = None

        # Initialize PaddleOCR
        if self.use_ocr:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', device="cpu")
            
        # Initialize Vision Encoder (CLIP)
        if self.use_vision:
            try:
                self.vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            except Exception as e:
                print(f"Warning: Vision model could not be initialized: {e}")
                self.use_vision = False

    def extract_layout_fitz(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Layer 1: Structural text + layout extraction using PyMuPDF.
        """
        doc = fitz.open(pdf_path)
        blocks = []
        for page_num, page in enumerate(doc):
            text_blocks = page.get_text("dict")["blocks"]
            for block in text_blocks:
                if block["type"] == 0:
                    blocks.append({
                        "page": page_num,
                        "bbox": block["bbox"],    # (x0, y0, x1, y1)
                        "type": "text",
                        "content": block.get("lines", []),
                        "font": block["lines"][0]["spans"][0]["font"] if block.get("lines") and block["lines"][0].get("spans") else "",
                        "size": block["lines"][0]["spans"][0]["size"] if block.get("lines") and block["lines"][0].get("spans") else 0
                    })
                elif block["type"] == 1:
                    blocks.append({
                        "page": page_num,
                        "bbox": block["bbox"],
                        "type": "image",
                        "image_id": block.get("number", 0)
                    })
        return blocks

    def extract_layout_pdfplumber(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract detailed layout information (tables) using pdfplumber.
        """
        results = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.find_tables()
                for table in tables:
                    results.append({
                        "page": i,
                        "bbox": table.bbox,
                        "type": "table",
                        "content": table.extract()
                    })
        return results

    def get_page_image(self, file_path: str, page_num: int = 0, dpi: int = 300) -> Image.Image:
        """
        Retrieves an image for a given page of a PDF or the image itself if file_path is an image.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            return Image.open(file_path).convert("RGB")
        
        # PDF handling
        doc = fitz.open(file_path)
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        return Image.open(io.BytesIO(pix.tobytes())).convert("RGB")

    def crop_region(self, page_img: Image.Image, bbox: List[float], page_size: tuple) -> Image.Image:
        pdf_w, pdf_h = page_size
        img_w, img_h = page_img.size
        x0, y0, x1, y1 = bbox
        
        # Guard against zero page size
        if pdf_w == 0 or pdf_h == 0:
            return page_img.crop((x0, y0, x1, y1))
            
        left = (x0 / pdf_w) * img_w
        top = (y0 / pdf_h) * img_h
        right = (x1 / pdf_w) * img_w
        bottom = (y1 / pdf_h) * img_h
        return page_img.crop((left, top, right, bottom))
        
    def encode_vision_region(self, img: Image.Image) -> List[float]:
        """Layer 3: Vision encoding for non-text regions."""
        if not self.use_vision: return []
        inputs = self.vision_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            image_features = self.vision_model.get_image_features(**inputs)
        return image_features[0].tolist()

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Parsing pipeline for a single image.
        """
        filename = os.path.basename(image_path)
        image = Image.open(image_path).convert("RGB")
        
        self.doc_data = {
            "filename": filename,
            "blocks": [],
            "pages": [{
                "page_no": 0,
                "width": image.width,
                "height": image.height
            }]
        }
        
        page_img = image
        img_np = np.array(page_img)
        
        # OCR
        if self.use_ocr:
            ocr_result = self.ocr.ocr(img_np, cls=True)
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    bbox, (text, conf) = line
                    self.doc_data["blocks"].append({
                        "page": 0,
                        "bbox": [bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]],
                        "type": "text",
                        "content": [{"spans": [{"text": text, "size": 10, "font": "OCR"}]}],
                        "confidence": conf
                    })
        
        # Layout classification
        if self.layout_model:
            layout = self.layout_model.detect(img_np)
            for block in layout:
                bbox = block.block
                if block.type in ["Figure", "Image"]:
                    x0, y0, x1, y1 = bbox.x_1, bbox.y_1, bbox.x_2, bbox.y_2
                    crop_img = page_img.crop((x0, y0, x1, y1))
                    emb = self.encode_vision_region(crop_img)
                    self.doc_data["blocks"].append({
                        "page": 0,
                        "bbox": [x0, y0, x1, y1],
                        "type": "figure",
                        "embedding": emb
                    })
                    
        return self.doc_data

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Comprehensive parsing pipeline for PDF.
        """
        filename = os.path.basename(pdf_path)
        self.doc_data = {
            "filename": filename,
            "blocks": [],
            "pages": []
        }
        
        # Metadata about pages
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            self.doc_data["pages"].append({
                "page_no": i,
                "width": page.rect.width,
                "height": page.rect.height
            })
        
        fitz_blocks = self.extract_layout_fitz(pdf_path)
        plumber_blocks = self.extract_layout_pdfplumber(pdf_path)
        
        self.doc_data["blocks"].extend(fitz_blocks)
        self.doc_data["blocks"].extend(plumber_blocks)
        
        # Perform layout classification and OCR fallback per page
        for i, page_info in enumerate(self.doc_data["pages"]):
            page_blocks = [b for b in self.doc_data["blocks"] if b["page"] == i]
            # Check if page has no text blocks (needs OCR fallback)
            has_text = any(b["type"] == "text" for b in page_blocks)
            
            page_img = None
            
            if not has_text and self.use_ocr:
                print(f"Page {i} appears scanned. Applying OCR fallback...")
                page_img = self.get_page_image(pdf_path, i)
                img_np = np.array(page_img)
                ocr_result = self.ocr.ocr(img_np, cls=True)
                if ocr_result and ocr_result[0]:
                    for line in ocr_result[0]:
                        bbox, (text, conf) = line
                        # Convert bbox back to pdf coords roughly
                        img_w, img_h = page_img.size
                        pdf_w, pdf_h = page_info["width"], page_info["height"]
                        x0 = bbox[0][0] / img_w * pdf_w
                        y0 = bbox[0][1] / img_h * pdf_h
                        x1 = bbox[2][0] / img_w * pdf_w
                        y1 = bbox[2][1] / img_h * pdf_h
                        self.doc_data["blocks"].append({
                            "page": i,
                            "bbox": [x0, y0, x1, y1],
                            "type": "text",
                            "content": [{"spans": [{"text": text, "size": 10, "font": "OCR"}]}],
                            "confidence": conf
                        })
                        
            # Apply layout classification
            if self.layout_model:
                if page_img is None:
                    page_img = self.get_page_image(pdf_path, i)
                img_np = np.array(page_img)
                layout = self.layout_model.detect(img_np)
                for block in layout:
                    bbox = block.block
                    img_w, img_h = page_img.size
                    pdf_w, pdf_h = page_info["width"], page_info["height"]
                    x0 = bbox.x_1 / img_w * pdf_w
                    y0 = bbox.y_1 / img_h * pdf_h
                    x1 = bbox.x_2 / img_w * pdf_w
                    y1 = bbox.y_2 / img_h * pdf_h
                    
                    if block.type in ["Figure", "Image"]:
                        crop_img = self.crop_region(page_img, [x0, y0, x1, y1], (pdf_w, pdf_h))
                        emb = self.encode_vision_region(crop_img)
                        self.doc_data["blocks"].append({
                            "page": i,
                            "bbox": [x0, y0, x1, y1],
                            "type": "figure",
                            "embedding": emb
                        })
            
            # Process existing image blocks for vision embeddings
            if self.use_vision:
                if page_img is None:
                    page_img = self.get_page_image(pdf_path, i)
                for block in page_blocks:
                    if block["type"] == "image" and "embedding" not in block:
                        crop_img = self.crop_region(page_img, block["bbox"], (page_info["width"], page_info["height"]))
                        block["embedding"] = self.encode_vision_region(crop_img)

        return self.doc_data

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        General entry point for both PDF and images.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.pdf']:
            return self.process_pdf(file_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            return self.process_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to PDF or image file")
    args = parser.parse_args()
    if args.file:
        p = DocParser()
        data = p.process_file(args.file)
        print(f"Processed {args.file}, found {len(data['blocks'])} blocks.")
        with open(os.path.join(p.output_dir, f"{os.path.basename(args.file)}.json"), "w") as f:
            json.dump(data, f, indent=2)
