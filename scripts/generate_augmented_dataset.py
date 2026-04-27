import os
import json
import fitz  # PyMuPDF
import random
from PIL import Image, ImageDraw
import numpy as np

class AugmentedDatasetGenerator:
    """
    Stage 2 & 3: Augmented Dataset Generation
    Synthetically renders selected options (filled bubbles, checkmarks, etc.) to train the vision model.
    """
    def __init__(self, output_dir="data/processed/augmented_dataset"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        self.annotations = []
        
    def add_selection_marker(self, img: Image.Image, marker_type: str) -> Image.Image:
        """Adds a synthetic marker to an option image crop."""
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        width, height = img_copy.size
        
        # Typically the marker is on the left side
        marker_area = (5, height//4, 5 + height//2, height//4 + height//2)
        
        if marker_type == "filled_bubble":
            draw.ellipse(marker_area, fill="black", outline="black")
        elif marker_type == "checkmark":
            points = [(marker_area[0], marker_area[1] + height//4), 
                      (marker_area[0] + height//8, marker_area[3]), 
                      (marker_area[2], marker_area[1])]
            draw.line(points, fill="black", width=2)
        elif marker_type == "circle":
            draw.ellipse(marker_area, outline="black", width=2)
        elif marker_type == "highlight":
            # Highlight the text area
            highlight_area = (marker_area[2], height//4, width - 5, height - height//4)
            overlay = Image.new('RGBA', img_copy.size, (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            draw_overlay.rectangle(highlight_area, fill=(255, 255, 0, 128))
            img_copy = Image.alpha_composite(img_copy.convert('RGBA'), overlay).convert('RGB')
            
        return img_copy

    def generate_from_mcq_data(self, pdf_path: str, mcqs: list, parser):
        """Generates positive (selected) and negative (unselected) samples from MCQ data."""
        filename = os.path.basename(pdf_path)
        page_images = {}
        
        for idx, mcq in enumerate(mcqs):
            page_num = mcq["page"]
            if page_num not in page_images:
                page_images[page_num] = parser.get_page_image(pdf_path, page_num)
            
            page_img = page_images[page_num]
            page_size = (parser.doc_data["pages"][page_num]["width"], 
                         parser.doc_data["pages"][page_num]["height"])
            
            for opt_idx, opt in enumerate(mcq["options"]):
                # Crop the original (negative sample)
                bbox = opt["bbox"]
                margin = 15
                crop_bbox = [bbox[0]-margin, bbox[1]-margin//2, bbox[2]+margin, bbox[3]+margin//2]
                try:
                    crop_img = parser.crop_region(page_img, crop_bbox, page_size)
                    
                    # Save negative
                    neg_filename = f"{filename}_q{idx}_opt{opt_idx}_neg.jpg"
                    neg_path = os.path.join(self.output_dir, "images", neg_filename)
                    crop_img.save(neg_path)
                    self.annotations.append({
                        "image_path": neg_path,
                        "is_selected": False,
                        "marker_type": "none"
                    })
                    
                    # Generate and save positive
                    marker_type = random.choice(["filled_bubble", "checkmark", "circle", "highlight"])
                    pos_img = self.add_selection_marker(crop_img, marker_type)
                    pos_filename = f"{filename}_q{idx}_opt{opt_idx}_pos_{marker_type}.jpg"
                    pos_path = os.path.join(self.output_dir, "images", pos_filename)
                    pos_img.save(pos_path)
                    self.annotations.append({
                        "image_path": pos_path,
                        "is_selected": True,
                        "marker_type": marker_type
                    })
                except Exception as e:
                    print(f"Failed to crop/generate option: {e}")

    def save_annotations(self):
        output_file = os.path.join(self.output_dir, "annotations.json")
        with open(output_file, "w") as f:
            json.dump(self.annotations, f, indent=2)
        print(f"Saved {len(self.annotations)} annotations to {output_file}")

if __name__ == "__main__":
    # Example usage
    from src.ingestion.parser import DocParser
    from src.ingestion.mcq_detector import MCQDetector
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, required=True, help="Path to PDF file to augment from")
    args = parser.parse_args()
    
    if os.path.exists(args.pdf):
        print(f"Generating augmented dataset from {args.pdf}")
        doc_parser = DocParser(use_ocr=False, use_vision=False)
        doc_data = doc_parser.process_file(args.pdf)
        
        mcq_detector = MCQDetector(doc_data)
        mcqs = mcq_detector.detect_mcqs()
        
        generator = AugmentedDatasetGenerator()
        generator.generate_from_mcq_data(args.pdf, mcqs, doc_parser)
        generator.save_annotations()
    else:
        print("File not found.")
