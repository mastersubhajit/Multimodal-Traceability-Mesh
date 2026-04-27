from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
import os
import json
from typing import List, Dict, Any, Tuple, Optional

class ProvenanceVisualizer:
    # ... (existing code)
    def __init__(self, output_dir: str = "data/processed/visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def draw_bbox(self, page: fitz.Page, bbox: List[float], color: tuple, label: str = ""):
        """Draws a colored rectangle and an optional label on a page."""
        rect = fitz.Rect(bbox)
        # Add a slight padding to the rectangle
        rect.x0 -= 2
        rect.y0 -= 2
        rect.x1 += 2
        rect.y1 += 2
        
        # Draw the rectangle
        page.draw_rect(rect, color=color, width=1.5)
        
        # Draw the label text just above the rectangle
        if label:
            # Using a simple built-in font
            page.insert_text((rect.x0, rect.y0 - 5), label, fontsize=8, color=color)

    def visualize_verification(self, pdf_path: str, verification_results: Dict[str, Any]):
        """
        Takes a PDF and the verification results JSON, and outputs a new annotated PDF.
        """
        doc = fitz.open(pdf_path)
        filename = os.path.basename(pdf_path)
        
        for q_res in verification_results.get("results", []):
            # 1. We need the original MCQ data to know the page and question bbox.
            # In a full pipeline, we'd retrieve this from the graph or pass it along.
            # For this standalone visualizer, we assume the question bbox and page are available.
            q_index = q_res["question_index"]
            # If the results JSON doesn't contain page/bbox info, we can't draw the question.
            # Assuming the pipeline was updated to include this metadata in the final output:
            
            page_no = q_res.get("page_no", 0) # Fallback to 0 if not provided
            
            if page_no >= len(doc):
                continue
                
            page = doc[page_no]
            
            # Color coding: Green for correct selection, Red for incorrect, Blue for evidence
            color = (1, 0, 0) # Default Red
            if q_res["is_selected_correct"] == "Yes":
                color = (0, 1, 0) # Green
            
            # Draw Question Bbox (if available)
            if "question_bbox" in q_res:
                self.draw_bbox(page, q_res["question_bbox"], color, f"Q{q_index}: {q_res['is_selected_correct']}")
                
            # Draw Selected Option Bbox (if available)
            if "selected_option" in q_res and q_res["selected_option"]:
                opt_bbox = q_res["selected_option"].get("detection_bbox")
                if opt_bbox:
                    self.draw_bbox(page, opt_bbox, (1, 0.5, 0), f"Selected: {q_res['detected_selection']}")
                    
            # Draw Evidence Blocks
            # Assuming evidence_used contains block references with bboxes
            for idx, evidence in enumerate(q_res.get("evidence", [])):
                if "bbox" in evidence and "page_no" in evidence:
                    ev_page = doc[evidence["page_no"]]
                    self.draw_bbox(ev_page, evidence["bbox"], (0, 0, 1), f"Evidence for Q{q_index} [{idx}]")
                    
        # Add summary page at the end
        if "score" in verification_results:
            summary_page = doc.new_page()
            score = verification_results["score"]
            text = f"Document: {verification_results.get('document', filename)}\n"
            text += f"Total Questions: {verification_results.get('total_questions', 0)}\n"
            text += f"Correct: {score.get('correct', 0)}\n"
            text += f"Incorrect: {score.get('incorrect', 0)}\n"
            text += f"Score: {score.get('percentage', 0):.2f}%\n"
            
            summary_page.insert_text((50, 50), "Verification Summary", fontsize=16, color=(0,0,0))
            summary_page.insert_text((50, 80), text, fontsize=12, color=(0,0,0))

        output_path = os.path.join(self.output_dir, f"{filename.replace('.pdf', '')}_annotated.pdf")
        doc.save(output_path)
        print(f"Visualized provenance saved to {output_path}")
        doc.close()

class MeshVisualizer:
    """
    New Component: Visual Graph Overlay
    Renders the Traceability Mesh (graph nodes/edges) directly on page images.
    This helps the Multimodal model 'see' the graph structure.
    """
    def __init__(self, font_path: Optional[str] = None):
        try:
            self.font = ImageFont.truetype(font_path, 15) if font_path else ImageFont.load_default()
        except:
            self.font = ImageFont.load_default()

    def draw_mesh_on_image(
        self, 
        image: Image.Image, 
        nodes: List[Dict[str, Any]], 
        relationships: List[Tuple[int, int, str]],
        page_size: Tuple[float, float]
    ) -> Image.Image:
        """
        nodes: List of dicts with 'bbox', 'label', 'type'
        relationships: List of (source_idx, target_idx, type)
        page_size: (width, height) in PDF points
        """
        draw = ImageDraw.Draw(image)
        img_w, img_h = image.size
        pdf_w, pdf_h = page_size
        
        # Helper to convert PDF coords to Image coords
        def to_img_coords(bbox):
            x0, y0, x1, y1 = bbox
            # Ensure coordinates are properly ordered and non-degenerate
            if x1 < x0: x0, x1 = x1, x0
            if y1 < y0: y0, y1 = y1, y0
            if x0 == x1: x1 += 1
            if y0 == y1: y1 += 1
            
            return [
                (x0 / pdf_w) * img_w,
                (y0 / pdf_h) * img_h,
                (x1 / pdf_w) * img_w,
                (y1 / pdf_h) * img_h
            ]

        # Draw nodes
        node_centers = []
        for node in nodes:
            img_bbox = to_img_coords(node["bbox"])
            color = "red" if node["type"] == "question" else "blue" if node["type"] == "evidence" else "green"
            draw.rectangle(img_bbox, outline=color, width=3)
            draw.text((img_bbox[0], img_bbox[1] - 20), node.get("label", ""), fill=color, font=self.font)
            # Store center for drawing edges
            node_centers.append(((img_bbox[0] + img_bbox[2]) / 2, (img_bbox[1] + img_bbox[3]) / 2))

        # Draw relationships (Mesh edges)
        for src_idx, tgt_idx, rel_type in relationships:
            if src_idx < len(node_centers) and tgt_idx < len(node_centers):
                start = node_centers[src_idx]
                end = node_centers[tgt_idx]
                draw.line([start, end], fill="yellow", width=2)
                # Draw a small circle at the end to indicate direction
                draw.ellipse([end[0]-5, end[1]-5, end[0]+5, end[1]+5], fill="yellow")
                
        return image

if __name__ == "__main__":
    # ... (existing main)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, required=True, help="Path to original PDF file")
    parser.add_argument("--results", type=str, required=True, help="Path to verification JSON results")
    args = parser.parse_args()
    
    if os.path.exists(args.pdf) and os.path.exists(args.results):
        with open(args.results, 'r') as f:
            data = json.load(f)
        visualizer = ProvenanceVisualizer()
        visualizer.visualize_verification(args.pdf, data)
    else:
        print("PDF or results file not found.")
