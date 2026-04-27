import re
import json
import os
from typing import List, Dict, Any, Tuple

class MCQDetector:
    """
    Stage 2: MCQ Detection & Selection Detection
    Detects question stems and option sets using regex and spatial heuristics.
    Improved for granular OCR results.
    """
    
    def __init__(self, json_data: Dict[str, Any] = None):
        self.data = json_data
        # Patterns for questions: 1. , 1) , Question 1:
        self.q_pattern = re.compile(r'^(\d+[\.\)]|Question\s+\d+[:\.]|Q\d+[:\.])', re.IGNORECASE)
        # Patterns for options: (A) , A) , A.
        self.opt_pattern = re.compile(r'^[\(\[]?([A-D1-4])[\)\]\.]', re.IGNORECASE)

    def detect_mcqs(self) -> List[Dict[str, Any]]:
        if not self.data:
            return []
            
        blocks = self.data.get("blocks", [])
        text_blocks = []
        for b in blocks:
            if b["type"] != "text": continue
            
            # Reconstruct text if not already present
            full_text = ""
            lines = b.get("content", [])
            for line in lines:
                for span in line.get("spans", []):
                    full_text += span.get("text", "") + " "
            b["text"] = full_text.strip()
            if b["text"]:
                text_blocks.append(b)
        
        # Sort by page and vertical position (y0)
        text_blocks.sort(key=lambda x: (x["page"], x["bbox"][1], x["bbox"][0]))
        
        mcqs = []
        current_q = None
        
        for block in text_blocks:
            text = block["text"]
            
            # 1. New Question Stem
            if self.q_pattern.match(text):
                if current_q:
                    mcqs.append(current_q)
                current_q = {
                    "question_text": text,
                    "bbox": block["bbox"],
                    "page": block["page"],
                    "options": []
                }
            # 2. Potential Option
            elif current_q and self.opt_pattern.match(text):
                match = self.opt_pattern.match(text)
                label = match.group(1).upper()
                current_q["options"].append({
                    "label": label,
                    "text": text,
                    "bbox": block["bbox"]
                })
            # 3. Continuation/Merging Heuristic
            elif current_q:
                last_opt = current_q["options"][-1] if current_q["options"] else None
                
                # Check distance from last element (either question stem or last option)
                prev_bbox = last_opt["bbox"] if last_opt else current_q["bbox"]
                v_dist = block["bbox"][1] - prev_bbox[3]
                h_dist = block["bbox"][0] - prev_bbox[0]
                
                if block["page"] == current_q["page"] and v_dist < 15:
                    if last_opt and abs(v_dist) < 5: # Horizontal continuation of option
                         last_opt["text"] += " " + text
                         last_opt["bbox"][2] = max(last_opt["bbox"][2], block["bbox"][2])
                         last_opt["bbox"][3] = max(last_opt["bbox"][3], block["bbox"][3])
                    elif not last_opt: # Continuation of question stem
                         current_q["question_text"] += " " + text
                         current_q["bbox"][2] = max(current_q["bbox"][2], block["bbox"][2])
                         current_q["bbox"][3] = max(current_q["bbox"][3], block["bbox"][3])
                    else: # Vertical continuation of last option text
                         last_opt["text"] += " " + text
                         last_opt["bbox"][3] = max(last_opt["bbox"][3], block["bbox"][3])
        
        if current_q:
            mcqs.append(current_q)
            
        return mcqs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, help="Path to processed PDF JSON")
    args = parser.parse_args()
    
    if args.json:
        with open(args.json, 'r') as f:
            data = json.load(f)
        detector = MCQDetector(data)
        mcqs = detector.detect_mcqs()
        print(f"Detected {len(mcqs)} MCQs.")
        
        for i, q in enumerate(mcqs):
            print(f"Q{i+1}: {q['question_text'][:50]}... ({len(q['options'])} options)")
