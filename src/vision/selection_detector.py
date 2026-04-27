from PIL import Image
from typing import Dict, Any, Optional
import torch
import os
from transformers import MllamaForConditionalGeneration, AutoProcessor

class SelectionDetector:
    """
    Stage 3: Option Selection Detection
    Detects if an option is marked (e.g., bubble, tick, highlight).
    Uses LLaMA 3.2 11B Vision Instruct (or a mock if model_path is not provided).
    """
    
    def __init__(self, model_path: Optional[str] = None, model=None, processor=None):
        self.model_loaded = False
        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
            self.model_loaded = True
            print("Vision model initialized from pre-loaded model.")
        elif model_path and os.path.exists(model_path):
            try:
                print(f"Loading Vision model from {model_path}...")
                self.model = MllamaForConditionalGeneration.from_pretrained(
                    model_path, 
                    device_map="auto", 
                    torch_dtype=torch.bfloat16
                )
                self.processor = AutoProcessor.from_pretrained(model_path)
                self.model_loaded = True
                print("Vision model loaded successfully.")
            except Exception as e:
                print(f"Failed to load vision model: {e}")

    def detect_selection(self, option_crop: Image.Image) -> Dict[str, Any]:
        """
        Input: Cropped image of an option.
        Output: Selection confidence and marker type.
        """
        if not self.model_loaded:
            # Fallback to mock detection
            return {
                "is_selected": False,
                "confidence": 0.5,
                "marker_type": "none"
            }
            
        prompt = "<|system|>\nYou analyze cropped images of MCQ options.\n<|user|>\nDoes this image show a selected/marked/circled/highlighted answer option? Answer YES or NO and describe the selection marker type.\n<|assistant|>\n"
        
        try:
            inputs = self.processor(images=option_crop, text=prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=50)
            
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            # Basic parsing of the response
            response_upper = response.upper()
            is_selected = "YES" in response_upper and "NO" not in response_upper[:10]
            
            marker_type = "unknown"
            if "BUBBLE" in response_upper: marker_type = "filled_bubble"
            elif "TICK" in response_upper or "CHECK" in response_upper: marker_type = "checkmark"
            elif "CIRCLE" in response_upper: marker_type = "circle"
            elif "HIGHLIGHT" in response_upper: marker_type = "highlight"
            
            return {
                "is_selected": is_selected,
                "confidence": 0.9, # Mock confidence
                "marker_type": marker_type if is_selected else "none"
            }
        except Exception as e:
            print(f"Vision model inference failed: {e}")
            return {
                "is_selected": False,
                "confidence": 0.0,
                "marker_type": "error"
            }

    def process_mcq_options(self, parser, file_path: str, mcq: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crops each option image and runs detection.
        """
        page_img = parser.get_page_image(file_path, mcq["page"])
        page_size = (parser.doc_data["pages"][mcq["page"]]["width"], 
                     parser.doc_data["pages"][mcq["page"]]["height"])
                     
        for opt in mcq["options"]:
            # Crop option (with slight margin for marker)
            bbox = opt["bbox"]
            margin = 10
            crop_bbox = [bbox[0]-margin, bbox[1], bbox[2], bbox[3]] # Adjust x0 to catch marker
            crop_img = parser.crop_region(page_img, crop_bbox, page_size)
            
            result = self.detect_selection(crop_img)
            opt["is_selected"] = result["is_selected"]
            opt["selection_confidence"] = result["confidence"]
            opt["selection_marker_type"] = result["marker_type"]
            
        return mcq
