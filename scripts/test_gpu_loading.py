import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

model_id = "meta-llama/Llama-3.2-1B-Instruct" # Small model for quick test

def test_load():
    print(f"Testing load of {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("Model loaded successfully!")
        print(f"Device map: {model.hf_device_map}")
        
        input_text = "What is multimodal traceability?"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=20)
        print(f"Response: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_load()
    else:
        print("CUDA not available on this node.")
