import os
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

def download_datasets():
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("Warning: HF_TOKEN not found in .env. Some datasets may require authentication.")

    # Target Datasets from Blueprint with specific configs where needed
    datasets_to_download = [
        {"name": "DocVQA", "path": "lmms-lab/DocVQA", "config": "DocVQA"},
        {"name": "InfographicVQA", "path": "ayoubkirouane/infographic-VQA", "config": None},
        {"name": "TextVQA", "path": "lmms-lab/textvqa", "config": "default"}, 
        {"name": "VMCBench", "path": "suyc21/VMCBench", "config": None},
    ]

    os.makedirs("data/raw", exist_ok=True)

    for item in datasets_to_download:
        name = item["name"]
        path = item["path"]
        config = item["config"]
        
        print(f"Downloading {name} from {path} (config: {config})...")
        try:
            # We try without trust_remote_code as suggested by the error
            if config:
                ds = load_dataset(path, config, cache_dir="data/raw")
            else:
                ds = load_dataset(path, cache_dir="data/raw")
            print(f"Successfully downloaded {name}.")
        except Exception as e:
            print(f"Error downloading {name}: {e}")

if __name__ == "__main__":
    download_datasets()
