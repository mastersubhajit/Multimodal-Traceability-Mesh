import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from typing import List, Dict, Any

class VectorIndexManager:
    """
    Stage 4: Vector Index Management
    Handles embedding of text blocks and similarity search.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.model = SentenceTransformer(model_name)
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.id_map = [] # To map index positions back to graph IDs

    def add_blocks(self, blocks: List[Dict[str, Any]], block_ids: List[str]):
        texts = []
        for b in blocks:
            content = b.get("content", "")
            clean_text = ""
            if isinstance(content, list):
                for line in content:
                    if isinstance(line, dict) and "spans" in line:
                        for span in line["spans"]:
                            clean_text += span.get("text", "") + " "
                    elif isinstance(line, str):
                        clean_text += line + " "
                clean_text = clean_text.strip()
            else:
                clean_text = str(content)
            texts.append(clean_text)

        if not texts:
            return
            
        embeddings = self.model.encode(texts)
        self.index.add(np.array(embeddings).astype('float32'))
        self.id_map.extend(block_ids)

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode([query_text])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "id": self.id_map[idx],
                    "score": float(distances[0][i])
                })
        return results

    def save(self, path: str):
        faiss.write_index(self.index, path + ".index")
        with open(path + ".ids", 'w') as f:
            for item in self.id_map:
                f.write("%s\n" % item)

    def load(self, path: str):
        self.index = faiss.read_index(path + ".index")
        with open(path + ".ids", 'r') as f:
            self.id_map = [line.strip() for line in f.readlines()]
