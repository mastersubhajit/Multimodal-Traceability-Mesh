from typing import List, Dict, Any, Optional, Tuple
from src.graph.neo4j_manager import Neo4jManager
from src.rag.vector_index import VectorIndexManager
from src.vision.visualize import MeshVisualizer
from src.ingestion.parser import DocParser
from src.rag.vllm_engine import VLLMTurboEngine
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, MllamaForConditionalGeneration, AutoProcessor
from sentence_transformers import CrossEncoder
from PIL import Image

class RAGPipeline:
    """
    Stage 4 & 5: Multi-Stage Reasoning & Verification
    Coordinates retrieval from Neo4j and FAISS, reranks, then reasons using an LLM or VLM.
    """
    
    def __init__(self, doc_id: str, neo4j_manager: Neo4jManager, vector_index: VectorIndexManager, 
                 llm_path: Optional[str] = None, 
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 use_graph: bool = True,
                 use_vector: bool = True,
                 use_vision: bool = True,
                 use_vllm: bool = False,
                 tensor_parallel_size: int = 4):
        self.doc_id = doc_id
        self.neo4j = neo4j_manager
        self.vector_index = vector_index
        self.use_graph = use_graph
        self.use_vector = use_vector
        self.use_vision = use_vision
        self.parser = DocParser() # For getting page images
        self.mesh_viz = MeshVisualizer()
        
        print(f"Loading CrossEncoder: {cross_encoder_model}")
        try:
            self.cross_encoder = CrossEncoder(cross_encoder_model)
        except Exception as e:
            print(f"Failed to load CrossEncoder: {e}")
            self.cross_encoder = None
            
        self.llm_loaded = False
        self.is_vision_model = False
        self.vllm_engine = None

        if use_vllm and llm_path:
            try:
                self.vllm_engine = VLLMTurboEngine(
                    model_id=llm_path, 
                    tensor_parallel_size=tensor_parallel_size,
                    kv_cache_compression=True
                )
                self.llm_loaded = True
            except Exception as e:
                print(f"Failed to initialize VLLM Turbo Engine: {e}")
        
        if not self.vllm_engine and llm_path and os.path.exists(llm_path):
            try:
                print(f"Loading Model from {llm_path} (Transformers Fallback)...")
                if "vision" in llm_path.lower() or "mllama" in llm_path.lower():
                    self.llm = MllamaForConditionalGeneration.from_pretrained(
                        llm_path, device_map="auto", torch_dtype=torch.bfloat16
                    )
                    self.processor = AutoProcessor.from_pretrained(llm_path)
                    self.is_vision_model = True
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
                    self.llm = AutoModelForCausalLM.from_pretrained(
                        llm_path, device_map="auto", torch_dtype=torch.bfloat16
                    )
                self.llm_loaded = True
            except Exception as e:
                print(f"Failed to load LLM/VLM: {e}")

    def rerank_evidence(self, query: str, evidence_blocks: List[str], top_k: int = 5) -> List[str]:
        if not self.cross_encoder or not evidence_blocks:
            return evidence_blocks[:top_k]
            
        pairs = [[query, block] for block in evidence_blocks]
        scores = self.cross_encoder.predict(pairs)
        
        scored_blocks = sorted(zip(scores, evidence_blocks), key=lambda x: x[0], reverse=True)
        return [block for score, block in scored_blocks[:top_k]]

    def retrieve_context(self, q_index: int, top_k: int = 5) -> Dict[str, Any]:
        """
        Hybrid retrieval: Neo4j traversal + FAISS vector search + CrossEncoder reranking.
        """
        graph_record = self.neo4j.get_mcq_context(self.doc_id, q_index)
        if not graph_record:
            return {}
            
        question = graph_record["q"]
        options = graph_record["options"]
        
        graph_blocks = []
        if self.use_graph:
            graph_blocks = [b["text"] for b in graph_record.get("context_blocks", [])]
        
        vector_results = []
        vector_texts = []
        vector_blocks_data = []
        if self.use_vector:
            # 2. Vector search results
            vector_results = self.vector_index.query(question["stem_text"], top_k=top_k*2)
            vector_block_ids = [r["id"] for r in vector_results]
            
            # 3. Retrieve text for vector results from Neo4j
            vector_blocks_data = self.neo4j.get_blocks_by_id(vector_block_ids)
            vector_texts = [b["text"] for b in vector_blocks_data]
        
        # Combine evidence
        combined_evidence = list(set(graph_blocks + vector_texts))
        
        # Re-rank combined evidence
        reranked_evidence = self.rerank_evidence(question["stem_text"], combined_evidence, top_k=top_k)
        
        return {
            "question": question,
            "options": options,
            "graph_context": reranked_evidence,
            "vector_evidence": vector_results,
            "vector_blocks_data": vector_blocks_data
        }

    def update_graph_with_verification(self, q_index: int, derivation: Dict[str, Any], is_correct: str, evidence_ids: List[str]):
        """
        Stage 6: Graph Update
        Writes verification result back to provenance graph.
        """
        q_id = f"{self.doc_id}_q{q_index}"
        answer_node_data = {
            "correct_label": derivation.get("correct_option", "N/A"),
            "reasoning": derivation.get("reasoning", derivation.get("rationale", "N/A")),
            "verification_result": "CORRECT" if is_correct == "Yes" else "INCORRECT" if is_correct == "No" else "NA",
            "evidence_ids": evidence_ids
        }
        self.neo4j.create_answer_node(q_id, answer_node_data)
        print(f"Graph updated for Q{q_index} with verification result.")

    def derive_answer(self, context: Dict[str, Any], file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Stage 4: Answer Derivation
        Uses the retrieved context to determine the correct answer via LLM/VLM Grounding.
        If a vision model is loaded, it generates a 'Visual Mesh' overlay for reasoning.
        """
        if not context or "question" not in context:
            return {
                "correct_option": "N/A",
                "rationale": "Insufficient context retrieved from graph/vector index.",
                "evidence_ids": [],
                "is_ood": True
            }

        q_text = context["question"]["stem_text"]
        opts = [f"{o['label']}: {o['text']}" for o in context["options"]]
        
        # Check if we have any actual evidence
        evidence_blocks = context.get("graph_context", [])
        if not evidence_blocks:
             return {
                "correct_option": "N/A",
                "rationale": f"In the document it has no mention of the information needed to answer: '{q_text}'. The document focuses on other sections.",
                "evidence_ids": [],
                "is_ood": True
            }

        prompt = f"""<|system|>
You are a document analysis assistant with expertise in multimodal reasoning.
You analyze PDF documents and images to detect MCQ questions, identify selected options, determine correct answers with grounded evidence.
If the question cannot be answered using the provided evidence, explicitly state that "in the document it has no mention of that" and provide reasoning based on your general knowledge of what the document covers.
<|user|>
"""
        image = None
        if self.use_vision and self.is_vision_model and file_path:
            # Generate Visual Mesh
            try:
                page_no = context["question"]["page_no"]
                page_img = self.parser.get_page_image(file_path, page_no)
                page_size = (self.parser.doc_data["pages"][page_no]["width"], 
                             self.parser.doc_data["pages"][page_no]["height"])
                
                # Prepare nodes for MeshVisualizer
                nodes = []
                nodes.append({"bbox": context["question"]["bbox"], "label": "QUESTION", "type": "question"})
                for opt in context["options"]:
                    nodes.append({"bbox": opt["bbox"], "label": f"OPT_{opt['label']}", "type": "option"})
                
                for i, block in enumerate(context.get("vector_blocks_data", [])):
                    nodes.append({"bbox": block["bbox"], "label": f"EVIDENCE_{i}", "type": "evidence"})
                
                # Relationships: Q -> Options, Q -> Evidence (spatial/contextual)
                relationships = []
                for i in range(1, len(context["options"]) + 1):
                    relationships.append((0, i, "HAS_OPTION"))
                for i in range(len(context["options"]) + 1, len(nodes)):
                    relationships.append((0, i, "SUPPORTED_BY"))
                    
                image = self.mesh_viz.draw_mesh_on_image(page_img, nodes, relationships, page_size)
                prompt += "<image>\n"
            except Exception as e:
                print(f"Vision mesh generation failed: {e}")

        # Build numbered evidence list for citation grounding
        numbered_evidence = "\n".join(
            f"[{i+1}] {blk}" for i, blk in enumerate(evidence_blocks)
        )
        prompt += f"""Evidence blocks (cite by number, e.g. [1]):
{numbered_evidence}

Question: {q_text}
Options: {', '.join(opts)}
Task: Select the correct option and justify your answer by citing the evidence block numbers above. 
If the evidence is irrelevant, state: "In the document it has no mention of that" followed by your reasoning.

Format your answer as:
REASONING: <your reasoning with inline citations like [1], [2]>
CORRECT_OPTION: <single letter or N/A>
<|assistant|>
"""
        if not self.llm_loaded:
            return {
                "correct_option": "A",
                "reasoning": f"(Mock) Based on [1]: '{evidence_blocks[0][:60] if evidence_blocks else ''}...', option A seems correct.",
                "citations": [1] if evidence_blocks else [],
                "confidence": 0.85,
                "is_ood": False
            }
            
        try:
            if self.vllm_engine:
                response = self.vllm_engine.generate([prompt], max_tokens=200)[0]
            elif self.is_vision_model:
                # Vision model path — processor handles both image+text and text-only inputs
                if image is not None:
                    inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.llm.device)
                else:
                    inputs = self.processor(text=prompt, return_tensors="pt").to(self.llm.device)
                with torch.no_grad():
                    outputs = self.llm.generate(**inputs, max_new_tokens=200, do_sample=False)
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
                with torch.no_grad():
                    outputs = self.llm.generate(**inputs, max_new_tokens=200, do_sample=False)
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract assistant turn
            if "<|start_header_id|>assistant<|end_header_id|>" in response:
                assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1] \
                                             .replace("<|eot_id|>", "").strip()
            elif "<|assistant|>" in response:
                assistant_response = response.split("<|assistant|>")[-1].strip()
            else:
                assistant_response = response.strip()

            correct_option = "N/A"
            if "CORRECT_OPTION:" in assistant_response:
                raw = assistant_response.split("CORRECT_OPTION:")[-1].strip()
                correct_option = raw[0] if raw else "N/A"

            is_ood = "in the document it has no mention of that" in assistant_response.lower()

            # Extract cited evidence indices from [n] patterns
            import re
            cited_indices = [int(m) - 1 for m in re.findall(r'\[(\d+)\]', assistant_response)
                             if m.isdigit() and 0 < int(m) <= len(evidence_blocks)]

            return {
                "correct_option": correct_option,
                "reasoning":      assistant_response,
                "citations":      cited_indices,
                "confidence":     0.9 if not is_ood else 0.3,
                "is_ood":         is_ood
            }
        except Exception as e:
            print(f"LLM/VLM Generation failed: {e}")
            return {
                "correct_option": "Error",
                "reasoning": str(e),
                "confidence": 0.0,
                "is_ood": False
            }

    def handle_query(self, query_text: str, threshold: float = 1.0) -> Dict[str, Any]:
        """
        Main entry point for any user query (MCQ or Open-ended).
        Includes Out-of-Domain (OOD) detection and citation-grounded answers.
        """
        import re
        vector_results = self.vector_index.query(query_text, top_k=5)

        is_retrieval_poor = not vector_results or vector_results[0]["score"] > threshold

        if is_retrieval_poor:
            print(f"Potential OOD detected for query: '{query_text}'")
            self.neo4j.log_ood_query(self.doc_id, query_text)

            # Generate a reasoned refusal if LLM is loaded
            refusal_reasoning = "The retrieval system could not find relevant context."
            if self.llm_loaded:
                try:
                    refusal_prompt = (
                        f"<|system|>\nYou are a document analysis assistant. "
                        f"If a question is asked about something not in the document, you MUST respond: "
                        f"'In the document it has no mention of that' followed by reasoning based on what you know about the document's content.\n"
                        f"<|user|>\nQuestion: {query_text}\nDocument context: (No relevant information found in retrieved blocks)\n<|assistant|>\n"
                    )
                    refusal_reasoning = self._run_inference(refusal_prompt, max_new_tokens=150)
                except Exception as e:
                    refusal_reasoning = f"Error generating refusal: {e}"

            return {
                "answer":     refusal_reasoning if "in the document it has no mention" in refusal_reasoning.lower() else f"In the document it has no mention of that. {refusal_reasoning}",
                "is_ood":     True,
                "confidence": "LOW (OOD)",
                "reasoning":  refusal_reasoning,
                "citations":  [],
            }

        # Retrieve block texts for evidence
        block_ids = [r["id"] for r in vector_results]
        blocks_data = self.neo4j.get_blocks_by_id(block_ids)
        evidence_blocks = [b["text"] for b in blocks_data]
        numbered_evidence = "\n".join(f"[{i+1}] {blk}" for i, blk in enumerate(evidence_blocks))

        if not self.llm_loaded:
            return {
                "answer":       " ".join(evidence_blocks) if evidence_blocks else "",
                "is_ood":       False,
                "citations":    list(range(len(evidence_blocks))),
                "evidence_ids": block_ids,
            }

        try:
            open_prompt = (
                f"<|system|>\nYou are a document analysis assistant. "
                f"Answer the question using only the provided evidence blocks. "
                f"Cite evidence inline using [n] notation.\n"
                f"<|user|>\nEvidence blocks:\n{numbered_evidence}\n\n"
                f"Question: {query_text}\n<|assistant|>\n"
            )
            answer_text = self._run_inference(open_prompt, max_new_tokens=150)

            cited_indices = [int(m) - 1 for m in re.findall(r'\[(\d+)\]', answer_text)
                             if m.isdigit() and 0 < int(m) <= len(evidence_blocks)]
            return {
                "answer":       answer_text,
                "is_ood":       False,
                "citations":    cited_indices,
                "evidence_ids": block_ids,
            }
        except Exception as e:
            return {
                "answer":       evidence_blocks[0] if evidence_blocks else "",
                "is_ood":       False,
                "citations":    [],
                "evidence_ids": block_ids,
                "error":        str(e),
            }

    def retrieve_by_query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Text-based retrieval for evaluation: FAISS vector search + CrossEncoder reranking.
        Does NOT require MCQ indices — works for any free-text query.
        """
        vector_results = self.vector_index.query(query_text, top_k=top_k * 2) if self.use_vector else []
        retrieved_ids = [r["id"] for r in vector_results]

        blocks_data = self.neo4j.get_blocks_by_id(retrieved_ids) if retrieved_ids else []
        block_texts = [b["text"] for b in blocks_data if b.get("text")]

        if self.cross_encoder and block_texts:
            block_texts = self.rerank_evidence(query_text, block_texts, top_k=top_k)

        return {
            "question_text": query_text,
            "evidence_blocks": block_texts[:top_k],
            "evidence_ids": retrieved_ids[:top_k],
            "vector_results": vector_results[:top_k],
        }

    def _run_inference(self, prompt: str, image=None, max_new_tokens: int = 200) -> str:
        """Shared inference helper for both RAG and base-model generation."""
        if not self.llm_loaded:
            return "(no model loaded)"

        if self.vllm_engine:
            resp = self.vllm_engine.generate([prompt], max_tokens=max_new_tokens)[0]
            return resp.split("<|assistant|>")[-1].strip() if "<|assistant|>" in resp else resp.strip()

        if self.is_vision_model:
            if image is not None:
                inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.llm.device)
            else:
                inputs = self.processor(text=prompt, return_tensors="pt").to(self.llm.device)
            with torch.no_grad():
                out = self.llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            resp = self.processor.decode(out[0], skip_special_tokens=True)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
            with torch.no_grad():
                out = self.llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            resp = self.tokenizer.decode(out[0], skip_special_tokens=True)

        if "<|start_header_id|>assistant<|end_header_id|>" in resp:
            resp = resp.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        elif "<|assistant|>" in resp:
            resp = resp.split("<|assistant|>")[-1]
        return resp.strip()

    def generate_rag_answer(self, query_text: str, image=None,
                            doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        RAG-adapted model evaluation path.
        Retrieves graph + vector context, then generates an answer with citation grounding.
        """
        context = self.retrieve_by_query(query_text)
        evidence_blocks = context.get("evidence_blocks", [])
        evidence_ids = context.get("evidence_ids", [])

        # OOD detection: no evidence found or retrieval confidence too low
        is_ood = not evidence_blocks or (
            context["vector_results"] and context["vector_results"][0]["score"] > 1.0
        )

        if is_ood:
            if doc_id:
                self.neo4j.log_ood_query(doc_id, query_text)
            ood_prompt = (
                "<|system|>\nYou are a document analysis assistant. "
                "If the question is not addressed by the document, respond: "
                "'In the document it has no mention of that' and explain what the document covers.\n"
                f"<|user|>\nQuestion: {query_text}\nDocument context: (No relevant blocks found)\n"
                "<|assistant|>\n"
            )
            reasoning = self._run_inference(ood_prompt, image, max_new_tokens=150)
            full = reasoning if "in the document it has no mention" in reasoning.lower() \
                else f"In the document it has no mention of that. {reasoning}"
            return {
                "answer": full, "reasoning": reasoning,
                "citations": [], "evidence_ids": [],
                "is_ood": True, "confidence": "LOW",
            }

        numbered = "\n".join(f"[{i+1}] {blk}" for i, blk in enumerate(evidence_blocks))
        img_tag = "<image>\n" if (self.use_vision and self.is_vision_model and image is not None) else ""
        prompt = (
            "<|system|>\nYou are a document analysis assistant. "
            "Answer using the provided evidence blocks with inline [n] citations. "
            "If the evidence does not support the question, say 'In the document it has no mention of that'.\n"
            f"<|user|>\n{img_tag}"
            f"Evidence blocks:\n{numbered}\n\n"
            f"Question: {query_text}\n"
            "<|assistant|>\n"
        )
        answer = self._run_inference(prompt, image if self.is_vision_model else None)

        import re as _re
        cited = [int(m) - 1 for m in _re.findall(r'\[(\d+)\]', answer)
                 if m.isdigit() and 0 < int(m) <= len(evidence_blocks)]
        is_ood_resp = "in the document it has no mention" in answer.lower()

        return {
            "answer": answer,
            "reasoning": answer,
            "citations": cited,
            "evidence_ids": evidence_ids,
            "evidence_blocks": evidence_blocks,
            "is_ood": is_ood_resp,
            "confidence": 0.3 if is_ood_resp else 0.9,
        }

    def generate_base_answer(self, query_text: str, image=None) -> Dict[str, Any]:
        """
        Base model evaluation path — no graph context is provided.
        Used to measure how much the graph context helps the RAG-adapted model.
        """
        img_tag = "<image>\n" if (self.use_vision and self.is_vision_model and image is not None) else ""
        prompt = (
            "<|system|>\nYou are a helpful document analysis assistant. "
            "Answer the question based on the provided image or your general knowledge.\n"
            f"<|user|>\n{img_tag}Question: {query_text}\n"
            "<|assistant|>\n"
        )
        answer = self._run_inference(prompt, image if self.is_vision_model else None)
        return {"answer": answer, "reasoning": answer, "citations": [],
                "evidence_ids": [], "is_ood": False}

    def verify_selection(self, q_index: int, detected_selection: Optional[str] = None) -> Dict[str, Any]:
        """
        Stage 5: Verification
        Compares user selection with derived correct answer.
        """
        context = self.retrieve_context(q_index)
        if not context:
            return {"error": "MCQ not found"}

        derivation = self.derive_answer(context)

        is_correct = "N/A"
        if detected_selection:
            is_correct = "Yes" if detected_selection == derivation["correct_option"] else "No"

        return {
            "question_index": q_index,
            "detected_selection": detected_selection,
            "correct_answer": derivation.get("correct_option", "N/A"),
            "is_selected_correct": is_correct,
            "reasoning": derivation.get("reasoning", derivation.get("rationale", "N/A")),
            "evidence_used": context.get("graph_context", [])
        }

if __name__ == "__main__":
    pass
