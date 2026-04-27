import os
import json
import torch
import numpy as np
from dotenv import load_dotenv
from typing import Tuple, Dict, List
from trulens.core import Feedback, TruSession as Tru
from trulens.core.feedback import provider as core_provider
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, MllamaForConditionalGeneration, MllamaProcessor

load_dotenv()

class LocalLLMProvider(core_provider.Provider):
    def __init__(self, model_id="meta-llama/Llama-3.3-70B-Instruct"):
        self.model_id = model_id
        print(f"Loading {model_id} in 4-bit on GPUs...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.is_vision = "vision" in model_id.lower() or "mllama" in model_id.lower()
        
        if self.is_vision:
            self.processor = MllamaProcessor.from_pretrained(model_id)
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        self.model.eval()
        super().__init__()

    def _generate(self, prompt: str) -> str:
        if self.is_vision:
            # For evaluation, we usually just need text reasoning. 
            # If we need vision, we'd pass an image here.
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=150,
                temperature=0.1,
                do_sample=False
            )
        
        input_len = inputs.input_ids.shape[1]
        if self.is_vision:
            response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        else:
            response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return response.strip()

    def groundedness_measure_with_cot_reasons(self, source: str, statement: str) -> Tuple[float, str]:
        prompt = f"""[INST] You are an expert evaluator. Analyze if the following Statement is grounded in the Source provided.
        
Source: {source}
Statement: {statement}

Provide your evaluation in the following format:
Score: <a value between 0.0 and 1.0>
Reason: <brief explanation>
[/INST]"""
        response = self._generate(prompt)
        score = 0.0
        try:
            for line in response.split('\n'):
                if "score:" in line.lower():
                    score_str = line.lower().split("score:")[1].strip().split()[0]
                    score = float(score_str)
        except:
            pass
        return score, response

    def relevance(self, prompt: str, response: str) -> float:
        query = f"[INST] Evaluate the relevance of the Response to the given Prompt.
Prompt: {prompt}
Response: {response}

Provide a relevance score between 0.0 and 1.0. Just the number.
[/INST] Score:"
        res = self._generate(query)
        try:
            return float(res.split()[0])
        except:
            return 0.5

    def qs_relevance(self, question: str, context: str) -> float:
        return self.relevance(question, context)

class Evaluator:
    def __init__(self, model_id="meta-llama/Llama-3.3-70B-Instruct"):
        self.tru = Tru()
        self.provider = LocalLLMProvider(model_id=model_id)

    def evaluate_results(self, verification_results_path: str):
        if not os.path.exists(verification_results_path):
            print(f"File not found: {verification_results_path}")
            return

        with open(verification_results_path, 'r') as f:
            data = json.load(f)
            
        results = data.get("details") or data.get("results") or data.get("per_question_results")
        if not results:
            print("No results to evaluate.")
            return

        print(f"Evaluating {len(results)} samples using {self.provider.model_id}...")
        for res in results:
            question = res.get("question_text") or res.get("question") or ""
            answer = res.get("generated") or ""
            context = res.get("evidence") or res.get("reasoning") or ""
            
            scores = {}
            # Groundedness
            score, reason = self.provider.groundedness_measure_with_cot_reasons(context, answer)
            scores["Groundedness"] = score
            
            # Relevance
            scores["Answer Relevance"] = self.provider.relevance(question, answer)
            scores["Context Relevance"] = self.provider.qs_relevance(question, context)
            
            res["evaluation"] = scores
            
        eval_path = verification_results_path.replace(".json", "_evaluated_local.json")
        with open(eval_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Evaluation complete. Saved to {eval_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    args = parser.parse_args()
    
    evaluator = Evaluator(model_id=args.model)
    evaluator.evaluate_results(args.results)
