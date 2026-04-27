import os
import json
from trulens.apps.langchain import TruChain
from trulens.core import Feedback, TruSession as Tru
from trulens.providers.openai import OpenAI as fOpenAI
from trulens.providers.huggingface import HuggingfaceLocal
import numpy as np
from dotenv import load_dotenv
import torch

load_dotenv()

class Evaluator:
    """
    Stage 7: Evaluation & Observability
    Integrates TruLens for measuring Faithfulness, Groundedness, and Answer Relevancy.
    """
    def __init__(self, use_local=False, model_id="meta-llama/Llama-3.3-70B-Instruct"):
        self.tru = Tru()

        if use_local:
            print(f"Initializing local HuggingFace provider with {model_id}...")
            # Using HuggingfaceLocal which wraps a local model
            # Note: For 70B we definitely want 4-bit quantization
            self.provider = HuggingfaceLocal(
                model_engine=model_id,
                device="cuda",
                # Pass quantization config via task_kwargs or similar if supported, 
                # but trulens HuggingfaceLocal is a bit limited in its default init.
                # Often it's better to use a custom provider for complex local setups.
            )
        else:
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                # Configure TruLens to use Groq via OpenAI-compatible interface
                print("Configuring TruLens with GROQ provider...")
                self.provider = fOpenAI(
                    base_url="https://api.groq.com/openai/v1/",
                    api_key=groq_key,
                    model_engine="llama-3.3-70b-versatile"
                )
            else:
                # Initialize standard OpenAI provider (requires OPENAI_API_KEY in .env)
                try:
                    self.provider = fOpenAI()
                except Exception as e:
                    print(f"Warning: OpenAI provider could not be initialized for evaluation: {e}")
                    self.provider = None

    def setup_feedbacks(self):
        if not self.provider:
            return []
            
        # 1. Groundedness (Faithfulness)
        f_groundedness = (
            Feedback(self.provider.groundedness_measure_with_cot_reasons, name="Groundedness")
            .on_input_output()
        )
        
        # 2. Answer Relevance
        f_answer_relevance = (
            Feedback(self.provider.relevance, name="Answer Relevance")
            .on_input_output()
        )
        
        # 3. Context Relevance
        f_context_relevance = (
            Feedback(self.provider.qs_relevance, name="Context Relevance")
            .on_input()
            .on_output() # Ideally .on(context)
            .aggregate(np.mean)
        )
        
        return [f_groundedness, f_answer_relevance, f_context_relevance]

    def evaluate_results(self, verification_results_path: str):
        """
        Evaluates the generated verification results using TruLens feedback functions.
        """
        if not self.provider:
            print("Cannot evaluate without a valid provider (e.g., OpenAI API Key).")
            return
            
        with open(verification_results_path, 'r') as f:
            data = json.load(f)
            
        feedbacks = self.setup_feedbacks()
        
        # Support both eval_generation.py (details) and verify_answers.py (results)
        results = data.get("details") or data.get("results") or data.get("per_question_results")
        if not results:
            print("No per-sample results found in JSON.")
            return

        print(f"Evaluating {len(results)} samples from {verification_results_path}...")
        
        for res in results:
            # Extract question and answer based on available fields
            question = res.get("question_text") or res.get("question") or f"Sample {res.get('index')}"
            answer = res.get("generated") or f"Option {res.get('detected_selection')}"
            
            scores = {}
            for f in feedbacks:
                try:
                    # In many TruLens versions, feedback functions are called with (input, output)
                    # For groundedness, it often expects (context, output)
                    if f.name == "Groundedness":
                        context = res.get("evidence") or res.get("reasoning") or ""
                        score, reasons = self.provider.groundedness_measure_with_cot_reasons(context, answer)
                        scores[f.name] = score
                    else:
                        score = f(question, answer)
                        scores[f.name] = score
                except Exception as e:
                    scores[f.name] = None
                    
            res["evaluation"] = scores
            
        # Save evaluated results
        eval_path = verification_results_path.replace(".json", "_evaluated.json")
        with open(eval_path, "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"Evaluation complete. Saved to {eval_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="Path to verification JSON")
    args = parser.parse_args()
    
    evaluator = Evaluator()
    if os.path.exists(args.results):
        evaluator.evaluate_results(args.results)
    else:
        print("Results file not found.")
