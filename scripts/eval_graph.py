import os
import json
import torch
import argparse
from src.graph.neo4j_manager import Neo4jManager
from transformers import MllamaProcessor, MllamaForConditionalGeneration
from peft import PeftModel
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

class GraphEvaluator:
    def __init__(self, model_id, adapter_path=None):
        hf_token = os.getenv("HF_TOKEN")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = MllamaProcessor.from_pretrained(model_id, token=hf_token)
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto", token=hf_token
        )
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.eval()
        self.neo4j = Neo4jManager()

    def generate_graph_questions(self, doc_id):
        """
        Synthesizes questions from the Neo4j graph for a specific document.
        """
        # 1. Total nodes and edges
        counts = self.neo4j.query(
            "MATCH (n {doc_id: $doc_id}) RETURN count(n) as node_count",
            {"doc_id": doc_id}
        )
        edge_counts = self.neo4j.query(
            "MATCH (n {doc_id: $doc_id})-[r]->() RETURN count(r) as edge_count",
            {"doc_id": doc_id}
        )

        node_count = counts[0]['node_count']
        edge_count = edge_counts[0]['edge_count']

        # 2. Highest degree node
        highest_deg = self.neo4j.query(
            "MATCH (n {doc_id: $doc_id})-[r]-() "
            "RETURN n.id as id, count(r) as degree ORDER BY degree DESC LIMIT 1",
            {"doc_id": doc_id}
        )

        # 3. Sample nodes for description (targeting Blocks or Questions)
        sample_nodes = self.neo4j.query(
            "MATCH (n {doc_id: $doc_id}) "
            "WHERE n:Block OR n:Question "
            "RETURN n LIMIT 5",
            {"doc_id": doc_id}
        )

        # 4. Triple listing
        triples = self.neo4j.query(
            "MATCH (n {doc_id: $doc_id})-[r]->(m) RETURN n.id as subject, type(r) as predicate, m.id as object LIMIT 5",
            {"doc_id": doc_id}
        )
        triple_str = ", ".join([f"({t['subject']}-{t['predicate']}->{t['object']})" for t in triples])
        
        questions = [
            {"type": "N. number", "question": f"How many nodes are in the graph for document {doc_id}?", "answer": str(node_count)},
            {"type": "E. number", "question": f"How many edges are in the graph for document {doc_id}?", "answer": str(edge_count)},
            {"type": "Triple listing", "question": f"List some triples (relationships) from the graph for document {doc_id}.", "answer": triple_str}
        ]
        
        if highest_deg:
            questions.append({
                "type": "Highest N. degree", 
                "question": f"Which node ID has the highest degree in the graph for document {doc_id}?", 
                "answer": str(highest_deg[0]['id'])
            })
            questions.append({
                "type": "N. degree",
                "question": f"What is the degree of node {highest_deg[0]['id']}?",
                "answer": str(highest_deg[0]['degree'])
            })

        for node in sample_nodes:
            # Safely get text from various node property names
            node_data = node['n']
            content = node_data.get('text') or node_data.get('question_text') or node_data.get('stem_text') or "N/A"
            
            questions.append({
                "type": "N. description",
                "question": f"Describe the content of node {node_data['id']}.",
                "answer": str(content)
            })

        return questions

    def evaluate(self, doc_id):
        questions = self.generate_graph_questions(doc_id)
        results = []
        
        print(f"Running graph comprehension evaluation for {doc_id}...")
        # Pre-fetch context once; extract text from whichever property exists per node type
        graph_context = self.neo4j.query(
            "MATCH (n {doc_id: $doc_id}) RETURN n LIMIT 20",
            {"doc_id": doc_id}
        )
        context_texts = [
            n['n'].get('text') or n['n'].get('stem_text') or n['n'].get('question_text') or ""
            for n in graph_context
        ]
        context_str = json.dumps([t for t in context_texts if t])

        for q in tqdm(questions):
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are an expert in graph analysis."}]},
                {"role": "user", "content": [{"type": "text", "text": f"Graph Context: {context_str}\nQuestion: {q['question']}"}]}
            ]
            
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=50, do_sample=False)
            
            generated = self.processor.decode(output[0], skip_special_tokens=True)
            if "assistant" in generated:
                answer = generated.split("assistant")[-1].strip()
            else:
                full_decoded = self.processor.decode(output[0], skip_special_tokens=False)
                if "<|start_header_id|>assistant<|end_header_id|>\n\n" in full_decoded:
                    answer = full_decoded.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].replace("<|eot_id|>", "").strip()
                else:
                    input_decoded = self.processor.decode(inputs["input_ids"][0], skip_special_tokens=True)
                    answer = generated[len(input_decoded):].strip()
            
            results.append({
                "type": q['type'],
                "question": q['question'],
                "expected": q['answer'],
                "generated": answer,
                "is_correct": q['answer'].lower() in answer.lower()
            })
            
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_id", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    
    evaluator = GraphEvaluator("meta-llama/Llama-3.2-11B-Vision-Instruct", args.adapter)
    results = evaluator.evaluate(args.doc_id)
    
    output_path = args.output_path if args.output_path else f"logs/eval_graph_{args.doc_id}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Simple accuracy print
    types = set(r['type'] for r in results)
    for t in types:
        type_res = [r for r in results if r['type'] == t]
        acc = sum(1 for r in type_res if r['is_correct']) / len(type_res)
        print(f"{t}: {acc:.2f}")
