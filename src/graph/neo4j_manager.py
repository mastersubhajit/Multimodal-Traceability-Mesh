from neo4j import GraphDatabase
import os
import time
import hashlib
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from neo4j.exceptions import ServiceUnavailable, SessionExpired

load_dotenv()

class Neo4jManager:
    """
    Stage 3: Provenance Graph Construction — Neo4j
    Manages the creation and querying of the provenance knowledge graph.
    Improved with robust retry logic for cloud/remote connections.
    """
    
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def _run_with_retry(self, cypher: str, params: Dict[str, Any] = None, max_retries: int = 3):
        """
        Helper to run a query with exponential backoff on connection errors.
        """
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                with self.driver.session() as session:
                    result = session.run(cypher, **(params or {}))
                    # Consume result to ensure it's fetched before session closes
                    return [record for record in result]
            except (ServiceUnavailable, SessionExpired) as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Neo4j connection error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
        return []

    def setup_database(self):
        """
        Initializes constraints and indexes for optimized lookups.
        """
        print("Setting up database constraints and indexes...")
        self._run_with_retry("CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
        self._run_with_retry("CREATE CONSTRAINT page_id_unique IF NOT EXISTS FOR (p:Page) REQUIRE p.id IS UNIQUE")
        self._run_with_retry("CREATE CONSTRAINT block_id_unique IF NOT EXISTS FOR (b:Block) REQUIRE b.id IS UNIQUE")
        self._run_with_retry("CREATE CONSTRAINT question_id_unique IF NOT EXISTS FOR (q:Question) REQUIRE q.id IS UNIQUE")
        self._run_with_retry("CREATE CONSTRAINT option_id_unique IF NOT EXISTS FOR (o:Option) REQUIRE o.id IS UNIQUE")
        self._run_with_retry("CREATE CONSTRAINT answer_id_unique IF NOT EXISTS FOR (a:Answer) REQUIRE a.id IS UNIQUE")
        print("Database setup complete.")

    def create_document_node(self, doc_id: str, filename: str, page_count: int, image_path: str = None):
        self._run_with_retry("""
            MERGE (d:Document {id: $doc_id})
            SET d.filename = $filename, d.page_count = $page_count, d.image_path = $image_path, d.created_at = datetime()
        """, {"doc_id": doc_id, "filename": filename, "page_count": page_count, "image_path": image_path})

    def create_docvqa_question_node(self, doc_id: str, question_id: str, question_text: str, answers: List[str]):
        self._run_with_retry("""
            MATCH (d:Document {id: $doc_id})
            MERGE (q:Question {id: $question_id})
            SET q.question_text = $question_text, q.answers = $answers, q.doc_id = $doc_id
            MERGE (d)-[:HAS_QUESTION]->(q)
        """, {"doc_id": doc_id, "question_id": question_id, "question_text": question_text, "answers": answers})

    def create_page_node(self, doc_id: str, page_no: int, width: float, height: float):
        self._run_with_retry("""
            MATCH (d:Document {id: $doc_id})
            MERGE (p:Page {id: $page_id})
            SET p.page_no = $page_no, p.width = $width, p.height = $height, p.doc_id = $doc_id
            MERGE (d)-[:HAS_PAGE]->(p)
        """, {"doc_id": doc_id, "page_id": f"{doc_id}_p{page_no}", "page_no": page_no, "width": width, "height": height})

    def create_block_node(self, doc_id: str, page_no: int, block: Dict[str, Any]):
        bbox_str = str(block.get('bbox'))
        bbox_hash = hashlib.md5(bbox_str.encode()).hexdigest()[:8]
        block_id = f"{doc_id}_p{page_no}_b{block.get('image_id', 'txt')}_{bbox_hash}"
        
        content = block.get("content", "")
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

        self._run_with_retry("""
            MERGE (p:Page {id: $page_id})
            ON CREATE SET p.page_no = $page_no, p.doc_id = $doc_id
            MERGE (b:Block {id: $block_id})
            SET b.type = $type, b.bbox = $bbox, b.text = $text, b.page_no = $page_no, b.doc_id = $doc_id
            MERGE (p)-[:CONTAINS]->(b)
        """, {"page_id": f"{doc_id}_p{page_no}", "block_id": block_id, "type": block["type"], 
              "bbox": block["bbox"], "text": clean_text, "page_no": page_no, "doc_id": doc_id})

    def create_mcq_structure(self, doc_id: str, mcq: Dict[str, Any], index: int):
        q_id = f"{doc_id}_q{index}"
        page_id = f"{doc_id}_p{mcq['page']}"
        
        # 1. Create Question
        self._run_with_retry("""
            MERGE (p:Page {id: $page_id})
            ON CREATE SET p.page_no = $page_no, p.doc_id = $doc_id
            MERGE (q:Question {id: $q_id})
            SET q.stem_text = $text, q.bbox = $bbox, q.page_no = $page_no, q.index = $index, q.doc_id = $doc_id
            MERGE (p)-[:CONTAINS]->(q)
        """, {"page_id": page_id, "q_id": q_id, "page_no": mcq["page"],
               "text": mcq["question_text"], "bbox": mcq["bbox"], "index": index, "doc_id": doc_id})
        
        # 2. Create Options
        for opt in mcq.get("options", []):
            opt_id = f"{q_id}_opt_{opt['label']}"
            self._run_with_retry("""
                MATCH (q:Question {id: $q_id})
                MERGE (o:Option {id: $opt_id})
                SET o.label = $label, o.text = $text, o.bbox = $bbox, o.page_no = $page_no, o.doc_id = $doc_id
                MERGE (q)-[:HAS_OPTION]->(o)
            """, {"q_id": q_id, "opt_id": opt_id, "label": opt["label"], 
                   "text": opt["text"], "bbox": opt["bbox"], "page_no": mcq["page"], "doc_id": doc_id})

    def log_ood_query(self, doc_id: str, query_text: str):
        """
        Logs a query that was detected as Out-of-Domain (OOD).
        """
        ood_id = f"ood_{hashlib.md5(query_text.encode()).hexdigest()[:10]}"
        self._run_with_retry("""
            MATCH (d:Document {id: $doc_id})
            MERGE (q:Question {id: $ood_id})
            SET q.question_text = $query_text, q.is_ood = True, q.timestamp = datetime()
            MERGE (d)-[:HAS_OOD_QUERY]->(q)
        """, {"doc_id": doc_id, "ood_id": ood_id, "query_text": query_text})

    def get_blocks_by_id(self, block_ids: List[str]) -> List[Dict[str, Any]]:
        records = self._run_with_retry("""
            MATCH (b:Block)
            WHERE b.id IN $block_ids
            RETURN b.id as id, b.text as text, b.bbox as bbox, b.page_no as page_no
        """, {"block_ids": block_ids})
        return [record.data() for record in records]

    def create_answer_node(self, q_id: str, answer_data: Dict[str, Any]):
        ans_id = f"{q_id}_ans"
        self._run_with_retry("""
            MATCH (q:Question {id: $q_id})
            MERGE (a:Answer {id: $ans_id})
            SET a.correct_label = $label, a.reasoning = $reasoning, 
                a.verification_result = $result, a.verified_at = datetime()
            MERGE (q)-[:HAS_ANSWER]->(a)
        """, {"q_id": q_id, "ans_id": ans_id, "label": answer_data["correct_label"], 
               "reasoning": answer_data["reasoning"], "result": answer_data["verification_result"]})
        
        for evidence_id in answer_data.get("evidence_ids", []):
            self._run_with_retry("""
                MATCH (a:Answer {id: $ans_id})
                MATCH (b:Block {id: $block_id})
                MERGE (a)-[:DERIVED_FROM]->(b)
                MERGE (a)-[:SUPPORTED_BY {relevance_score: 1.0}]->(b)
            """, {"ans_id": ans_id, "block_id": evidence_id})

    def get_mcq_context(self, doc_id: str, q_index: int):
        q_id = f"{doc_id}_q{q_index}"
        records = self._run_with_retry("""
            MATCH (q:Question {id: $q_id})
            OPTIONAL MATCH (q)-[:HAS_OPTION]->(o:Option)
            OPTIONAL MATCH (p:Page)-[:CONTAINS]->(q)
            OPTIONAL MATCH (p)-[:CONTAINS]->(b:Block)
            WITH q, collect(DISTINCT o) AS options,
                 [x IN collect(DISTINCT b) WHERE x.type = 'text'] AS context_blocks
            RETURN q, options, context_blocks
        """, {"q_id": q_id})
        return records[0] if records else None

    def ingest_eval_question(self, doc_id: str, q_id: str, question_text: str,
                             dataset: str = "", split: str = "test"):
        """
        Ingest a question node for evaluation WITHOUT storing the correct answer.
        The answer is kept only in the local evaluation JSON, not in the graph.
        """
        self._run_with_retry("""
            MATCH (d:Document {id: $doc_id})
            MERGE (q:Question {id: $q_id})
            SET q.question_text = $question_text,
                q.doc_id       = $doc_id,
                q.dataset      = $dataset,
                q.split        = $split,
                q.created_at   = datetime()
            MERGE (d)-[:HAS_QUESTION]->(q)
        """, {"doc_id": doc_id, "q_id": q_id, "question_text": question_text,
               "dataset": dataset, "split": split})

    def get_context_for_doc(self, doc_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        """Return text blocks for a document to use as RAG context (no answers)."""
        records = self._run_with_retry("""
            MATCH (d:Document {id: $doc_id})-[:HAS_PAGE]->(p:Page)-[:CONTAINS]->(b:Block)
            WHERE b.type = 'text' AND b.text IS NOT NULL AND trim(b.text) <> ''
            RETURN b.id AS id, b.text AS text, b.bbox AS bbox, b.page_no AS page_no
            ORDER BY b.page_no, b.bbox
            LIMIT $limit
        """, {"doc_id": doc_id, "limit": limit})
        return [r.data() for r in records]

    def get_question_by_id(self, q_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a question node (without answer) by its ID."""
        records = self._run_with_retry("""
            MATCH (q:Question {id: $q_id})
            RETURN q.question_text AS question_text, q.doc_id AS doc_id,
                   q.dataset AS dataset, q.split AS split
        """, {"q_id": q_id})
        return records[0].data() if records else None

    def query(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        records = self._run_with_retry(cypher, params)
        return [record.data() for record in records]

    def clear_database(self):
        print("Deleting all nodes and relationships...")
        self._run_with_retry("MATCH (n) DETACH DELETE n")
        
        print("Dropping all constraints...")
        # Note: SHOW CONSTRAINTS cannot be run easily in _run_with_retry because it's not a standard DML
        with self.driver.session() as session:
            constraints = session.run("SHOW CONSTRAINTS")
            for record in constraints:
                session.run(f"DROP CONSTRAINT {record['name']}")
            
            print("Dropping all indexes...")
            indexes = session.run("SHOW INDEXES")
            for record in indexes:
                if record['type'] != 'LOOKUP':
                    session.run(f"DROP INDEX {record['name']}")
        print("Database cleared successfully.")

if __name__ == "__main__":
    manager = Neo4jManager()
    manager.close()
