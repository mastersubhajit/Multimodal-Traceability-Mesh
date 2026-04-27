from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
import os
import uuid
import shutil
from typing import List, Optional
from scripts.ingest_to_graph import main as ingest_main
from scripts.verify_answers import main as verify_main
from src.graph.neo4j_manager import Neo4jManager
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Multimodal Traceability Mesh API")

# Temporary storage for uploaded PDFs
UPLOAD_DIR = "data/raw/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class VerificationRequest(BaseModel):
    pdf_path: str
    clear_db: Optional[bool] = False

@app.post("/ingest")
async def ingest_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Uploads and triggers background ingestion of a PDF."""
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    background_tasks.add_task(ingest_main, file_path)
    
    return {"status": "Ingestion started", "file_id": file_id, "file_path": file_path}

@app.post("/verify")
async def verify_document(request: VerificationRequest):
    """Triggers the verification pipeline for an already ingested document."""
    if not os.path.exists(request.pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    try:
        # Running the verification logic
        # In a production setup, this would also be a background task
        results = verify_main(request.pdf_path)
        return {"status": "Success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/provenance/{q_id}")
async def get_provenance(q_id: str):
    """Retrieves the full provenance chain for a specific question."""
    manager = Neo4jManager()
    try:
        chain = manager.get_provenance_chain(q_id)
        if not chain:
            return {"error": "Provenance not found"}
        return {"q_id": q_id, "chain": str(chain)}
    finally:
        manager.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
