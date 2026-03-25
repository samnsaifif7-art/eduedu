"""RAG router — ingest PDF, ask question, check status."""
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.models.schemas import IngestResponse, AskRequest, AskResponse, RAGStatusResponse
from backend.services.rag_service import rag_service

router = APIRouter(prefix="/api/rag", tags=["RAG Q&A"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload a PDF and index it for semantic search."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 50 MB.")
    try:
        chunks = rag_service.ingest_pdf(content, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return IngestResponse(status="success", filename=file.filename, chunks_indexed=chunks)


@router.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """Ask a question against the indexed document."""
    try:
        result = rag_service.ask(req.question, req.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return AskResponse(**result)


@router.get("/status", response_model=RAGStatusResponse)
async def rag_status():
    """Check whether a document is indexed."""
    return RAGStatusResponse(**rag_service.status())
