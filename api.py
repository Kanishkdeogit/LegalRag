import os
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query as FastQuery, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from legal_orchestrator import get_orchestrator, QueryResult, Citation

app = FastAPI(
    title="LegalRAG Assistant API",
    description="Production-grade Agentic Legal RAG API for User Uploaded Corpora, Commercial Contracts, and Indian Constitutional Law.",
    version="1.1.0"
)

# Enable CORS for Streamlit and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., description="Legal question or due diligence prompt", example="In the agreement, what is the limitation of liability?")
    domain: Optional[str] = Field("auto", description="Domain: 'user_corpus', 'auto', 'commercial_contracts', or 'constitutional_law'")
    doc_filter: Optional[str] = Field(None, description="Optional document name filter to strictly scope retrieval")
    top_k: Optional[int] = Field(None, description="Number of source sections to retrieve")


class CitationModel(BaseModel):
    document_name: str
    section: str
    citation_id: str
    hierarchy_level: int
    source: str
    excerpt: str
    rank: int


class QueryResponse(BaseModel):
    query: str
    answer: str
    domain: str
    doc_filter: Optional[str]
    citations: List[CitationModel]
    execution_time_seconds: float
    model: str
    embedding_model: str
    reranker_model: str


@app.get("/")
async def root():
    return {
        "service": "LegalRAG Assistant API",
        "status": "online",
        "version": "1.1.0",
        "domains": ["user_corpus", "commercial_contracts", "constitutional_law"],
        "endpoints": {
            "upload": "POST /api/upload",
            "clear_corpus": "POST /api/clear-user-corpus",
            "query": "POST /api/query",
            "documents": "GET /api/documents",
            "health": "GET /api/health",
            "docs": "/docs"
        }
    }


@app.post("/api/upload")
async def upload_user_documents(files: List[UploadFile] = File(...)):
    """
    Upload and index custom legal documents (PDF, TXT, MD) into the dedicated user corpus.
    Performs clause-aware chunking, vector embedding, and lexical BM25 indexing.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    file_tuples = []
    for f in files:
        content = await f.read()
        file_tuples.append((f.filename, content))

    try:
        orchestrator = get_orchestrator()
        result = await run_in_threadpool(orchestrator.ingest_user_documents, file_tuples)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest user documents: {str(e)}")


@app.post("/api/clear-user-corpus")
async def clear_user_corpus():
    """Clear and reset the active user uploaded corpus."""
    try:
        orchestrator = get_orchestrator()
        await run_in_threadpool(orchestrator.clear_user_corpus)
        return {"status": "success", "message": "User corpus cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear user corpus: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def execute_query(req: QueryRequest):
    """
    Execute full legal RAG pipeline:
    Agentic domain routing -> Hardware-filtered hybrid retrieval -> Small-to-Big expansion -> Answer generation with verified citations.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    try:
        orchestrator = get_orchestrator()
        result: QueryResult = await run_in_threadpool(
            orchestrator.query,
            query_text=req.query,
            domain=req.domain,
            doc_filter=req.doc_filter,
            top_k=req.top_k
        )
        return QueryResponse(
            query=result.query,
            answer=result.answer,
            domain=result.domain,
            doc_filter=result.doc_filter,
            citations=[CitationModel(**c) for c in result.citations],
            execution_time_seconds=result.execution_time_seconds,
            model=result.model,
            embedding_model=result.embedding_model,
            reranker_model=result.reranker_model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Legal RAG execution error: {str(e)}")


@app.get("/api/documents")
async def list_documents():
    """Return catalog of all indexed contracts and constitutional source documents."""
    try:
        orchestrator = get_orchestrator()
        catalog = await run_in_threadpool(orchestrator.get_catalog)
        return catalog
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve catalog: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check validating Qdrant Cloud, collections, and LLM readiness."""
    try:
        orchestrator = get_orchestrator()
        client = orchestrator.qdrant_client
        
        has_contracts = client.collection_exists("contracts_legal_rag")
        has_const = client.collection_exists("hierarchical_legal_rag")

        return {
            "status": "healthy" if (has_contracts and has_const) else "degraded",
            "qdrant": {
                "contracts_collection": has_contracts,
                "constitutional_collection": has_const
            },
            "llm": "gemini-3.5-flash-lite",
            "embeddings": "gemini-embedding-2",
            "reranker": "ms-marco-MiniLM-L-12-v2"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
