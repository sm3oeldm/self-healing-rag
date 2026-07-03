"""
FastAPI server for the self-healing RAG pipeline.

Provides:
  - GET  /health         — Health check
  - POST /query          — Single query (returns JSON)
  - POST /query/stream   — Stream query (SSE events for progress + tokens)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.schemas import HealthResponse, QueryRequest, QueryResponse
from src.config import (
    CHROMA_DB_PATH,
    CORS_ORIGINS,
    GOOGLE_API_KEY,
    SERVER_HOST,
    SERVER_PORT,
)
from src.pipeline.graph import arun_pipeline, stream_pipeline

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag-api")

# Track when the server started (used by health check)
_startup_time: str | None = None


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    global _startup_time
    from datetime import datetime, timezone
    _startup_time = datetime.now(timezone.utc).isoformat() + "Z"
    log.info(f"Self-Healing RAG API starting on {SERVER_HOST}:{SERVER_PORT}")
    yield
    log.info("Shutting down.")


# ── App Factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Self-Healing RAG Pipeline API",
    description="A RAG pipeline that critiques its own output and retries on hallucination.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the configured origins (defaults to all origins for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _llm_configured() -> bool:
    """Return True if at least one LLM provider is configured."""
    return bool(GOOGLE_API_KEY)


def _vectorstore_ready() -> bool:
    """Return True if the ChromaDB directory exists."""
    return os.path.isdir(CHROMA_DB_PATH)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    """Check whether the API and its dependencies are ready."""
    return HealthResponse(
        status="ok" if _vectorstore_ready() and _llm_configured() else "degraded",
        vectorstore_ready=_vectorstore_ready(),
        llm_configured=_llm_configured(),
    )


@app.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest):
    """Run the full self-healing RAG pipeline and return the result as JSON."""
    if not _vectorstore_ready():
        raise HTTPException(
            status_code=503,
            detail="Vector store not found. Run `python main.py` once to build it, "
                   "or call POST /admin/rebuild.",
        )
    if not _llm_configured():
        raise HTTPException(
            status_code=503,
            detail="No LLM API key configured. Set GOOGLE_API_KEY in your .env file.",
        )

    try:
        result = await arun_pipeline(body.question)
        return QueryResponse(
            answer=result.get("final_answer", "No answer generated."),
            verdict=result.get("verdict", "N/A"),
            retry_count=result.get("retry_count", 0),
            reason=result.get("reason"),
            original_question=result.get("original_question", body.question),
        )
    except Exception as exc:
        log.error(f"Pipeline error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query/stream")
async def query_stream(body: QueryRequest, request: Request):
    """
    Run the pipeline and stream events via Server-Sent Events.

    Event types yielded:
      - step:    pipeline stage lifecycle (start / end)
      - token:   an answer token from the generation step
      - retry:   retrieval is about to retry with a reformulated query
      - result:  final pipeline result (answer, verdict, …)
      - error:   a non-recoverable error occurred
    """
    if not _vectorstore_ready():
        raise HTTPException(status_code=503, detail="Vector store not ready.")
    if not _llm_configured():
        raise HTTPException(status_code=503, detail="LLM not configured.")

    async def event_stream() -> AsyncGenerator[bytes, None]:
        """Yield SSE-formatted bytes from the pipeline event generator."""
        try:
            async for event in stream_pipeline(body.question):
                # Respect client disconnection
                if await request.is_disconnected():
                    log.info("Client disconnected, stopping stream.")
                    break

                event_type = event.get("type", "unknown")
                data = event.get("data", event)  # pass through whole dict if no nested 'data'

                # Build SSE line
                payload = json.dumps(data, default=str)
                yield f"event: {event_type}\ndata: {payload}\n\n".encode("utf-8")

                # If the pipeline finished or errored, stop
                if event_type in ("result", "error"):
                    break

        except Exception as exc:
            log.error(f"Stream error: {exc}", exc_info=True)
            payload = json.dumps({"message": str(exc)}, default=str)
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


# ── CLI runner (when called as `python -m src.api.server`) ────────────────────
def run_server():
    """Start the uvicorn server. Called from main.py --serve or directly."""
    import uvicorn
    log.info(f"Starting server at http://{SERVER_HOST}:{SERVER_PORT}")
    uvicorn.run(
        "src.api.server:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
