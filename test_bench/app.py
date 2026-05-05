"""
FastAPI Test Bench — multi-file invoice processing via Modal OCR + GPT-4o-mini.

Run:
    python -m test_bench.app

Serves:
    GET  /           → Drag-drop upload UI
    POST /process    → Multi-file upload → JSON results
    GET  /health     → Health check
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()

from test_bench.services.structured_extractor import StructuredExtractor
from test_bench.services.sheets_service import GoogleSheetsService
from test_bench.services.gl_classifier import GLClassifier
from test_bench.pipeline import process_single_file


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DSOCR-2 Test Bench",
    description="Modal DeepSeek-OCR → GPT-4o-mini structured extraction → Google Sheets",
    version="1.0.0",
)

# ── Singletons (initialised lazily) ──────────────────────────────────────────
_extractor: Optional[StructuredExtractor] = None
_sheets_service: Optional[GoogleSheetsService] = None
_gl_classifier: Optional[GLClassifier] = None


def _get_extractor() -> StructuredExtractor:
    global _extractor
    if _extractor is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise HTTPException(500, "OPENAI_API_KEY not set")
        _extractor = StructuredExtractor(api_key=api_key)
    return _extractor


def _get_sheets_service() -> Optional[GoogleSheetsService]:
    global _sheets_service
    if _sheets_service is None:
        sheet_id = os.getenv("GOOGLE_SHEET_ID", "")
        if not sheet_id:
            print("[App] GOOGLE_SHEET_ID not set — Sheets logging disabled")
            return None
        try:
            from test_bench.utils.credentials_helper import get_credentials_path
            cred_path = get_credentials_path()
            _sheets_service = GoogleSheetsService(cred_path, sheet_id)
            print(f"[App] Sheets service initialised — sheet: {sheet_id}")
        except Exception as e:
            print(f"[App] Sheets service failed to initialise: {e}")
            return None
    return _sheets_service


def _get_gl_classifier() -> Optional[GLClassifier]:
    global _gl_classifier
    if _gl_classifier is None:
        mapping_sheet_id = os.getenv("GL_MAPPING_SHEET_ID", "")
        if not mapping_sheet_id:
            print("[App] GL_MAPPING_SHEET_ID not set — GL classification via sheet disabled (extractor GL still active)")
            return None
        sheets_svc = _get_sheets_service()
        if not sheets_svc:
            print("[App] Cannot init GL classifier without Sheets service")
            return None
        _gl_classifier = GLClassifier(sheets_svc, mapping_sheet_id)
    return _gl_classifier


# ── Static files ──────────────────────────────────────────────────────────────
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the drag-drop upload UI."""
    index_path = _static_dir / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=500)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "modal_url": os.getenv("MODAL_OCR_URL", "(default)"),
        "sheets_configured": bool(os.getenv("GOOGLE_SHEET_ID")),
        "gl_mapping_configured": bool(os.getenv("GL_MAPPING_SHEET_ID")),
    }


ALLOWED_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/jpg",
    "image/webp",
}
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".webp"}


@app.post("/process")
async def process_invoices(files: List[UploadFile] = File(...)):
    """
    Upload one or more invoice files (PDF, JPG, PNG).
    Returns JSON results with extracted data + per-invoice token cost.
    """
    if not files:
        raise HTTPException(400, "No files uploaded")

    # Validate files
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                400,
                f"Unsupported file type: {f.filename} ({ext}). "
                f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )

    extractor = _get_extractor()
    sheets_svc = _get_sheets_service()
    gl_clf = _get_gl_classifier()

    # Read all files into memory first
    file_data = []
    for f in files:
        content = await f.read()
        file_data.append({
            "bytes": content,
            "filename": f.filename or "unknown",
            "content_type": f.content_type or "application/octet-stream",
        })

    # Process all files concurrently
    tasks = [
        process_single_file(
            file_bytes=fd["bytes"],
            filename=fd["filename"],
            content_type=fd["content_type"],
            extractor=extractor,
            sheets_service=sheets_svc,
            gl_classifier=gl_clf,
        )
        for fd in file_data
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions from gather
    processed_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            processed_results.append({
                "filename": file_data[i]["filename"],
                "status": "error",
                "invoice_data": None,
                "token_usage": None,
                "error": f"{type(r).__name__}: {str(r)}",
                "sheets_logged": False,
            })
        else:
            processed_results.append(r)

    # Summary
    total_cost = sum(
        r.get("token_usage", {}).get("estimated_cost_usd", 0)
        for r in processed_results
        if r.get("token_usage")
    )
    success_count = sum(1 for r in processed_results if r["status"] == "success")

    return JSONResponse({
        "summary": {
            "total_files": len(files),
            "successful": success_count,
            "failed": len(files) - success_count,
            "total_estimated_cost_usd": round(total_cost, 6),
        },
        "results": processed_results,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "test_bench.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
