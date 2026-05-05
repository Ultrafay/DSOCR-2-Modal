"""
Pipeline Orchestrator — processes a single uploaded file end-to-end:

  1. PDF → JPEG conversion (if needed)
  2. Modal OCR → markdown
  3. GPT-4o-mini structured extraction → InvoiceData JSON
  4. VAT processor
  5. GL classifier (per-line)
  6. Sheets logger (one row per line item)

Returns a result dict with extracted data + token cost.
"""
import io
import os
import uuid
import traceback
from typing import Optional

from test_bench.services.modal_client import call_modal_ocr
from test_bench.services.structured_extractor import StructuredExtractor, InvoiceData
from test_bench.services.vat_processor import process_vat
from test_bench.services.gl_classifier import GLClassifier, FALLBACK_GL_NAME
from test_bench.services.sheets_service import GoogleSheetsService


def _pdf_to_jpeg_bytes(pdf_bytes: bytes) -> bytes:
    """
    Convert the first page of a PDF to JPEG bytes.
    Requires PyMuPDF (fitz) which is much easier to install on Windows than poppler.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]

    # Render at 200 DPI for good OCR quality
    zoom = 200 / 72  # 72 is default DPI
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    img_bytes = pix.tobytes("jpeg")
    doc.close()
    print(f"[Pipeline] PDF page 1 → JPEG ({len(img_bytes):,} bytes)")
    return img_bytes


async def process_single_file(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    extractor: StructuredExtractor,
    sheets_service: Optional[GoogleSheetsService] = None,
    gl_classifier: Optional[GLClassifier] = None,
) -> dict:
    """
    Full pipeline for one uploaded file.

    Returns a dict:
      {
        "filename": str,
        "status": "success" | "error",
        "invoice_data": { ... } | None,
        "token_usage": { ... } | None,
        "error": str | None,
        "sheets_logged": bool,
      }
    """
    result = {
        "filename": filename,
        "status": "error",
        "invoice_data": None,
        "token_usage": None,
        "error": None,
        "sheets_logged": False,
    }

    try:
        # ── Step 1: Convert PDF → JPEG if needed ─────────────────────────
        is_pdf = (
            content_type == "application/pdf"
            or filename.lower().endswith(".pdf")
        )
        if is_pdf:
            image_bytes = _pdf_to_jpeg_bytes(file_bytes)
        else:
            image_bytes = file_bytes

        # ── Step 2: Modal OCR ────────────────────────────────────────────
        markdown_text = await call_modal_ocr(image_bytes)

        # ── Step 3: Structured extraction (GPT-4o-mini) ──────────────────
        invoice_data, token_usage = await extractor.extract_from_text(markdown_text)
        inv_dict = invoice_data.model_dump()
        result["token_usage"] = token_usage

        # ── Step 4: VAT processor ────────────────────────────────────────
        inv_dict = process_vat(inv_dict)

        # ── Step 5: GL classifier (per line) ─────────────────────────────
        if gl_classifier:
            for item in inv_dict.get("line_items", []):
                desc = item.get("description", "")
                gl_name, matched_kw = gl_classifier.classify_line(desc)
                if gl_name:
                    item["gl_code"] = gl_name
                    item["gl_matched_keyword"] = matched_kw
                else:
                    # Use extractor's suggestion or fallback
                    if not item.get("gl_code"):
                        item["gl_code"] = inv_dict.get("gl_code_suggested") or FALLBACK_GL_NAME
                    gl_classifier.log_pending_review_line(item, inv_dict)

        # ── Step 6: Sheets logger ────────────────────────────────────────
        if sheets_service:
            file_id = f"upload-{uuid.uuid4().hex[:8]}"
            logged = sheets_service.append_invoice(inv_dict, file_id, filename)
            result["sheets_logged"] = logged
            if logged:
                print(f"[Pipeline] ✓ Logged to Sheets: {filename}")
            else:
                print(f"[Pipeline] ✗ Failed to log to Sheets: {filename}")

        # ── Done ─────────────────────────────────────────────────────────
        # Remove internal keys before returning
        for key in ("raw_response", "_pre_tax_amount", "_tax_portion"):
            inv_dict.pop(key, None)
        for item in inv_dict.get("line_items", []):
            item.pop("_pre_tax_amount", None)
            item.pop("_tax_portion", None)

        result["invoice_data"] = inv_dict
        result["status"] = "success"
        print(f"[Pipeline] ✓ Completed: {filename}")

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        print(f"[Pipeline] ✗ Failed: {filename} — {result['error']}")
        traceback.print_exc()

    return result
