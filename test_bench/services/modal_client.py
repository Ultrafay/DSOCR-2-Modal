"""
Modal OCR Client — sends base64-encoded images to the Modal DeepSeek-OCR endpoint
and returns the extracted markdown text.
"""
import base64
import httpx
import os

MODAL_OCR_URL = os.getenv(
    "MODAL_OCR_URL",
    "https://ultrafay--deepseek-ocr-ocrmodel-ocr.modal.run"
)
MODAL_PROXY_TOKEN_ID = os.getenv("MODAL_PROXY_TOKEN_ID", "")
MODAL_PROXY_TOKEN_SECRET = os.getenv("MODAL_PROXY_TOKEN_SECRET", "")


async def call_modal_ocr(image_bytes: bytes) -> str:
    """
    Send an image (bytes) to the Modal OCR endpoint.

    Returns the extracted markdown text.
    Raises on HTTP errors or empty responses.
    """
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Modal-Key": MODAL_PROXY_TOKEN_ID,
        "Modal-Secret": MODAL_PROXY_TOKEN_SECRET,
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        print(f"[Modal] Sending {len(image_bytes):,} bytes to OCR endpoint…")
        resp = await client.post(
            MODAL_OCR_URL,
            headers=headers,
            json={"image_base64": image_b64},
        )
        resp.raise_for_status()

    data = resp.json()
    text = data.get("text", "")
    if not text:
        raise ValueError(
            f"Modal OCR returned empty text. "
            f"Debug: result_type={data.get('debug_result_type')}, "
            f"files={data.get('debug_output_files')}"
        )

    print(f"[Modal] OCR returned {len(text)} characters of markdown")
    return text
