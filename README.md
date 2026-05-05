# DSOCR-2 Test Bench

FastAPI test bench that processes invoices through:

1. **Modal DeepSeek-OCR** — converts invoice images to markdown via the GPU endpoint
2. **GPT-4o-mini** — structured extraction with UAE VAT/GL classification rules
3. **VAT Processor** — validates per-line tax codes (SR/EX/ZR/RC/IG)
4. **GL Classifier** — keyword-based GL account matching (optional sheet-driven)
5. **Google Sheets** — logs one row per line item

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in env vars
copy .env.example .env
# Edit .env with your actual keys

# 3. Run
python -m test_bench.app
```

Open [http://localhost:8000](http://localhost:8000) → drag-drop invoices → view results.

## Architecture

```
test_bench/
  app.py                    # FastAPI server + routes
  pipeline.py               # Per-file orchestrator
  services/
    modal_client.py         # Async HTTP to Modal OCR
    structured_extractor.py # GPT-4o-mini text→JSON (adapted from QB-Pipeline)
    vat_processor.py        # UAE VAT validation (from QB-Pipeline)
    gl_classifier.py        # Sheet-driven GL matching (from QB-Pipeline)
    gl_reference_data.py    # Hardcoded GL keyword rules (from QB-Pipeline)
    sheets_service.py       # Google Sheets logger (QBO columns removed)
  utils/
    credentials_helper.py   # Google service account resolver
  static/
    index.html              # Drag-drop upload UI
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
| `MODAL_OCR_URL` | ✅ | Modal endpoint URL |
| `MODAL_PROXY_TOKEN_ID` | ✅ | Modal proxy auth token ID |
| `MODAL_PROXY_TOKEN_SECRET` | ✅ | Modal proxy auth token secret |
| `GOOGLE_SHEET_ID` | ⬜ | Google Sheet for logging (optional) |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | ⬜ | Path to service account JSON |
| `GOOGLE_SERVICE_ACCOUNT_CONTENT` | ⬜ | Service account JSON as string |
| `GL_MAPPING_SHEET_ID` | ⬜ | GL mapping sheet ID (optional) |

## API Endpoints

- `GET /` — Upload UI
- `POST /process` — Multi-file upload (form-data `files`)
- `GET /health` — Health check

## Supported File Types

PDF, JPG, JPEG, PNG, WebP
