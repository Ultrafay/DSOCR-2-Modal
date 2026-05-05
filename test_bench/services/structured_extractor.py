"""
Structured Extractor — takes OCR markdown text and uses GPT-4o-mini
to extract structured invoice data (JSON).

Adapted from QB-Pipeline-ATMT's openai_extractor.py:
  - Input is markdown text (not images)
  - Uses gpt-4o-mini for cost efficiency
  - Returns token usage for cost tracking
  - Same InvoiceData schema, same UAE VAT / GL classification rules
"""
import json
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import os

from test_bench.services.gl_reference_data import build_gl_prompt_section


class LineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None
    tax_percentage: Optional[float] = None  # 0, 5, or null
    tax_code: Optional[str] = None  # SR, EX, ZR, RC, IG
    gl_code: Optional[str] = None  # GL Account Name for this line


class InvoiceData(BaseModel):
    date: Optional[str] = None
    supplier_name: Optional[str] = None
    supplier_trn: Optional[str] = None
    supplier_address: Optional[str] = None
    invoice_number: Optional[str] = None
    description: Optional[str] = None
    due_date: Optional[str] = None
    credit_terms: Optional[str] = None
    bill_to: Optional[str] = None
    bill_to_trn: Optional[str] = None
    purchase_location: Optional[str] = None
    gl_code_suggested: Optional[str] = None
    exclusive_amount: Optional[float] = None
    vat_amount: Optional[float] = None
    invoice_tax_amount: Optional[float] = None
    invoice_tax_percentage: Optional[float] = None
    total_amount: Optional[float] = None
    currency: str = "AED"
    line_items: List[LineItem] = []
    extraction_confidence: str = "medium"
    extraction_method: str = "gpt4o_mini_text"
    notes: Optional[str] = None
    raw_response: Optional[str] = None


# ── GPT-4o-mini pricing (per 1M tokens, as of 2024-07) ───────────────────────
INPUT_COST_PER_M  = 0.15   # $0.15 / 1M input tokens
OUTPUT_COST_PER_M = 0.60   # $0.60 / 1M output tokens


class StructuredExtractor:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

        # GL prompt section (uses hardcoded reference data)
        self._gl_prompt: str = build_gl_prompt_section()

    def set_chart_of_accounts(self, account_names: List[str]) -> None:
        """Re-build the GL prompt section with a live chart of accounts."""
        self._gl_prompt = build_gl_prompt_section(chart_of_accounts=account_names)
        print(f"[Extractor] GL prompt updated with {len(account_names)} accounts")

    # ── System Prompt ────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        return f"""You are an expert invoice data extraction system for a UAE-based company.

Analyze the OCR'd markdown text of an invoice and extract ALL relevant data. The invoice may contain English, Arabic, or both languages.

CRITICAL INSTRUCTIONS:
1. EXTRACT EVERY SINGLE LINE ITEM. Do not group them. Do not summarize them.
2. For Arabic company names, provide both Arabic and English transliteration if visible
3. All amounts must be numeric only (no currency symbols, no commas)
4. Dates must be in YYYY-MM-DD format
5. If a field is not visible or unclear, use null
6. TRN (Tax Registration Number) in UAE is 15 digits starting with "100"
7. Extract the exact currency code (e.g., AED, USD, EUR). Default to USD if none is detected.
8. Extract the supplier's full address as a single string.
9. For EACH line item, extract the VAT/tax percentage applied (0, 5, or null if not shown).
10. For EACH line item, assign a tax_code based on the TAX CODE CLASSIFICATION RULES below.
11. For EACH line item, assign a gl_code based on the GL CODE CLASSIFICATION section below.
12. Extract the total tax/VAT amount shown on the invoice into invoice_tax_amount. This is the
    total tax the invoice charges (could be UAE VAT, US sales tax, UK VAT, etc.). If no tax is
    shown, set to 0.
13. Extract the explicit tax percentage applied to the entire invoice into invoice_tax_percentage.
    If the invoice clearly says "VAT 8%" or "Tax 15%", use that number. If it is mixed or not
    explicitly stated, use null.

IDENTIFY CORRECTLY:
- SUPPLIER = The company SENDING the invoice
- BILL TO = The company RECEIVING the invoice

TAX CODE CLASSIFICATION RULES:
Assign one of these codes to EACH line item based on supplier location and item type:

  "SR" — Standard Rated (5% VAT). Normal taxable goods and services from a UAE-based supplier.
  "EX" — Exempt (0%). Government fees, visa charges, labour/immigration fees, fines,
          bank charges, insurance premiums passed through at cost. Use for any
          regulatory or government-imposed charge.
  "ZR" — Zero Rated (0%). Exports, international transport, certain education and
          healthcare supplies. Rare on domestic purchase invoices.
  "RC" — Reverse Charge (0%). ANY supplier located OUTSIDE the UAE and outside the GCC.
  "IG" — Intra GCC (0%). Supplier is in a GCC country (Saudi Arabia, Bahrain, Oman,
          Kuwait, Qatar) but NOT UAE VAT-registered.

DECISION LOGIC:
  Step 1: Determine supplier location from their address and TRN.
    - If supplier has a UAE TRN (15 digits starting with 100) or address contains
      a UAE city/emirate → supplier is UAE-based → go to Step 2.
    - If supplier address mentions Saudi Arabia, Bahrain, Oman, Kuwait, or Qatar
      → use "IG" for ALL lines.
    - If supplier is outside UAE and outside GCC → use "RC" for ALL lines.
  Step 2 (UAE suppliers only): Classify EACH line individually.
    - Government/regulatory fees, visa fees, labour fees, fines, stamps,
      attestation charges, municipality fees, permit fees, typing fees,
      medical test fees (for visa), bank charges, insurance premiums
      passed at cost → "EX"
    - Normal taxable goods and services (consulting, supplies, equipment,
      maintenance, medical supplies, marketing, software, professional services)
      → "SR"
    - Exports, international freight/transport, certain education/healthcare
      supplies designated zero‑rated → "ZR"
    - When unsure between "EX" and "ZR", default to "EX".
    - When unsure between "EX" and "SR", look at the tax column on the invoice:
      if the line shows 5% tax, use "SR"; if it shows 0% or no tax, use "EX".

{self._gl_prompt}

EXTRACT INTO THIS EXACT JSON STRUCTURE:

{{
  "date": "YYYY-MM-DD",
  "supplier_name": "Company issuing the invoice",
  "supplier_trn": "15-digit TRN or null",
  "supplier_address": "Full supplier address as a single string, or null",
  "purchase_location": "Name of the branch or location if mentioned (e.g., Dubai Marina, Head Office), or null",
  "invoice_number": "Invoice reference number",
  "description": "General description of invoice (optional)",
  "due_date": "YYYY-MM-DD or null",
  "credit_terms": "NET 30, Cheque, Immediate, etc.",
  "bill_to": "Customer name",
  "gl_code_suggested": "Primary GL category for the invoice overall (from keyword mapping or general knowledge)",
  "exclusive_amount": 0.00,
  "vat_amount": 0.00,
  "invoice_tax_amount": 0.00,
  "invoice_tax_percentage": null,
  "total_amount": 0.00,
  "currency": "AED (default to USD if unknown)",
  "line_items": [
    {{
      "description": "Professional consulting service",
      "quantity": 1.0,
      "unit_price": 1000.00,
      "amount": 1000.00,
      "tax_percentage": 5,
      "tax_code": "SR",
      "gl_code": "Legal & Professional Fees"
    }}
  ],
  "extraction_confidence": "high|medium|low",
  "notes": "Any issues found"
}}

IMPORTANT:
- If an item is free/bonus (FOC), set unit_price to 0 and amount to 0.
- Capture the EXACT item description for each line.
- The 'amount' in line_items should be the line total (Quantity * Unit Price) BEFORE tax.
- tax_percentage per line: use 5 for 5% VAT, 0 for zero-rated/exempt, null if not visible.
- tax_code per line: MUST be one of SR, EX, ZR, RC, IG. Follow the classification rules above.
- gl_code per line: MUST be a GL Account Name from the keyword mapping or chart of accounts.
- invoice_tax_amount: Extract the total tax amount shown on the invoice (e.g., "VAT 5%: 50.00"
  means invoice_tax_amount = 50.00). This includes foreign taxes (US sales tax, UK VAT, etc.).
  If no tax line is shown, set to 0.
- invoice_tax_percentage: Extract the exact percentage stated if present (e.g. 5, 8, 15).
- Ensure 'total_amount' matches the sum of line items + tax.

Return ONLY valid JSON. No markdown."""

    # ── Extract from markdown text ────────────────────────────────────────────

    async def extract_from_text(self, markdown_text: str) -> Tuple[InvoiceData, dict]:
        """
        Extract invoice data from OCR'd markdown text using GPT-4o-mini.

        Returns:
            (InvoiceData, usage_dict) where usage_dict contains
            prompt_tokens, completion_tokens, total_tokens, estimated_cost_usd.
        """
        print(f"[Extractor] Sending {len(markdown_text)} chars of markdown to {self.model}…")

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {
                    "role": "user",
                    "content": (
                        "Here is the OCR'd markdown of an invoice. "
                        "Extract all structured fields.\n\n"
                        f"---\n{markdown_text}\n---"
                    ),
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=4096,
            temperature=0.1,
        )

        # ── Token usage tracking ──────────────────────────────────────────────
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost = (
            (prompt_tokens / 1_000_000) * INPUT_COST_PER_M
            + (completion_tokens / 1_000_000) * OUTPUT_COST_PER_M
        )

        usage_dict = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 6),
        }
        print(
            f"[Extractor] Tokens: {prompt_tokens} in / {completion_tokens} out "
            f"= {total_tokens} total — est. ${estimated_cost:.4f}"
        )

        invoice = self._parse_response(response.choices[0].message.content)
        return invoice, usage_dict

    # ── Response parser ───────────────────────────────────────────────────────

    def _parse_response(self, response_text: str) -> InvoiceData:
        """Parse the JSON response from OpenAI into an InvoiceData object."""
        try:
            # Clean up markdown formatting if present
            clean_text = response_text
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0]
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0]
            clean_text = clean_text.strip()

            data = json.loads(clean_text)

            # Default missing/unknown currency to USD
            extracted_currency = str(data.get("currency", "") or "").upper().strip()
            # Keep only the first word in case model returns "AED (default...)" style values
            extracted_currency = extracted_currency.split()[0] if extracted_currency else ""
            # Validate it's a proper 3-letter ISO currency code; fall back to USD otherwise
            if not extracted_currency or len(extracted_currency) != 3 or not extracted_currency.isalpha():
                data["currency"] = "USD"
            else:
                data["currency"] = extracted_currency

            invoice = InvoiceData(**data)
            invoice.raw_response = clean_text
            return invoice
        except Exception as e:
            print(f"Error parsing OpenAI response: {e}")
            print(f"Raw response: {response_text}")
            raise ValueError(f"Failed to parse JSON from OpenAI response: {e}")
