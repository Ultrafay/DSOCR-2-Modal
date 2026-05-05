"""
VAT Processor — validates and finalises per-line tax codes.

The AI extractor assigns a `tax_code` (SR / EX / ZR / RC / IG) to each
line item.  This module:

  1. Determines the supplier location category (UAE / GCC / Foreign) from
     TRN + address — used as a safety net.
  2. Validates each line's `tax_code` against the location. If the code is
     missing or inconsistent, it assigns a sensible fallback and flags for
     review.
  3. Maps shorthand codes to full names.
  4. For non-UAE invoices with foreign tax, distributes the tax into line
     amounts and sets a flag for TaxInclusive mode.
  5. Computes implied tax totals from per-line codes and compares them to
     the invoice-level tax amount. Flags mismatches > threshold.
"""
import re
from typing import List

# ── Shorthand → full TaxCode name ─────────────────────────────────────
TAX_CODE_MAP = {
    "SR": "SR Standard Rated",
    "EX": "EX Exempt",
    "ZR": "ZR Zero Rated",
    "RC": "RC Reverse Charge",
    "IG": "IG Intra GCC",
}

# Tax rates implied by each code (used for mismatch validation)
TAX_RATE_MAP = {
    "SR": 0.05,
    "EX": 0.0,
    "ZR": 0.0,
    "RC": 0.0,
    "IG": 0.0,
}

VALID_CODES = set(TAX_CODE_MAP.keys())

# Mismatch threshold (in invoice currency units)
_MISMATCH_THRESHOLD = 1.0

# ── Location keywords ─────────────────────────────────────────────────
_UAE_KEYWORDS = [
    "uae", "united arab emirates",
    "dubai", "abu dhabi", "sharjah", "ajman",
    "fujairah", "ras al khaimah", "umm al quwain",
]
_GCC_KEYWORDS = [
    "saudi arabia", "ksa",
    "oman",
    "bahrain",
    "kuwait",
    "qatar",
]


def _is_uae_trn(trn: str) -> bool:
    if not trn:
        return False
    digits = re.sub(r"\D", "", str(trn))
    return len(digits) == 15 and digits.startswith("100")


def get_location_category(invoice_data: dict) -> str:
    """Returns 'UAE', 'GCC', or 'Foreign' based on TRN / address heuristics."""
    trn = str(invoice_data.get("supplier_trn", "") or "").strip()
    address = str(invoice_data.get("supplier_address", "") or "").strip().lower()

    if _is_uae_trn(trn):
        return "UAE"

    for kw in _UAE_KEYWORDS:
        if kw in address:
            return "UAE"

    for kw in _GCC_KEYWORDS:
        if kw in address:
            return "GCC"

    return "Foreign"

# ── Per-line validation helpers ───────────────────────────────────────
def _valid_codes_for_location(category: str) -> set:
    """Return the set of tax codes that are valid for a given location."""
    if category == "UAE":
        return {"SR", "EX", "ZR"}
    elif category == "GCC":
        return {"IG"}
    else:  # Foreign
        return {"RC"}


def _fallback_code_for_location(category: str, tax_pct, has_invoice_vat: bool) -> str:
    """
    Pick a sensible fallback when the extractor didn't provide a tax_code
    or provided an invalid one.
    """
    if category == "GCC":
        return "IG"
    if category == "Foreign":
        return "RC"
    # UAE — use tax_percentage hint if available
    if tax_pct is not None:
        pct = float(tax_pct)
        if pct == 5.0:
            return "SR"
        if pct == 0.0:
            return "EX"
    # No percentage hint — guess from invoice-level VAT
    return "SR" if has_invoice_vat else "EX"

def _determine_rcm_tax(invoice_data: dict, subtotal: float) -> tuple:
    """
    Determine the actual RCM tax percentage and absolute amount.

    Priority order:
      1. Explicit invoice_tax_percentage (e.g., "Sales Tax 8.25%" → 8.25)
      2. Calculated from invoice_tax_amount / subtotal
      3. Fallback to 0.0 (no RCM tax)

    Returns (percentage, amount)
    """
    tax_pct = invoice_data.get("invoice_tax_percentage")
    tax_amt = float(invoice_data.get("invoice_tax_amount", 0.0) or 0.0)

    if tax_pct is not None and tax_pct != "":
        tax_pct = float(tax_pct)
        # If the invoice gave us the percentage but not the amount, derive it
        if tax_amt <= 0 and subtotal > 0:
            tax_amt = round(subtotal * (tax_pct / 100.0), 2)
        print(f"[VAT] RCM tax determined from explicit invoice_tax_percentage: {tax_pct}% → amount={tax_amt}")
        return tax_pct, tax_amt

    if tax_amt > 0 and subtotal > 0:
        calculated_pct = round((tax_amt / subtotal) * 100.0, 2)
        print(f"[VAT] RCM tax calculated from invoice_tax_amount/subtotal: {tax_amt}/{subtotal} = {calculated_pct}%")
        return calculated_pct, tax_amt

    print("[VAT] No RCM tax information found — defaulting to 0.0%")
    return 0.0, 0.0


def _distribute_foreign_tax(invoice_data: dict, line_items: List[dict]) -> List[dict]:
    """
    For non-UAE invoices, apply the foreign tax PROPORTIONALLY to each line item.

    Each line receives: gross_amount = pre_tax_amount × (1 + tax_rate)

    This ensures:
      - Larger lines absorb more tax than smaller ones (correct)
      - The sum of grossed-up line amounts ≈ invoice total_amount
      - The RCM journal entry uses the same exact total tax amount

    After distribution, each line's ``amount`` is the gross (tax-inclusive) value.
    The original pre-tax value is preserved in ``_pre_tax_amount`` for audit.
    """
    subtotal = sum(float(item.get("amount", 0.0) or 0.0) for item in line_items)

    rcm_pct, rcm_amt = _determine_rcm_tax(invoice_data, subtotal)

    # Store so downstream can use the exact amount
    invoice_data["rcm_tax_percentage"] = rcm_pct
    invoice_data["rcm_tax_amount"] = rcm_amt

    if rcm_amt <= 0:
        print("[VAT] rcm_tax_amount = 0 — no foreign tax to distribute")
        return line_items

    tax_rate = rcm_pct / 100.0
    valid_lines = [(i, float(item.get("amount", 0.0) or 0.0))
                   for i, item in enumerate(line_items)
                   if float(item.get("amount", 0.0) or 0.0) > 0]
    num_valid = len(valid_lines)

    if num_valid == 0:
        return line_items

    print(
        f"[VAT] Proportional foreign-tax distribution: rate={rcm_pct}%, "
        f"total_tax={rcm_amt}, subtotal={subtotal}, lines={num_valid}"
    )

    # Build grossed-up list using proportional allocation
    grossed_up = list(line_items)  # shallow copy of list
    distributed_tax = 0.0

    for rank, (idx, item_amount) in enumerate(valid_lines):
        item = dict(grossed_up[idx])  # copy the dict so we don't mutate in place

        if rank == num_valid - 1:
            # Last line: absorb any rounding remainder so totals reconcile exactly
            tax_portion = round(rcm_amt - distributed_tax, 2)
        else:
            tax_portion = round(item_amount * tax_rate, 2)

        new_amount = round(item_amount + tax_portion, 2)
        distributed_tax += tax_portion

        print(
            f"[VAT]   Line {idx + 1}: pre-tax={item_amount} "
            f"+ tax={tax_portion} ({rcm_pct}%) = gross={new_amount}"
        )

        item["amount"] = new_amount
        item["_pre_tax_amount"] = item_amount  # preserved for audit / mismatch check
        item["_tax_portion"] = tax_portion
        grossed_up[idx] = item

    print(f"[VAT] Total distributed tax: {round(distributed_tax, 2)} (target: {rcm_amt})")
    return grossed_up

def process_vat(invoice_data: dict) -> dict:
    """
    Validate per-line tax codes, assign fallbacks where missing, map to
    full names, handle foreign tax distribution, and run tax-total
    mismatch check.
    """
    category = get_location_category(invoice_data)
    vat_amount = float(invoice_data.get("vat_amount", 0.0) or 0.0)
    invoice_tax = float(invoice_data.get("invoice_tax_amount", 0.0) or 0.0)
    line_items: List[dict] = invoice_data.get("line_items", []) or []
    has_invoice_vat = vat_amount > 0 or invoice_tax > 0

    print(f"[VAT] Supplier Location: {category} — VAT: {vat_amount}, Invoice Tax: {invoice_tax}, Lines: {len(line_items)}")

    invoice_data["supplier_location_category"] = category
    valid_codes = _valid_codes_for_location(category)
    review_messages: List[str] = []

    # ── Validate / assign per-line codes ──────────────────────────────────
    for idx, item in enumerate(line_items, start=1):
        raw_code = str(item.get("tax_code", "") or "").upper().strip()

        if raw_code in VALID_CODES:
            # Code is syntactically valid — check it fits the location
            if raw_code not in valid_codes:
                # Mismatch: e.g. extractor said "SR" for a Foreign vendor
                fallback = _fallback_code_for_location(
                    category, item.get("tax_percentage"), has_invoice_vat
                )
                review_messages.append(
                    f"Line {idx}: tax_code '{raw_code}' invalid for {category} vendor, "
                    f"overridden to '{fallback}'"
                )
                raw_code = fallback
        else:
            # Missing or unrecognised code — assign fallback
            fallback = _fallback_code_for_location(
                category, item.get("tax_percentage"), has_invoice_vat
            )
            if raw_code:
                review_messages.append(
                    f"Line {idx}: unrecognised tax_code '{raw_code}', "
                    f"defaulted to '{fallback}'"
                )
            raw_code = fallback

        # Write the validated shorthand back and the full name
        item["tax_code"] = raw_code
        item["qbo_tax_code"] = TAX_CODE_MAP[raw_code]

    # ── Foreign tax distribution (non-UAE only) ───────────────────────────
    if category in ("GCC", "Foreign"):
        line_items = _distribute_foreign_tax(invoice_data, line_items)
        if invoice_data.get("rcm_tax_amount", 0.0) > 0:
            invoice_data["tax_inclusive"] = True
            print(f"[VAT] Foreign tax distributed. Bill will use TaxInclusive mode.")
        else:
            invoice_data["tax_inclusive"] = False
    else:
        invoice_data["tax_inclusive"] = False

    # ── Tax mismatch validation ───────────────────────────────────────────
    # Use invoice_tax_amount (the actual tax on the invoice) for comparison.
    # Fall back to vat_amount if invoice_tax_amount is not set.
    reference_tax = invoice_tax if invoice_tax > 0 else vat_amount

    if category == "UAE":
        # UAE: sum of implied tax from per-line codes
        implied_tax = 0.0
        for item in line_items:
            item_amount = float(item.get("_pre_tax_amount", item.get("amount", 0.0)) or 0.0)
            rate = TAX_RATE_MAP.get(item.get("tax_code", ""), 0.0)
            implied_tax += item_amount * rate

        implied_tax = round(implied_tax, 2)
        diff = abs(implied_tax - reference_tax)

        if diff > _MISMATCH_THRESHOLD:
            msg = (
                f"TAX MISMATCH: computed tax = {implied_tax}, "
                f"invoice tax = {reference_tax}, diff = {diff:.2f} — review line tax codes"
            )
            review_messages.append(msg)
            print(f"[VAT] {msg}")
    else:
        # Non-UAE: verify grossed-up total equals invoice total
        total_amount = float(invoice_data.get("total_amount", 0.0) or 0.0)
        # Use rcm_tax_amount. Only check if there was tax.
        assigned_tax = invoice_data.get("rcm_tax_amount", 0.0)
        if total_amount > 0 and assigned_tax > 0:
            grossed_sum = sum(float(item.get("amount", 0.0) or 0.0) for item in line_items)
            grossed_sum = round(grossed_sum, 2)
            diff = abs(grossed_sum - total_amount)

            if diff > _MISMATCH_THRESHOLD:
                msg = (
                    f"TAX MISMATCH: grossed-up line total = {grossed_sum}, "
                    f"invoice total = {total_amount}, diff = {diff:.2f} — review amounts"
                )
                review_messages.append(msg)
                print(f"[VAT] {msg}")

    # ── Assemble review memo ──────────────────────────────────────────────
    if review_messages:
        combined = " | ".join(review_messages)
        existing_memo = invoice_data.get("manual_review_memo", "") or ""
        invoice_data["manual_review_memo"] = (
            f"{existing_memo} | {combined}" if existing_memo else combined
        )
        print(f"[VAT] Review flagged: {combined}")

    # ── Metadata for downstream consumers ─────────────────────────────────
    invoice_data["line_items"] = line_items
    invoice_data["is_uae_invoice"] = (category == "UAE")

    # For GCC / Foreign WITHOUT foreign tax distribution, zero out vat_amount
    if category in ("GCC", "Foreign") and not invoice_data.get("tax_inclusive"):
        invoice_data["vat_amount"] = 0.0

    return invoice_data
