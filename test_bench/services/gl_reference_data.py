from typing import List

GL_KEYWORD_MAPPING = [
    {
        "keywords": ["facebook", "instagram", "meta", "campaign", "advertising", "ad spend", "sponsored", "boosted", "awareness", "social media ad", "digital ad"],
        "gl_account": "Advertising",
        "priority": 1,
        "notes": "Meta/Facebook/Instagram ad invoices"
    },
    {
        "keywords": ["marketing", "promotion", "expo", "franchise expo", "exhibition", "branding", "media group", "banner", "display", "lead generation"],
        "gl_account": "Marketing",
        "priority": 2,
        "notes": "Marketing and promotional spend"
    },
    {
        "keywords": ["injection", "sachet", "mg", "pharmaceutical", "medicine", "drug", "supplement", "ampoule", "vial", "syringe", "profhilo", "lisathyone", "physiosens", "structura", "treatment product", "serum", "botox", "filler"],
        "gl_account": "COGS",
        "priority": 1,
        "notes": "Medical/pharma products for resale or treatment"
    },
    {
        "keywords": ["scrub", "uniform", "workwear", "apparel", "clothing", "shirt", "pants", "cargo", "embroidery", "logo", "staff wear", "cherokee"],
        "gl_account": "Job Materials",
        "priority": 1,
        "notes": "Staff uniforms and workwear"
    },
    {
        "keywords": ["massage bed", "treatment bed", "machine", "motor", "equipment", "device", "bed frame", "trolley", "roller", "paper roll holder", "silver fox"],
        "gl_account": "Equipment Rental",
        "priority": 1,
        "notes": "Medical/treatment equipment purchase"
    },
    {
        "keywords": ["furniture", "sofa", "chair", "table", "cabinet", "shelf", "desk", "reception", "interior", "decor", "fit-out", "fitout", "down payment furniture"],
        "gl_account": "Job Materials",
        "priority": 2,
        "notes": "Furniture and interior items"
    },
    {
        "keywords": ["visa", "labor", "labour", "offer letter", "submission", "work permit", "labor card", "labor contract", "clearance", "immigration", "residency"],
        "gl_account": "Legal & Professional Fees",
        "priority": 1,
        "notes": "Visa and labor document processing"
    },
    {
        "keywords": ["insurance", "health insurance", "medical insurance", "workers compensation", "liability insurance", "policy"],
        "gl_account": "Insurance",
        "priority": 1,
        "notes": "Insurance premiums"
    },
    {
        "keywords": ["accounting", "bookkeeping", "auditing", "audit", "lawyer", "legal", "compliance", "tax advisory", "professional fee", "consultation"],
        "gl_account": "Legal & Professional Fees",
        "priority": 2,
        "notes": "Accounting and legal professional services"
    },
    {
        "keywords": ["repair", "maintenance", "building repair", "computer repair", "equipment repair", "servicing", "fixing", "technical support"],
        "gl_account": "Maintenance and Repair",
        "priority": 1,
        "notes": "Repair and maintenance services"
    },
    {
        "keywords": ["subscription", "software", "saas", "license", "dues", "membership", "platform fee", "annual fee", "renewal"],
        "gl_account": "Dues & Subscriptions",
        "priority": 1,
        "notes": "Software subscriptions and memberships"
    },
    {
        "keywords": ["bank charge", "bank fee", "transfer fee", "transaction fee", "wire fee", "service charge"],
        "gl_account": "Bank Charges",
        "priority": 1,
        "notes": "Bank and transaction charges"
    },
    {
        "keywords": ["cleanser", "tonic", "skincare", "cream", "lotion", "gel", "moisturizer", "aha", "bha", "purifying", "aesthetic product"],
        "gl_account": "COGS",
        "priority": 2,
        "notes": "Skincare and aesthetic products"
    },
    {
        "keywords": ["fuel", "petrol", "diesel", "gasoline"],
        "gl_account": "Automobile",
        "priority": 1,
        "notes": "Vehicle fuel"
    },
    {
        "keywords": ["office fee", "printing", "stationery", "supplies", "office supplies", "paper"],
        "gl_account": "Job Expenses",
        "priority": 1,
        "notes": "Office and admin supplies"
    }
]

def build_gl_prompt_section(chart_of_accounts: List[str] = None) -> str:
    """
    Format the keyword mapping rules (and optionally the exact QBO Chart of Accounts)
    into a structured prompt section for GPT-4o.
    """
    prompt = "GL CODE CLASSIFICATION:\n"
    prompt += "Classify EACH line item's GL category by scanning the description against these keyword mapping rules.\n"
    prompt += "Use Priority 1 matches first. If multiple match, use the one with most overlap.\n"
    prompt += "If no keywords match, use general accounting knowledge to classify by expense nature.\n\n"
    prompt += "KEYWORD MAPPING (Format: GL Name: keywords...):\n"

    for rule in sorted(GL_KEYWORD_MAPPING, key=lambda x: x["priority"]):
        keywords_joined = ", ".join(rule["keywords"])
        prompt += f"- {rule['gl_account']}: {keywords_joined}\n"

    if chart_of_accounts and len(chart_of_accounts) > 0:
        prompt += "\nVALID QBO ACCOUNTS AVAILABLE (Must fall back to one of these if mapping rule fails):\n"
        prompt += ", ".join(chart_of_accounts) + "\n"
    else:
        prompt += "\nNOTE: If the matched GL account doesn't perfectly match a known QBO account, the system will fall back to Uncategorised Expense.\n"

    return prompt
