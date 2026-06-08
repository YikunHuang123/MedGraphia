"""
Domain-specific medical knowledge: search queries, drug lists, etc.
Centralized here to keep the pipeline code clean and focused on orchestration.
"""

from __future__ import annotations

DOMAIN_QUERIES: dict[str, str] = {
    "t2dm": (
        "type 2 diabetes mellitus[MeSH] AND "
        "(drug therapy[MeSH] OR treatment[MeSH]) AND English[Language]"
    ),
    "cardiovascular": (
        "cardiovascular diseases[MeSH] AND drug therapy[MeSH] AND English[Language]"
    ),
    "oncology": "neoplasms[MeSH] AND drug therapy[MeSH] AND English[Language]",
    "hypertension": ("hypertension[MeSH] AND antihypertensive agents[MeSH] AND English[Language]"),
}

DOMAIN_DRUGS: dict[str, list[str]] = {
    "t2dm": ["metformin", "insulin", "sitagliptin", "empagliflozin", "liraglutide"],
    "cardiovascular": ["warfarin", "aspirin", "atorvastatin", "lisinopril", "metoprolol"],
    "hypertension": ["amlodipine", "lisinopril", "losartan", "hydrochlorothiazide"],
}
