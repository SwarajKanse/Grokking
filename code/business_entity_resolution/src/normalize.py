"""
Shared text normalization module for Business Entity Resolution.
Used consistently across candidate blocking, pairwise feature engineering,
and inference.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, List, Optional


# ---------------------------------------------------------------------------
# Dictionaries for Name Normalization
# ---------------------------------------------------------------------------

LEGAL_SUFFIX_MAP: dict[str, str] = {
    r"\bpvt\b|\bpte\b|\bpvtltd\b": "private",
    r"\bltd\b": "limited",
    r"\binc\b|\bincorp\b": "incorporated",
    r"\bcorp\b": "corporation",
    r"\bco\b": "company",
    r"\bllc\b": "llc",
    r"\bllp\b": "llp",
    r"\blp\b": "lp",
    r"\bpllc\b": "pllc",
    # French business entity types (for unseen test set)
    r"\bsarl\b": "sarl",
    r"\bsas\b": "sas",
    r"\bsasu\b": "sasu",
    r"\bsa\b": "sa",
    r"\beurl\b": "eurl",
    r"\bsnc\b": "snc",
    r"\bsci\b": "sci",
    r"\bgmbh\b": "gmbh",
    r"\bcie\b": "company",
}

# Generic business term abbreviation expansions (standardized in norm, NOT stripped in clean_norm)
GENERIC_ABBREV_MAP: dict[str, str] = {
    r"\bent\b": "enterprises",
    r"\bind\b": "industries",
    r"\bintl\b": "international",
    r"\bmfg\b": "manufacturing",
    r"\bsvcs\b": "services",
    r"\btech\b": "technologies",
    r"\bassn\b|\bassoc\b": "associates",
    r"\bdept\b": "department",
    r"\bgrp\b": "group",
    r"\bsoln\b|\bsolns\b": "solutions",
}

# Real legal-entity-type suffix tokens ONLY (stripped in clean_norm)
# Generic descriptors (Services, Industries, Solutions, Enterprises) are strictly excluded
# to prevent false merges between distinct businesses sharing a root name.
LEGAL_SUFFIX_TOKENS: set[str] = {
    "private",
    "limited",
    "incorporated",
    "corporation",
    "company",
    "llc",
    "llp",
    "lp",
    "pllc",
    "sarl",
    "sas",
    "sasu",
    "sa",
    "eurl",
    "snc",
    "sci",
    "gmbh",
    # Hindi / Devanagari transliterated suffixes
    "प्राइवेट",
    "लिमिटेड",
    "कंपनी",
    "प्रा",
    "लि",
    # Unexpanded short forms if any
    "pvt",
    "ltd",
    "inc",
    "corp",
    "co",
}

# Multi-word legal suffix phrases
LEGAL_SUFFIX_PHRASES: list[re.Pattern] = [
    re.compile(r"\bprivate\s+limited\b", re.IGNORECASE),
    re.compile(r"\blimited\s+liability\s+company\b", re.IGNORECASE),
    re.compile(r"\blimited\s+liability\s+partnership\b", re.IGNORECASE),
    re.compile(r"\blimited\s+partnership\b", re.IGNORECASE),
    re.compile(r"प्राइवेट\s+लिमिटेड", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Dictionaries for Address Normalization
# ---------------------------------------------------------------------------

ADDR_ABBREV_MAP: dict[str, str] = {
    # Thoroughfare types
    r"\brd\b": "road",
    r"\bst\b|\bstr\b": "street",
    r"\bave\b|\bav\b": "avenue",
    r"\bblvd\b|\bbld\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bexpy\b": "expressway",
    r"\bfwy\b": "freeway",
    r"\bcir\b": "circle",
    r"\btrl\b": "trail",
    r"\bsq\b": "square",
    r"\bplz\b": "plaza",
    # French thoroughfares
    r"\bche\b": "chemin",
    r"\ball\b": "allee",
    r"\bimp\b": "impasse",
    r"\brte\b": "route",
    r"\bcrs\b": "cours",
    # Subunits & Buildings
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bbldg\b": "building",
    r"\bfl\b|\bflr\b": "floor",
    r"\brm\b": "room",
    r"\bsec\b": "sector",
    r"\bsoc\b": "society",
    r"\bcol\b": "colony",
    r"\bres\b": "residence",
    r"\bbat\b": "batiment",
    # Indian / Generic Landmarks & Administrative markers
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
    r"\bdist\b": "district",
    r"\btq\b": "taluk",
    r"\bpo\b": "post office",
    r"\bno\b": "number",
    r"\bh[\.\s/]*no\b": "house number",
    r"\bb[\./]h\b": "behind",
    r"\bc[\./]o\b": "care of",
    r"\bw[\./]o\b": "wife of",
    r"\bs[\./]o\b": "son of",
    r"\bd[\./]o\b": "daughter of",
    # Directionals
    r"\bn\b": "north",
    r"\bs\b": "south",
    r"\be\b": "east",
    r"\bw\b": "west",
    r"\bne\b": "northeast",
    r"\bnw\b": "northwest",
    r"\bse\b": "southeast",
    r"\bsw\b": "southwest",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NormalizedName:
    raw: str
    norm: str
    clean_norm: str
    tokens: list[str] = field(default_factory=list)
    has_name: bool = True
    legal_suffix: str = ""


@dataclass
class NormalizedAddress:
    raw: str
    norm: str
    tokens: list[str] = field(default_factory=list)
    has_address: bool = True
    postal_code: Optional[str] = None
    street_number: Optional[str] = None


@dataclass
class NormalizedRecord:
    entity_id: str
    country: str
    name: NormalizedName
    address: NormalizedAddress


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def is_missing_value(val: Any) -> bool:
    """Check if value is None, NaN, empty string, or sentinel placeholder."""
    if val is None:
        return True
    if isinstance(val, float):
        if math.isnan(val):
            return True
    s = str(val).strip().lower()
    return s in ("", "nan", "null", "none", "<na>")


def strip_punctuation_unicode(text: str) -> str:
    """
    Strips punctuation, symbols, and formatting while preserving all Unicode:
    - Letters (category L*): Latin, Devanagari, Tamil, etc.
    - Combining Marks (category M*): Vowel signs (matras), viramas, diacritics.
    - Numbers (category N*): Digits, decimals.

    Crucially avoids destroying non-Latin scripts (Devanagari, Tamil, etc.)
    where vowel signs and viramas are classified as Mn/Mc (not covered by standard \\w).
    """
    return "".join(
        ch if unicodedata.category(ch).startswith(("L", "M", "N")) else " "
        for ch in text
    )


def clean_name_tokens(norm_name: str) -> str:
    """
    Removes legal-suffix tokens and phrases wherever they occur in the token sequence
    (e.g., 'Private X Limited', 'X Inc Center', 'LLC X', 'X Pvt Ltd', 'X Private').
    Preserves core entity tokens. Falls back to norm_name if removing suffix tokens
    would leave the string completely empty.
    """
    if not norm_name:
        return ""

    s = norm_name
    for pat in LEGAL_SUFFIX_PHRASES:
        s = pat.sub(" ", s)

    tokens = s.split()
    remaining = [t for t in tokens if t.lower() not in LEGAL_SUFFIX_TOKENS]

    # Strip leading and trailing connector words ('and', '&')
    while remaining and remaining[0] in ("and", "&"):
        remaining = remaining[1:]
    while remaining and remaining[-1] in ("and", "&"):
        remaining = remaining[:-1]

    if remaining:
        return " ".join(remaining)
    return norm_name


def extract_postal_and_street(tokens: list[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Country-agnostic digit-group extractor:
    - Loose postal code: trailing digit group (length >= 3 digits) or token in latter half
    - Street number: leading digit group
    """
    if not tokens:
        return None, None

    all_digit_tokens = [(i, t) for i, t in enumerate(tokens) if t.isdigit()]
    if not all_digit_tokens:
        return None, None

    first_idx, first_val = all_digit_tokens[0]
    last_idx, last_val = all_digit_tokens[-1]

    street_num: Optional[str] = None
    postal_code: Optional[str] = None

    if len(all_digit_tokens) == 1:
        if first_idx <= 1:
            street_num = first_val
            # If the single number is >= 4 digits and appears near the end of a short address
            if len(first_val) >= 4 and first_idx >= len(tokens) - 2:
                postal_code = first_val
        elif first_idx >= len(tokens) / 2:
            if len(first_val) >= 3:
                postal_code = first_val
            else:
                street_num = first_val
        else:
            street_num = first_val
    else:
        if first_idx <= 2:
            street_num = first_val
        if len(last_val) >= 3 or last_idx >= len(tokens) - 2:
            postal_code = last_val

    return postal_code, street_num


# ---------------------------------------------------------------------------
# Core Normalization Functions
# ---------------------------------------------------------------------------

def normalize_name(raw_name: Any) -> NormalizedName:
    """
    Normalizes a business name:
    1. Check missing values explicitly.
    2. NFKC Unicode normalization + casefold.
    3. Expand standard symbols (& -> and, @ -> at, + -> and).
    4. Strip English possessive 's / ’s.
    5. Strip punctuation safely using Unicode character categories (L*, M*, N* preserved).
    6. Expand legal suffixes and abbreviations.
    7. Produce both full norm and sequence-wide suffix-stripped clean_norm.
    """
    if is_missing_value(raw_name):
        return NormalizedName(
            raw="",
            norm="",
            clean_norm="",
            tokens=[],
            has_name=False,
            legal_suffix="",
        )

    raw_str = str(raw_name).strip()
    s = unicodedata.normalize("NFKC", raw_str).casefold()

    # Expand common symbols before stripping punctuation
    s = re.sub(r"&", " and ", s)
    s = re.sub(r"@", " at ", s)
    s = re.sub(r"\+", " and ", s)

    # Remove possessive 's or trailing apostrophes before punctuation stripping
    s = re.sub(r"['’]s\b", "", s)
    s = re.sub(r"['’]", "", s)

    # Collapse dotted legal acronyms before punctuation stripping
    s = re.sub(r"\bl\s*\.\s*p\s*\.\b|\bl\s*\.\s*p\b", " lp ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bl\s*\.\s*l\s*\.\s*c\s*\.\b|\bl\s*\.\s*l\s*\.\s*c\b", " llc ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bl\s*\.\s*l\s*\.\s*p\s*\.\b|\bl\s*\.\s*l\s*\.\s*p\b", " llp ", s, flags=re.IGNORECASE)

    # Unicode-safe punctuation stripping: preserves L*, M* (vowels/matras/viramas), N*
    s = strip_punctuation_unicode(s)
    s = re.sub(r"\s+", " ", s).strip()

    if not s:
        return NormalizedName(
            raw=raw_str,
            norm="",
            clean_norm="",
            tokens=[],
            has_name=False,
            legal_suffix="",
        )

    # Expand legal suffixes and company terms
    for pat, rep in LEGAL_SUFFIX_MAP.items():
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)

    # Standardize generic business abbreviations in norm (kept in clean_norm)
    for pat, rep in GENERIC_ABBREV_MAP.items():
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)

    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split()

    clean_norm = clean_name_tokens(s)

    return NormalizedName(
        raw=raw_str,
        norm=s,
        clean_norm=clean_norm,
        tokens=tokens,
        has_name=True,
        legal_suffix="",
    )


def normalize_address(raw_address: Any) -> NormalizedAddress:
    """
    Normalizes a business address:
    1. Check missing values explicitly (~3.3% of records have no address).
    2. NFKC Unicode normalization + casefold.
    3. Expand standard symbols and slashes.
    4. Expand address abbreviations (thoroughfares, units, landmarks).
    5. Strip punctuation safely using Unicode character categories (L*, M*, N* preserved).
    6. Extract loose postal code (trailing digit group) & street number.
    7. Tokenize.
    """
    if is_missing_value(raw_address):
        return NormalizedAddress(
            raw="",
            norm="",
            tokens=[],
            has_address=False,
            postal_code=None,
            street_number=None,
        )

    raw_str = str(raw_address).strip()
    s = unicodedata.normalize("NFKC", raw_str).casefold()

    # Handle standard separators and symbols
    s = re.sub(r"&", " and ", s)
    s = re.sub(r"['’]s\b", "", s)
    s = re.sub(r"['’]", "", s)
    s = re.sub(r"[/\\,]", " ", s)

    # Expand address abbreviations
    for pat, rep in ADDR_ABBREV_MAP.items():
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)

    # Unicode-safe punctuation stripping: preserves L*, M* (vowels/matras/viramas), N*
    s = strip_punctuation_unicode(s)
    s = re.sub(r"\s+", " ", s).strip()

    if not s:
        return NormalizedAddress(
            raw=raw_str,
            norm="",
            tokens=[],
            has_address=False,
            postal_code=None,
            street_number=None,
        )

    tokens = s.split()
    postal_code, street_number = extract_postal_and_street(tokens)

    return NormalizedAddress(
        raw=raw_str,
        norm=s,
        tokens=tokens,
        has_address=True,
        postal_code=postal_code,
        street_number=street_number,
    )


def normalize_record(
    entity_id: str,
    business_name: Any,
    business_address: Any,
    country: Any = "",
) -> NormalizedRecord:
    """Normalizes an entire record into a NormalizedRecord dataclass."""
    c_str = "" if is_missing_value(country) else str(country).strip()
    return NormalizedRecord(
        entity_id=str(entity_id),
        country=c_str,
        name=normalize_name(business_name),
        address=normalize_address(business_address),
    )
