"""
VERIFY Search/Retrieval Lite
512 MB Render için hafif sürüm.

Kullanım:
    from search_retrieval import search
    raw_sources = search(claim.text, max_sources=5)

Ağır bağımlılıklar YOK:
- sentence-transformers / torch
- CrossEncoder
- Playwright / Chromium
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter
from urllib.parse import urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

SURUM = "5.2-lite-512mb"
RETRIEVAL_API_VERSION = "search-v1"

SEARCH_RESULTS_PER_QUERY = int(os.getenv("LITE_RESULTS_PER_QUERY", "5"))
MAX_QUERY_COUNT = int(os.getenv("LITE_QUERY_COUNT", "3"))
PREFILTER_SOURCE_LIMIT = int(os.getenv("LITE_PREFILTER_SOURCE_LIMIT", "10"))
FETCH_SOURCE_LIMIT = int(os.getenv("LITE_FETCH_SOURCE_LIMIT", "8"))
MAX_CHUNKS_PER_SOURCE = int(os.getenv("LITE_MAX_CHUNKS_PER_SOURCE", "10"))
REQUEST_TIMEOUT = float(os.getenv("LITE_REQUEST_TIMEOUT", "8"))
MIN_FETCHED_TEXT_CHARS = int(os.getenv("LITE_MIN_FETCHED_TEXT_CHARS", "350"))
MIN_SNIPPET_CHARS = int(os.getenv("LITE_MIN_SNIPPET_CHARS", "60"))
CHUNK_TARGET_WORDS = int(os.getenv("LITE_CHUNK_TARGET_WORDS", "90"))
CHUNK_MAX_WORDS = int(os.getenv("LITE_CHUNK_MAX_WORDS", "135"))

LEXICAL_WEIGHT = 0.62
SEARCH_WEIGHT = 0.23
STRUCTURAL_WEIGHT = 0.15

MMR_RELEVANCE_WEIGHT = 0.84
MMR_REDUNDANCY_WEIGHT = 0.16
SAME_DOMAIN_PENALTY = 0.10

STOP_WORDS = {
    "ve", "veya", "ile", "bir", "bu", "şu", "o", "da", "de",
    "mi", "mı", "mu", "mü", "için", "olan", "olarak", "ise",
    "ki", "hem", "çok", "daha", "gibi", "göre", "sonra",
    "önce", "tarafından",

    "the", "a", "an", "and", "or", "of", "to", "in", "on",
    "for", "by", "with", "from", "as", "is", "are", "was",
    "were", "be", "been", "that", "this", "it", "at",
}

SOCIAL_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "x.com",
    "twitter.com",
    "youtube.com",
    "youtu.be",
}

_tavily_client = None


# ==========================================================
# TEXT HELPERS
# ==========================================================

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")

    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )

    text = text.casefold()

    text = re.sub(
        r"[^\w%+.,-]+",
        " ",
        text,
        flags=re.UNICODE,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _tokens(text: str) -> list[str]:
    raw = re.findall(
        r"\b[\w%+.,-]{2,}\b",
        _normalize(text),
        flags=re.UNICODE,
    )

    return [
        token
        for token in raw
        if token not in STOP_WORDS
    ]


def _token_set(text: str) -> set[str]:
    return set(_tokens(text))


def _unique(values) -> list[str]:
    result = []
    seen = set()

    for value in values:
        value = re.sub(
            r"\s+",
            " ",
            str(value or ""),
        ).strip()

        key = _normalize(value)

        if value and key and key not in seen:
            seen.add(key)
            result.append(value)

    return result


def _coverage(
    claim: str,
    text: str,
) -> float:

    claim_tokens = _token_set(claim)
    text_tokens = _token_set(text)

    if not claim_tokens:
        return 0.0

    return (
        len(claim_tokens & text_tokens)
        / len(claim_tokens)
    )


def _jaccard(
    a: str,
    b: str,
) -> float:

    a_tokens = _token_set(a)
    b_tokens = _token_set(b)

    if not a_tokens or not b_tokens:
        return 0.0

    return (
        len(a_tokens & b_tokens)
        / len(a_tokens | b_tokens)
    )


def _lexical(
    claim: str,
    text: str,
) -> float:

    return (
        0.72 * _coverage(claim, text)
        + 0.28 * _jaccard(claim, text)
    )


def _years(text: str) -> set[str]:
    return set(
        re.findall(
            r"\b(?:19|20)\d{2}\b",
            text or "",
        )
    )


def _numbers(text: str) -> set[str]:
    return set(
        re.findall(
            r"(?<!\w)[+-]?\d+(?:[.,]\d+)?\s*%?",
            text or "",
        )
    )


def _structural(
    claim: str,
    evidence: str,
) -> float:

    score = 0.0
    weight = 0.0

    claim_years = _years(claim)

    if claim_years:
        weight += 0.50

        if claim_years & _years(evidence):
            score += 0.50

    if _numbers(claim):
        weight += 0.25

        if _numbers(evidence):
            score += 0.25

    claim_tokens = list(
        dict.fromkeys(
            _tokens(claim)
        )
    )[:8]

    if claim_tokens:
        weight += 0.25

        evidence_tokens = _token_set(
            evidence
        )

        hits = sum(
            1
            for token in claim_tokens
            if token in evidence_tokens
        )

        score += (
            0.25
            * hits
            / len(claim_tokens)
        )

    if weight == 0:
        return _coverage(
            claim,
            evidence,
        )

    return max(
        0.0,
        min(
            1.0,
            score / weight,
        ),
    )


# ==========================================================
# URL HELPERS
# ==========================================================

def _domain(url: str) -> str:
    try:
        host = (
            urlparse(url or "")
            .netloc
            .lower()
        )

        if host.startswith("www."):
            host = host[4:]

        return host

    except Exception:
        return ""


def _canonical_url(url: str) -> str:
    try:
        parsed = urlparse(
            (url or "").strip()
        )

        scheme = (
            parsed.scheme
            or "https"
        )

        netloc = parsed.netloc.lower()

        path = re.sub(
            r"/+$",
            "",
            parsed.path or "",
        )

        return urlunparse(
            (
                scheme,
                netloc,
                path,
                "",
                "",
                "",
            )
        )

    except Exception:
        return (
            url or ""
        ).strip()


def _is_social(url: str) -> bool:
    return (
        _domain(url)
        in SOCIAL_DOMAINS
    )


def _is_pdf(url: str) -> bool:
    try:
        return (
            urlparse(url or "")
            .path
            .lower()
            .endswith(".pdf")
        )

    except Exception:
        return False


# ==========================================================
# OPTIONAL CLAIM FRAME
# ==========================================================

def _frame_hints(
    claim_frame,
) -> list[str]:

    if not isinstance(
        claim_frame,
        dict,
    ):
        return []

    hints = []

    def add(value):

        if (
            isinstance(value, str)
            and value.strip()
        ):
            hints.append(
                value.strip()
            )

        elif isinstance(
            value,
            (int, float),
        ):
            hints.append(
                str(value)
            )

        elif isinstance(
            value,
            list,
        ):
            for item in value:
                add(item)

        elif isinstance(
            value,
            dict,
        ):
            for key in (
                "text",
                "name",
                "canonical",
                "value",
                "year",
            ):
                if key in value:
                    add(
                        value[key]
                    )

    for key in (
        "subject",
        "entity",
        "entities",
        "location",
        "locations",
        "year",
        "years",
        "time",
        "predicate",
        "metric",
        "topic",
    ):
        if key in claim_frame:
            add(
                claim_frame[key]
            )

    return _unique(
        hints
    )


# ==========================================================
# QUERY GENERATION
# ==========================================================

def generate_queries(
    claim_text: str,
    claim_frame=None,
) -> list[str]:

    base = re.sub(
        r"\s+",
        " ",
        claim_text or "",
    ).strip().rstrip(".?!")

    if not base:
        return []

    hints = _frame_hints(
        claim_frame
    )

    important_tokens = list(
        dict.fromkeys(
            _tokens(base)
        )
    )[:16]

    short_query = " ".join(
        _unique(
            hints
            + sorted(_years(base))
            + important_tokens
        )
    )

    queries = [
        base,
        short_query,
        f"{base} official source",
    ]

    return (
        _unique(queries)
        [:MAX_QUERY_COUNT]
    )


# ==========================================================
# TAVILY
# ==========================================================

def _get_tavily_client():
    global _tavily_client

    if _tavily_client is not None:
        return _tavily_client

    api_key = (
        os.getenv(
            "TAVILY_API_KEY",
            "",
        )
        .strip()
    )

    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY bulunamadı. "
            "Render > Environment bölümüne ekle."
        )

    from tavily import TavilyClient

    _tavily_client = TavilyClient(
        api_key=api_key
    )

    return _tavily_client


def _search_web(
    claim_text: str,
    claim_frame=None,
) -> tuple[list[dict], list[str]]:

    client = _get_tavily_client()

    queries = generate_queries(
        claim_text,
        claim_frame,
    )

    results = []
    seen_urls = set()

    for query in queries:

        try:
            response = client.search(
                query=query,
                search_depth="basic",
                max_results=(
                    SEARCH_RESULTS_PER_QUERY
                ),
                include_raw_content=False,
            )

        except Exception:
            continue

        rows = (
            response.get(
                "results",
                [],
            )
            if isinstance(
                response,
                dict,
            )
            else []
        )

        for row in rows:

            url = (
                row.get("url")
                or ""
            ).strip()

            if not url:
                continue

            canonical = _canonical_url(
                url
            )

            if canonical in seen_urls:
                continue

            seen_urls.add(
                canonical
            )

            results.append({
                "title": re.sub(
                    r"\s+",
                    " ",
                    row.get(
                        "title",
                        "",
                    )
                    or "",
                ).strip(),

                "url": url,

                "content": re.sub(
                    r"\s+",
                    " ",
                    row.get(
                        "content",
                        "",
                    )
                    or "",
                ).strip(),

                "tavily_score": float(
                    row.get(
                        "score"
                    )
                    or 0.0
                ),

                "published_date": (
                    row.get(
                        "published_date"
                    )
                ),
            })

    return (
        results,
        queries,
    )


# ==========================================================
# PREFILTER
# ==========================================================

def _prefilter(
    claim_text: str,
    rows: list[dict],
) -> list[dict]:

    scored = []

    for row in rows:

        if _is_social(
            row.get(
                "url",
                "",
            )
        ):
            continue

        combined = (
            f"{row.get('title', '')} "
            f"{row.get('content', '')}"
        )

        lexical = _lexical(
            claim_text,
            combined,
        )

        tavily_score = max(
            0.0,
            min(
                1.0,
                float(
                    row.get(
                        "tavily_score"
                    )
                    or 0.0
                ),
            ),
        )

        search_score = (
            0.46 * lexical
            + 0.54 * tavily_score
        )

        item = dict(row)

        item["search_score"] = (
            search_score
        )

        scored.append(
            item
        )

    scored.sort(
        key=lambda item: (
            item["search_score"]
        ),
        reverse=True,
    )

    return scored[
        :PREFILTER_SOURCE_LIMIT
    ]


# ==========================================================
# HTML FETCH
# ==========================================================

def _html_to_text(
    html: str,
) -> str:

    soup = BeautifulSoup(
        html or "",
        "html.parser",
    )

    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "form",
        "nav",
        "footer",
        "header",
        "aside",
        "iframe",
    ]):
        tag.decompose()

    root = (
        soup.find("article")
        or soup.find("main")
        or soup.body
        or soup
    )

    return re.sub(
        r"\s+",
        " ",
        root.get_text(
            " ",
            strip=True,
        ),
    ).strip()


def _fetch_page(
    url: str,
) -> str:

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 "
            "Chrome/124 Safari/537.36"
        )
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

        response.raise_for_status()

        content_type = (
            response.headers.get(
                "Content-Type",
                "",
            )
            .lower()
        )

        if (
            "html"
            not in content_type
            and "text"
            not in content_type
        ):
            return ""

        return _html_to_text(
            response.text
        )

    except Exception:
        return ""


def _source_text(
    row: dict,
) -> tuple[str, str]:

    snippet = re.sub(
        r"\s+",
        " ",
        row.get(
            "content",
            "",
        )
        or "",
    ).strip()

    url = row.get(
        "url",
        "",
    )

    if _is_pdf(url):

        if (
            len(snippet)
            >= MIN_SNIPPET_CHARS
        ):
            return (
                snippet,
                "tavily_pdf_snippet",
            )

        return (
            "",
            "none",
        )

    fetched = _fetch_page(
        url
    )

    if (
        len(fetched)
        >= MIN_FETCHED_TEXT_CHARS
    ):
        return (
            fetched,
            "requests",
        )

    if (
        len(snippet)
        >= MIN_SNIPPET_CHARS
    ):
        return (
            snippet,
            "tavily_snippet",
        )

    if fetched:
        return (
            fetched,
            "requests_short",
        )

    return (
        "",
        "none",
    )


# ==========================================================
# CHUNKING
# ==========================================================

def _sentences(
    text: str,
) -> list[str]:

    text = re.sub(
        r"\s+",
        " ",
        text or "",
    ).strip()

    if not text:
        return []

    protected = re.sub(
        r"(?<=\d)\.(?=\d)",
        "<DOT>",
        text,
    )

    parts = re.split(
        r"(?<=[.!?])\s+",
        protected,
    )

    return [
        part.replace(
            "<DOT>",
            ".",
        ).strip()

        for part in parts

        if part.strip()
    ]


def _chunk_text(
    text: str,
) -> list[str]:

    sentences = _sentences(
        text
    )

    if not sentences:
        return []

    chunks = []

    current = []
    word_count = 0

    for sentence in sentences:

        sentence_words = len(
            sentence.split()
        )

        if (
            current
            and word_count
            + sentence_words
            > CHUNK_MAX_WORDS
        ):
            chunk = " ".join(
                current
            ).strip()

            if len(chunk) >= 60:
                chunks.append(
                    chunk
                )

            current = (
                current[-1:]
            )

            word_count = sum(
                len(item.split())
                for item in current
            )

        current.append(
            sentence
        )

        word_count += (
            sentence_words
        )

        if (
            word_count
            >= CHUNK_TARGET_WORDS
        ):
            chunk = " ".join(
                current
            ).strip()

            if len(chunk) >= 60:
                chunks.append(
                    chunk
                )

            current = (
                current[-1:]
            )

            word_count = sum(
                len(item.split())
                for item in current
            )

    if current:

        chunk = " ".join(
            current
        ).strip()

        if len(chunk) >= 60:
            chunks.append(
                chunk
            )

    return _unique(
        chunks
    )


def _focus(
    claim_text: str,
    chunk: str,
    max_sentences: int = 3,
) -> str:

    sentences = _sentences(
        chunk
    )

    if (
        len(sentences)
        <= max_sentences
    ):
        return chunk.strip()

    best_index = max(
        range(
            len(sentences)
        ),
        key=lambda index: (
            _lexical(
                claim_text,
                sentences[index],
            )
        ),
    )

    indexes = {
        best_index
    }

    if best_index > 0:
        indexes.add(
            best_index - 1
        )

    if (
        best_index + 1
        < len(sentences)
    ):
        indexes.add(
            best_index + 1
        )

    return " ".join(
        sentences[index]
        for index in sorted(
            indexes
        )[:max_sentences]
    ).strip()


# ==========================================================
# CANDIDATES
# ==========================================================

def _build_candidates(
    claim_text: str,
    rows: list[dict],
) -> tuple[list[dict], dict]:

    candidates = []

    requests_count = 0
    snippet_count = 0

    for row in rows[
        :FETCH_SOURCE_LIMIT
    ]:

        text, fetch_method = (
            _source_text(row)
        )

        if not text:
            continue

        if fetch_method.startswith(
            "requests"
        ):
            requests_count += 1

        elif fetch_method.startswith(
            "tavily"
        ):
            snippet_count += 1

        chunks = _chunk_text(
            text
        )

        if (
            not chunks
            and len(text)
            >= MIN_SNIPPET_CHARS
        ):
            chunks = [
                text
            ]

        chunks = sorted(
            chunks,
            key=lambda chunk: (
                _lexical(
                    claim_text,
                    chunk,
                )
            ),
            reverse=True,
        )[:MAX_CHUNKS_PER_SOURCE]

        for chunk in chunks:

            focused = _focus(
                claim_text,
                chunk,
            )

            ranking_text = (
                f"{row.get('title', '')}\n"
                f"{focused}"
            )

            lexical = _lexical(
                claim_text,
                ranking_text,
            )

            structural = _structural(
                claim_text,
                focused,
            )

            search_score = max(
                0.0,
                min(
                    1.0,
                    float(
                        row.get(
                            "search_score"
                        )
                        or 0.0
                    ),
                ),
            )

            retrieval_score = (
                LEXICAL_WEIGHT
                * lexical

                + SEARCH_WEIGHT
                * search_score

                + STRUCTURAL_WEIGHT
                * structural
            )

            candidates.append({
                "text": focused,

                "title": (
                    row.get(
                        "title",
                        "",
                    )
                    or ""
                ),

                "url": (
                    row.get(
                        "url",
                        "",
                    )
                    or ""
                ),

                "domain": _domain(
                    row.get(
                        "url",
                        "",
                    )
                ),

                "published_date": (
                    row.get(
                        "published_date"
                    )
                ),

                "fetch_method": (
                    fetch_method
                ),

                "lexical_score": (
                    lexical
                ),

                "search_score": (
                    search_score
                ),

                "structural_score": (
                    structural
                ),

                "retrieval_score": max(
                    0.0,
                    min(
                        1.0,
                        retrieval_score,
                    ),
                ),
            })

    candidates.sort(
        key=lambda item: (
            item[
                "retrieval_score"
            ]
        ),
        reverse=True,
    )

    diagnostics = {
        "requests_sources": (
            requests_count
        ),

        "snippet_sources": (
            snippet_count
        ),
    }

    return (
        candidates,
        diagnostics,
    )


# ==========================================================
# MMR / SOURCE DIVERSITY
# ==========================================================

def _select_diverse(
    candidates: list[dict],
    k: int,
) -> list[dict]:

    if (
        not candidates
        or k <= 0
    ):
        return []

    pool = candidates[
        :min(
            len(candidates),
            30,
        )
    ]

    selected = []

    used_domains = Counter()

    while (
        pool
        and len(selected)
        < k
    ):

        best_candidate = None
        best_score = -999.0

        for candidate in pool:

            if selected:
                redundancy = max(
                    _jaccard(
                        candidate["text"],
                        chosen["text"],
                    )
                    for chosen
                    in selected
                )

            else:
                redundancy = 0.0

            domain = candidate.get(
                "domain",
                "",
            )

            domain_penalty = (
                SAME_DOMAIN_PENALTY

                if (
                    domain
                    and used_domains[
                        domain
                    ] > 0
                )

                else 0.0
            )

            mmr_score = (
                MMR_RELEVANCE_WEIGHT
                * candidate[
                    "retrieval_score"
                ]

                - MMR_REDUNDANCY_WEIGHT
                * redundancy

                - domain_penalty
            )

            if (
                mmr_score
                > best_score
            ):
                best_score = (
                    mmr_score
                )

                best_candidate = (
                    candidate
                )

        if best_candidate is None:
            break

        best_candidate[
            "mmr_score"
        ] = best_score

        selected.append(
            best_candidate
        )

        domain = (
            best_candidate.get(
                "domain",
                "",
            )
        )

        if domain:
            used_domains[
                domain
            ] += 1

        pool.remove(
            best_candidate
        )

    return selected


# ==========================================================
# PUBLIC API
# ==========================================================

def search_detailed(
    claim_text: str,
    max_sources: int = 5,
    claim_frame=None,
    debug: bool = False,
) -> dict:

    if (
        not isinstance(
            claim_text,
            str,
        )
        or not claim_text.strip()
    ):
        return {
            "version": SURUM,

            "api_version": (
                RETRIEVAL_API_VERSION
            ),

            "claim": claim_text,

            "status": (
                "invalid_claim"
            ),

            "sources": [],

            "diagnostics": {},
        }

    claim_text = re.sub(
        r"\s+",
        " ",
        claim_text,
    ).strip()

    try:
        max_sources = int(
            max_sources
        )

    except (
        TypeError,
        ValueError,
    ):
        max_sources = 5

    max_sources = max(
        1,
        min(
            max_sources,
            10,
        ),
    )

    search_rows, queries = (
        _search_web(
            claim_text,
            claim_frame,
        )
    )

    prefiltered = _prefilter(
        claim_text,
        search_rows,
    )

    candidates, fetch_diagnostics = (
        _build_candidates(
            claim_text,
            prefiltered,
        )
    )

    selected = _select_diverse(
        candidates,
        max_sources,
    )

    sources = []

    for index, candidate in enumerate(
        selected,
        start=1,
    ):

        sources.append({
            "id": (
                f"source_{index}"
            ),

            "url": (
                candidate[
                    "url"
                ]
            ),

            "text": (
                candidate[
                    "text"
                ]
            ),

            "title": (
                candidate[
                    "title"
                ]
            ),

            "domain": (
                candidate[
                    "domain"
                ]
            ),

            "published_date": (
                candidate[
                    "published_date"
                ]
            ),

            "retrieval_score": round(
                candidate[
                    "retrieval_score"
                ],
                4,
            ),

            "candidate_type": (
                "lite_chunk"
            ),

            "metadata": {
                "retrieval_version": (
                    SURUM
                ),

                "api_version": (
                    RETRIEVAL_API_VERSION
                ),

                "scoring_mode": (
                    "lite_lexical"
                ),

                "lexical_score": round(
                    candidate[
                        "lexical_score"
                    ],
                    4,
                ),

                "search_score": round(
                    candidate[
                        "search_score"
                    ],
                    4,
                ),

                "structural_score": round(
                    candidate[
                        "structural_score"
                    ],
                    4,
                ),

                "mmr_score": round(
                    candidate.get(
                        "mmr_score",
                        0.0,
                    ),
                    4,
                ),

                "fetch_method": (
                    candidate[
                        "fetch_method"
                    ]
                ),
            },
        })

    result = {
        "version": SURUM,

        "api_version": (
            RETRIEVAL_API_VERSION
        ),

        "claim": (
            claim_text
        ),

        "status": (
            "evidence_found"
            if sources
            else "not_enough_evidence"
        ),

        "sources": (
            sources
        ),

        "diagnostics": {
            "mode": (
                "LITE"
            ),

            "queries": (
                queries
            ),

            "raw_search_results": len(
                search_rows
            ),

            "prefiltered_sources": len(
                prefiltered
            ),

            "candidate_count": len(
                candidates
            ),

            "selected_sources": len(
                sources
            ),

            **fetch_diagnostics,

            "heavy_features_disabled": {
                "sentence_transformers": True,
                "pytorch": True,
                "cross_encoder": True,
                "playwright": True,
                "chromium": True,
                "local_embeddings": True,
            },

            "truth_decision_in_retrieval": (
                False
            ),
        },
    }

    if debug:

        print(
            "\n"
            "=== VERIFY RETRIEVAL LITE ==="
        )

        print(
            "Claim:",
            claim_text,
        )

        print(
            "Queries:",
            queries,
        )

        print(
            "Search results:",
            len(search_rows),
        )

        print(
            "Candidates:",
            len(candidates),
        )

        print(
            "Selected:",
            len(sources),
        )

        for source in sources:

            print(
                "\n---"
            )

            print(
                source["id"],
                source["title"],
            )

            print(
                source["url"]
            )

            print(
                "score:",
                source[
                    "retrieval_score"
                ],
            )

            print(
                source["text"]
            )

    return result


def search(
    claim_text: str,
    max_sources: int = 5,
    claim_frame=None,
) -> list[dict]:

    """
    backend/main.py ile uyumlu drop-in fonksiyon.

    Mevcut backend:

        raw_sources = search(
            claim.text,
            max_sources=5
        )

    şeklinde çalışmaya devam eder.
    """

    return search_detailed(
        claim_text=claim_text,
        max_sources=max_sources,
        claim_frame=claim_frame,
        debug=False,
    )["sources"]


# ==========================================================
# SELF TEST
# ==========================================================

def _run_self_tests():

    checks = [
        (
            "Turkish normalization",

            "turkiye"
            in _normalize(
                "Türkiye"
            ),
        ),

        (
            "Query generation",

            len(
                generate_queries(
                    "Orion 2024 yılında "
                    "Atlas pazarında "
                    "yüzde 20 büyüdü."
                )
            )
            >= 2,
        ),

        (
            "Year extraction",

            _years(
                "2024 ve 2023"
            )
            == {
                "2023",
                "2024",
            },
        ),

        (
            "Counter evidence stays relevant",

            _structural(
                (
                    "Orion 2024 yılında "
                    "satışlarını yüzde 20 artırdı."
                ),
                (
                    "Orion'un 2024 satışları "
                    "yüzde 12 arttı."
                ),
            )
            > 0.40,
        ),
    ]

    passed = sum(
        1
        for _, success
        in checks
        if success
    )

    print(
        f"Lite self-tests: "
        f"{passed}/{len(checks)}"
    )

    for name, success in checks:

        print(
            (
                "PASS"
                if success
                else "FAIL"
            ),
            "-",
            name,
        )

    if passed != len(checks):
        raise SystemExit(1)


# ==========================================================
# CLI TEST
# ==========================================================

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "VERIFY Search/Retrieval Lite"
        )
    )

    parser.add_argument(
        "claim",
        nargs="*",
    )

    parser.add_argument(
        "--max-sources",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:

        _run_self_tests()

        return

    claim = " ".join(
        args.claim
    ).strip()

    if not claim:

        claim = input(
            "Atomic claim: "
        ).strip()

    search_detailed(
        claim_text=claim,
        max_sources=(
            args.max_sources
        ),
        debug=True,
    )


if __name__ == "__main__":
    main()