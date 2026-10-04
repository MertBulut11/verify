import os
import re
import sys
import json
import math
import unicodedata
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlunparse


# ==========================================================
# RETRIEVAL V5.2
# Integration-ready generic retrieval service
# ==========================================================

SURUM = "5.2"
JSON_DOSYA_ADI = "retrieval_sonucu_v5_2.json"

# Public integration contract. Person 3 can keep calling:
#     search(claim_text: str, max_sources: int = 5) -> list[dict]
# Extra fields are intentionally additive; Person 1 only needs id/url/text.
RETRIEVAL_API_VERSION = "search-v1"
SOURCE_ID_PREFIX = "source"


# Search / source settings
HER_SORGU_ICIN_SONUC = 6
ON_FILTRE_LIMITI = 12
TOP_KANIT_SAYISI = 5
MAX_CHUNK_PER_SOURCE = 28

# Chunking
CHUNK_HEDEF_KELIME = 115
CHUNK_MAX_KELIME = 155
CHUNK_OVERLAP_CUMLE = 1
ADJACENT_PAIR_ENABLED = True

# Retrieval pool
MIN_EMBEDDING_BENZERLIGI = 0.42
BI_ENCODER_TOP_N = 50
CROSS_ENCODER_TOP_N = 40

# Network
REQUEST_TIMEOUT = 10
PLAYWRIGHT_TIMEOUT = 30000
REQUESTS_MIN_KARAKTER = 450
PLAYWRIGHT_MIN_KARAKTER = 100
TAVILY_FALLBACK_MIN_KARAKTER = 80

# Models
EMBEDDING_MODEL_NAME = os.getenv(
    "RETRIEVAL_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

CROSS_ENCODER_MODEL_NAME = os.getenv(
    "RETRIEVAL_CROSS_ENCODER_MODEL",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
)

CROSS_ENCODER_ENABLED = os.getenv(
    "RETRIEVAL_CROSS_ENCODER_ENABLED",
    "1",
).strip().lower() not in {"0", "false", "hayir", "hayır", "no"}

# V5 philosophy:
# - Embedding/cross-encoder answer "Bu evidence claim ile ne kadar ilgili?"
# - Structured constraints answer "Claim'deki sayi/tarih/rank evidence'da nasil geciyor?"
# - Retrieval DOES NOT decide final truth. Person1 / verifier does.


# ==========================================================
# GENERIC LANGUAGE RESOURCES
# ==========================================================

STOP_WORDS = {
    "ve", "veya", "ile", "bir", "bu", "şu", "o", "da", "de",
    "mi", "mı", "mu", "mü", "için", "olan", "olarak", "dedi",
    "nin", "nın", "nun", "nün", "na", "ne", "daki", "deki",
    "ki", "ise", "hem", "çok", "daha",
}

SOSYAL_MEDYA_DOMAINLERI = {
    "facebook.com", "www.facebook.com",
    "instagram.com", "www.instagram.com",
    "youtube.com", "www.youtube.com", "youtu.be",
    "tiktok.com", "www.tiktok.com",
    "x.com", "www.x.com", "twitter.com", "www.twitter.com",
}

AYLAR = {
    "ocak", "şubat", "mart", "nisan", "mayıs", "haziran",
    "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık",
}

ORDINAL_KELIME_HARITASI = {
    "birinci": 1,
    "ikinci": 2,
    "üçüncü": 3,
    "dördüncü": 4,
    "beşinci": 5,
    "altıncı": 6,
    "yedinci": 7,
    "sekizinci": 8,
    "dokuzuncu": 9,
    "onuncu": 10,
    "on birinci": 11,
    "on ikinci": 12,
    "on üçüncü": 13,
    "on dördüncü": 14,
    "on beşinci": 15,
    "on altıncı": 16,
    "on yedinci": 17,
    "on sekizinci": 18,
    "on dokuzuncu": 19,
    "yirminci": 20,
}


# Generic category hints near ranking language. These are category words,
# not named entities or domain-specific brands. Unknown categories remain UNKNOWN.
RANKING_SCOPE_HINT_PATTERNS = [
    ("BRAND", [r"\bmarka(?:sı|sının|lar|ları)?\b"]),
    ("MODEL", [r"\bmodel(?:i|in|ler|leri)?\b"]),
    ("COUNTRY", [r"\bülke(?:si|nin|ler|leri)?\b"]),
    ("CITY", [r"\bşehir(?:i|in|ler|leri)?\b", r"\bkent(?:i|in|ler|leri)?\b"]),
    ("COMPANY", [r"\bşirket(?:i|in|ler|leri)?\b", r"\bfirma(?:sı|nın|lar|ları)?\b"]),
    ("PRODUCT", [r"\bürün(?:ü|ün|ler|leri)?\b"]),
    ("TEAM", [r"\btakım(?:ı|ın|lar|ları)?\b"]),
    ("UNIVERSITY", [r"\büniversite(?:si|nin|ler|leri)?\b"]),
    ("PERSON", [r"\bkişi(?:si|nin|ler|leri)?\b", r"\boyuncu(?:su|nun|lar|ları)?\b"]),
]

SUPERLATIVE_RANK1_PATTERNS = [
    r"\bbirinci\s+sıra(?:da|ya|yı|nın)?\b",
    r"\bilk\s+sıra(?:da|ya|yı|nın)?\b",
    r"\blider(?:i|lik|liği|liğini|liğine|liğinde|liğe|ler|leri|\s+marka|\s+ülke|\s+şirket)?\b",
    r"\ben\s+büyük\b",
    r"\ben\s+yüksek\b",
    r"\ben\s+fazla\b",
    r"\ben\s+çok\b",
]

COMPARISON_PATTERNS = [
    ("SAME", [r"\baynı\s+hızda\b", r"\beşit\b", r"\baynı\s+oranda\b"]),
    ("GREATER", [r"\bdaha\s+fazla\b", r"\bdaha\s+yüksek\b", r"\bdaha\s+büyük\b", r"\bönde\b"]),
    ("LESS", [r"\bdaha\s+az\b", r"\bdaha\s+düşük\b", r"\bdaha\s+küçük\b", r"\bgeride\b"]),
    ("FASTER", [r"\bdaha\s+hızlı\b"]),
    ("SLOWER", [r"\bdaha\s+yavaş\b"]),
]

UP_PATTERNS = [
    r"\bart(?:tı|ış|arak|ıyor|acak|mıştır|mış)\b",
    r"\byüksel(?:di|iş|erek|iyor|ecek|miştir|miş)\b",
    r"\bbüyü(?:dü|me|yerek|yor|yecek|müştür|müş)\b",
    r"\bçoğal(?:dı|ma|ıyor|acak)\b",
]

DOWN_PATTERNS = [
    r"\bazal(?:dı|ma|arak|ıyor|acak|mıştır|mış)\b",
    r"\bdüş(?:tü|üş|erek|üyor|ecek|müştür|müş)\b",
    r"\bgerile(?:di|me|yerek|iyor|yecek|miştir|miş)\b",
    r"\bdaral(?:dı|ma|ıyor|acak)\b",
]

FUTURE_PATTERNS = [
    r"\b(?:acak|ecek)\b",
    r"\bplanlanıyor\b",
    r"\bplanlıyor\b",
    r"\bbekleniyor\b",
    r"\böngörülüyor\b",
    r"\bhedefleniyor\b",
    r"\btahmin\s+ediliyor\b",
]

NEGATION_PATTERNS = [
    r"\bdeğil\b",
    r"\byok\b",
    r"\bhiç\b",
    r"\byasaklanmadı\b",
    r"\bgerçekleşmedi\b",
    r"\baçılmadı\b",
    r"\bolmadı\b",
]

RELATIVE_TIME_PATTERNS = {
    "today": [r"\bbugün\b"],
    "yesterday": [r"\bdün\b"],
    "last_week": [r"\bgeçen\s+hafta\b"],
    "this_week": [r"\bbu\s+hafta\b"],
    "last_month": [r"\bgeçen\s+ay\b"],
    "this_month": [r"\bbu\s+ay\b"],
}

VERBISH_SUFFIX_PATTERN = re.compile(
    r"(?:dı|di|du|dü|tı|ti|tu|tü|mış|miş|muş|müş|yor|acak|ecek|"
    r"ar|er|ır|ir|ur|ür|maz|mez|dır|dir|dur|dür|tır|tir|tur|tür)$",
    re.IGNORECASE,
)



# V5.2 generic semantic resources. These are linguistic/category patterns,
# never named brands, countries, people, products, or test-specific entities.
ENTITY_TYPE_HINTS = [
    ("MODEL", [r"\bmodel(?:i|in|ler|leri)?\b"]),
    ("BRAND", [r"\bmarka(?:sı|sının|lar|ları)?\b"]),
    ("COUNTRY", [r"\bülke(?:si|nin|ler|leri)?\b", r"\bdevlet(?:i|in|ler|leri)?\b"]),
    ("CITY", [r"\bşehir(?:i|in|ler|leri)?\b", r"\bkent(?:i|in|ler|leri)?\b"]),
    ("COMPANY", [r"\bşirket(?:i|in|ler|leri)?\b", r"\bfirma(?:sı|nın|lar|ları)?\b", r"\bkuruluş(?:u|un|lar|ları)?\b"]),
    ("TEAM", [r"\btakım(?:ı|ın|lar|ları)?\b", r"\bkulüp(?:ü|ün|ler|leri)?\b"]),
    ("UNIVERSITY", [r"\büniversite(?:si|nin|ler|leri)?\b"]),
    ("PERSON", [r"\bkişi(?:si|nin|ler|leri)?\b", r"\boyuncu(?:su|nun|lar|ları)?\b", r"\bsporcu(?:su|nun|lar|ları)?\b"]),
    ("PRODUCT", [r"\bürün(?:ü|ün|ler|leri)?\b", r"\bcihaz(?:ı|ın|lar|ları)?\b"]),
]

GENERIC_ANAPHORA_PATTERNS = [
    (r"^(?:bu\s+)?şirketin\b", "{subject} şirketinin"),
    (r"^(?:bu\s+)?şirket\b", "{subject} şirketi"),
    (r"^(?:bu\s+)?firmanın\b", "{subject} firmasının"),
    (r"^(?:bu\s+)?firma\b", "{subject} firması"),
    (r"^(?:bu\s+)?markanın\b", "{subject} markasının"),
    (r"^(?:bu\s+)?marka\b", "{subject} markası"),
    (r"^(?:bu\s+)?kurumun\b", "{subject} kurumunun"),
    (r"^(?:bu\s+)?kurum\b", "{subject} kurumu"),
    (r"^(?:bu\s+)?kuruluşun\b", "{subject} kuruluşunun"),
    (r"^(?:bu\s+)?takımın\b", "{subject} takımının"),
    (r"^(?:bu\s+)?takım\b", "{subject} takımı"),
    (r"^(?:bu\s+)?ülkenin\b", "{subject} ülkesinin"),
    (r"^(?:bu\s+)?kişinin\b", "{subject} kişisinin"),
    (r"^onun\b", "{subject}"),
]

PROPER_NOUN_FALSE_POSITIVES = {
    "Bu", "Bir", "İlk", "Son", "Geçen", "Yıl", "Ay", "Bugün", "Dün",
    "Ancak", "Ama", "Fakat", "Öte", "Buna", "Bunun", "Söz", "Aynı",
    "Kaynak", "Haber", "Metin", "Rapor", "Veri", "Veriler",
}

PREDICATE_EVENT_HINTS = [
    r"\baç(?:tı|ıldı|ıyor|ılacak)\b", r"\bkur(?:du|uldu|uyor|ulacak)\b",
    r"\bonaylan(?:dı|ıyor|acak)\b", r"\bkazan(?:dı|ıyor|acak)\b",
    r"\bkaybet(?:ti|iyor|ecek)\b", r"\bbaşla(?:dı|yor|yacak)\b",
    r"\bbit(?:ti|iyor|ecek)\b", r"\bgönder(?:di|iyor|ecek)\b",
    r"\byasaklan(?:dı|ıyor|acak)\b", r"\bgerçekleş(?:ti|iyor|ecek)\b",
]

# ==========================================================
# LAZY EXTERNAL DEPENDENCIES / MODELS
# ==========================================================

_embedding_model = None
_cross_encoder_model = None
_cross_encoder_failed_reason = None
_tavily_client = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Embedding modeli yükleniyor: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_cross_encoder_model():
    global _cross_encoder_model, _cross_encoder_failed_reason

    if not CROSS_ENCODER_ENABLED:
        _cross_encoder_failed_reason = "disabled"
        return None

    if _cross_encoder_model is not None:
        return _cross_encoder_model

    if _cross_encoder_failed_reason is not None:
        return None

    try:
        from sentence_transformers import CrossEncoder
        print(f"Cross-encoder yükleniyor: {CROSS_ENCODER_MODEL_NAME}")
        _cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
        return _cross_encoder_model
    except Exception as exc:
        _cross_encoder_failed_reason = f"{type(exc).__name__}: {exc}"
        print(
            "[UYARI] Cross-encoder yüklenemedi. "
            "Embedding + structured fallback kullanılacak."
        )
        print(f"[UYARI] Sebep: {_cross_encoder_failed_reason}")
        return None


def get_tavily_client():
    global _tavily_client
    if _tavily_client is not None:
        return _tavily_client

    from dotenv import load_dotenv
    from tavily import TavilyClient

    load_dotenv()
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY bulunamadı. .env dosyana TAVILY_API_KEY ekle."
        )

    _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


# ==========================================================
# BASIC TEXT HELPERS
# ==========================================================


def turkce_kucult(metin):
    if not metin:
        return ""
    return metin.replace("I", "ı").replace("İ", "i").lower()


def metni_normalize_et(metin):
    metin = turkce_kucult(metin or "")
    metin = unicodedata.normalize("NFKC", metin)
    metin = metin.replace("’", "'").replace("`", "'")
    metin = re.sub(r"\s+", " ", metin).strip()
    return metin


def tokenlara_ayir(metin):
    normal = metni_normalize_et(metin)
    return re.findall(r"[a-zçğıöşü0-9]+(?:'[a-zçğıöşü0-9]+)?", normal)


def icerik_tokenlari(metin):
    return [
        t for t in tokenlara_ayir(metin)
        if len(t) > 1 and t not in STOP_WORDS
    ]


def token_overlap(a, b):
    sa = set(icerik_tokenlari(a))
    sb = set(icerik_tokenlari(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa))


def jaccard_overlap(a, b):
    sa = set(icerik_tokenlari(a))
    sb = set(icerik_tokenlari(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def domain_bul(url):
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def sosyal_medya_mi(url):
    d = urlparse(url).netloc.lower()
    return d in SOSYAL_MEDYA_DOMAINLERI


def pdf_mi(url):
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")


def canonical_url(url):
    try:
        p = urlparse(url)
        host = p.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        path = re.sub(r"/+", "/", p.path).rstrip("/")
        return urlunparse((p.scheme.lower() or "https", host, path, "", "", ""))
    except Exception:
        return url


def sigmoid(x):
    try:
        x = float(x)
    except Exception:
        return 0.5
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ==========================================================
# INPUT
# ==========================================================


def iddia_al():
    print("\nKontrol etmek istediğiniz iddiayı girin.")
    print("Birden fazla satır yazabilirsiniz.")
    print("Bitirmek için boş bir satır bırakın.\n")

    satirlar = []
    while True:
        try:
            satir = input("> ")
        except EOFError:
            break
        if not satir.strip():
            break
        satirlar.append(satir.strip())

    return " ".join(satirlar).strip()


# ==========================================================
# RELATIVE TIME
# ==========================================================


def goreli_zaman_bilgisi(metin, now=None):
    normal = metni_normalize_et(metin)
    now = now or datetime.now()

    bulunan = None
    for key, patterns in RELATIVE_TIME_PATTERNS.items():
        if any(re.search(p, normal) for p in patterns):
            bulunan = key
            break

    if bulunan is None:
        return {
            "has_relative_time": False,
            "type": None,
            "start": None,
            "end": None,
        }

    today = now.date()

    if bulunan == "today":
        start = end = today
    elif bulunan == "yesterday":
        start = end = today - timedelta(days=1)
    elif bulunan == "last_week":
        this_monday = today - timedelta(days=today.weekday())
        start = this_monday - timedelta(days=7)
        end = this_monday - timedelta(days=1)
    elif bulunan == "this_week":
        start = today - timedelta(days=today.weekday())
        end = today
    elif bulunan == "last_month":
        first_this = today.replace(day=1)
        end = first_this - timedelta(days=1)
        start = end.replace(day=1)
    elif bulunan == "this_month":
        start = today.replace(day=1)
        end = today
    else:
        start = end = None

    return {
        "has_relative_time": True,
        "type": bulunan,
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
    }


# ==========================================================
# GENERIC STRUCTURED CLAIM / EVIDENCE SCHEMA (V5.1)
# ==========================================================


def yillari_cikar(metin):
    normal = metni_normalize_et(metin)
    return sorted({
        int(x) for x in re.findall(r"(?<!\d)(19\d{2}|20\d{2}|21\d{2})(?!\d)", normal)
    })


def yuzdeleri_cikar(metin):
    return [x["value"] for x in yuzde_adaylarini_cikar(metin)]


def _ordinal_context(normal, start, end, window=150):
    a = max(0, start - window)
    b = min(len(normal), end + window)
    return normal[a:b].strip()


def _sentence_span(normal, position):
    if not normal:
        return 0, 0
    left_marks = [normal.rfind(x, 0, max(0, position)) for x in [".", "!", "?", ";", "\n"]]
    left = max(left_marks) + 1
    right_candidates = []
    for x in [".", "!", "?", ";", "\n"]:
        p = normal.find(x, position)
        if p >= 0:
            right_candidates.append(p)
    right = min(right_candidates) if right_candidates else len(normal)
    return left, right


def _norm_position(original_text, original_pos):
    return len(metni_normalize_et((original_text or "")[:original_pos]))


def _entity_type_from_context(normal, start, end):
    window = normal[max(0, start - 70): min(len(normal), end + 90)]
    for label, patterns in ENTITY_TYPE_HINTS:
        if any(re.search(p, window) for p in patterns):
            return label
    return "UNKNOWN"


def _looks_locative_suffix(text_after):
    after = turkce_kucult(text_after or "")
    return bool(re.match(
        r"^[’'](?:de|da|te|ta|den|dan|ten|tan|deki|daki|teki|taki|ye|ya|e|a)\b",
        after,
    ))


def entity_adaylarini_cikar(metin):
    """Generic named-entity candidates from surface form.

    This is a fallback, not a knowledge base. It never contains named brands,
    countries or people. Person1/NER can later provide richer entities via schema.
    """
    text = metin or ""
    normal = metni_normalize_et(text)
    pattern = re.compile(
        r"(?<![\wÇĞİÖŞÜçğıöşü])"
        r"([A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü0-9&.-]*"
        r"(?:\s+[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü0-9&.-]*){0,3})"
    )
    out = []
    seen = set()
    for m in pattern.finditer(text):
        raw = m.group(1).strip(" ,.;:()[]{}")
        if not raw:
            continue
        if raw in PROPER_NOUN_FALSE_POSITIVES:
            continue
        if turkce_kucult(raw) in AYLAR:
            continue
        # Single generic sentence-initial category words are not named entities.
        if turkce_kucult(raw) in {
            "şirket", "firma", "marka", "ülke", "şehir", "takım", "kurum",
            "model", "ürün", "rapor", "veri", "kaynak",
        }:
            continue

        norm_start = _norm_position(text, m.start(1))
        norm_end = norm_start + len(metni_normalize_et(raw))
        suffix_after = text[m.end(1): m.end(1) + 12]
        role = "LOCATION" if _looks_locative_suffix(suffix_after) else "ENTITY"
        typ = _entity_type_from_context(normal, norm_start, norm_end)
        canonical = metni_normalize_et(raw)
        key = (canonical, norm_start, role)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "text": raw,
            "canonical": canonical,
            "type": typ,
            "role": role,
            "position": norm_start,
            "end_position": norm_end,
            "is_named": True,
        })
    return out


def lokasyonlari_cikar(metin, entities=None):
    normal = metni_normalize_et(metin)
    entities = entities if entities is not None else entity_adaylarini_cikar(metin)
    out = []
    seen = set()

    for e in entities:
        if e.get("role") == "LOCATION":
            key = e.get("canonical")
            if key and key not in seen:
                out.append({
                    "text": e.get("text"),
                    "canonical": key,
                    "type": e.get("type") if e.get("type") != "UNKNOWN" else "LOCATION",
                    "position": e.get("position"),
                })
                seen.add(key)

    # Generic global scope does not need a gazetteer.
    if re.search(r"\b(?:dünya\s+genelinde|dünya\s+çapında|küresel(?:de|\s+olarak)?)\b", normal):
        if "dünya" not in seen:
            out.append({"text": "dünya", "canonical": "dünya", "type": "GLOBAL", "position": 0})
            seen.add("dünya")

    return out


def _fallback_phrase_subject(metin, locations):
    normal = metni_normalize_et(metin)
    if not normal:
        return None

    # Remove leading locative phrases and years, then keep content before a likely predicate.
    work = re.sub(r"(?<!\d)(?:19\d{2}|20\d{2}|21\d{2})(?!\d)", " ", normal)
    for loc in locations or []:
        can = loc.get("canonical")
        if can:
            work = re.sub(rf"\b{re.escape(can)}(?:['’]?[a-zçğıöşü]+)?\b", " ", work)

    stop_match = re.search(
        r"\b(?:arttı|azaldı|yükseldi|düştü|büyüdü|geriledi|oldu|yer\s+aldı|"
        r"ulaştı|açtı|açıldı|kurdu|kuruldu|kazandı|kaybetti|gerçekleşti)\b",
        work,
    )
    before = work[:stop_match.start()] if stop_match else work
    subject_noise = {"yılında", "yılda", "yıla", "göre", "önceki", "geçen", "yüzde", "oranında"}
    toks = [
        t for t in icerik_tokenlari(before)
        if not re.fullmatch(r"\d+(?:[.,]\d+)?", t) and t not in subject_noise
    ]
    if not toks:
        return None
    phrase = " ".join(toks[-6:])
    if len(phrase) < 3:
        return None
    return {
        "text": phrase,
        "canonical": phrase,
        "type": "PHRASE",
        "role": "SUBJECT",
        "position": 0,
        "end_position": len(phrase),
        "is_named": False,
    }


def subject_cikar(metin, entities=None, locations=None):
    entities = entities if entities is not None else entity_adaylarini_cikar(metin)
    locations = locations if locations is not None else lokasyonlari_cikar(metin, entities)

    non_location = [e for e in entities if e.get("role") != "LOCATION"]
    if non_location:
        e = sorted(non_location, key=lambda x: x.get("position", 10**9))[0]
        out = dict(e)
        out["role"] = "SUBJECT"
        return out

    return _fallback_phrase_subject(metin, locations)


def ranking_scope_hint_cikar(metin):
    normal = metni_normalize_et(metin)
    hits = []
    for label, patterns in RANKING_SCOPE_HINT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, normal):
                hits.append(label)
                break
    unique = []
    for h in hits:
        if h not in unique:
            unique.append(h)
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        return "AMBIGUOUS:" + "+".join(unique)
    return "UNKNOWN"


def comparison_operator_cikar(metin):
    normal = metni_normalize_et(metin)
    for label, patterns in COMPARISON_PATTERNS:
        if any(re.search(p, normal) for p in patterns):
            return label
    return None


def change_direction_cikar(metin):
    normal = metni_normalize_et(metin)
    up = any(re.search(p, normal) for p in UP_PATTERNS)
    down = any(re.search(p, normal) for p in DOWN_PATTERNS)
    if up and not down:
        return "UP"
    if down and not up:
        return "DOWN"
    if up and down:
        return "MIXED"
    return None


def negation_cikar(metin):
    normal = metni_normalize_et(metin)
    return any(re.search(p, normal) for p in NEGATION_PATTERNS)


def modality_cikar(metin):
    normal = metni_normalize_et(metin)
    if any(re.search(p, normal) for p in FUTURE_PATTERNS):
        return "FUTURE_OR_PREDICTION"
    return "ASSERTED"


def time_scope_cikar(metin):
    normal = metni_normalize_et(metin)
    years = yillari_cikar(normal)
    partial_markers = [
        r"\bilk\s+\d+\s+ay(?:da|ında|inde|lık|lik)?\b",
        r"\bilk\s+(?:üç|dört|beş|altı|yedi|sekiz|dokuz|on|on bir)\s+ay(?:da|ında|inde|lık|lik)?\b",
        r"\bilk\s+yarı(?:da|sında|sinde)?\b",
        r"\bikinci\s+yarı(?:da|sında|sinde)?\b",
        r"\bçeyrek(?:te|inde|in|lik)?\b",
        r"\bocak\s*[-–]\s*(?:aralık|kasım|ekim|eylül|ağustos|temmuz|haziran)\b",
    ]
    if any(re.search(p, normal) for p in partial_markers):
        return "PARTIAL_PERIOD"
    if any(re.search(rf"\b{re.escape(ay)}\b", normal) for ay in AYLAR):
        return "PARTIAL_PERIOD"
    if years:
        return "FULL_YEAR_OR_UNSPECIFIED_WITHIN_YEAR"
    return "UNKNOWN"


def _context_metadata(metin, position, entities, locations, window=180):
    normal = metni_normalize_et(metin)
    left, right = _sentence_span(normal, position)
    # Slight expansion helps when the entity is immediately before a sentence fragment.
    left2 = max(0, left - 45)
    right2 = min(len(normal), right + 45)
    context = normal[left2:right2].strip()

    local_entities = []
    # Binding must stay inside the actual sentence. The expanded context is only
    # for topical semantics; otherwise entities from a neighboring sentence can
    # steal a rank/number constraint.
    for e in entities or []:
        p = e.get("position", -10**9)
        if left <= p <= right and e.get("role") != "LOCATION":
            local_entities.append(e)

    local_locations = []
    for loc in locations or []:
        p = loc.get("position")
        if p is None or left <= p <= right:
            if loc.get("canonical") in context:
                local_locations.append(loc)

    preceding = [e for e in local_entities if e.get("position", 0) <= position]
    primary_entity = None
    if preceding:
        primary_entity = min(preceding, key=lambda e: position - e.get("position", 0))
    elif local_entities:
        primary_entity = min(local_entities, key=lambda e: abs(position - e.get("position", 0)))

    return {
        "context": context,
        "bound_entities": [e.get("canonical") for e in local_entities if e.get("canonical")],
        "bound_entity_details": local_entities,
        "primary_bound_entity": primary_entity.get("canonical") if primary_entity else None,
        "context_locations": [l.get("canonical") for l in local_locations if l.get("canonical")],
        "context_location_details": local_locations,
        "context_years": yillari_cikar(context),
        "context_scope_hint": ranking_scope_hint_cikar(context),
        "context_change_direction": change_direction_cikar(context),
        "context_negated": negation_cikar(context),
        "context_modality": modality_cikar(context),
    }


def rank_adaylarini_cikar(metin, entities=None, locations=None):
    normal = metni_normalize_et(metin)
    entities = entities if entities is not None else entity_adaylarini_cikar(metin)
    locations = locations if locations is not None else lokasyonlari_cikar(metin, entities)
    adaylar = []

    for ifade, rank in sorted(ORDINAL_KELIME_HARITASI.items(), key=lambda x: len(x[0]), reverse=True):
        p = rf"(?<![a-zçğıöşü]){re.escape(ifade)}(?![a-zçğıöşü])"
        for m in re.finditer(p, normal):
            meta = _context_metadata(metin, m.start(), entities, locations)
            adaylar.append({
                "rank": rank, "trigger": ifade, "source": "word_ordinal",
                "context": meta["context"], "position": m.start(), **meta,
            })

    numeric_patterns = [
        r"(?<!\d)(\d{1,3})\s*\.\s*(?:sıra|sırada|sıraya|sıranın)\b",
        r"(?<!\d)(\d{1,3})\s*['’]?(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü)\b",
    ]
    for p in numeric_patterns:
        for m in re.finditer(p, normal):
            try:
                rank = int(m.group(1))
            except Exception:
                continue
            meta = _context_metadata(metin, m.start(), entities, locations)
            adaylar.append({
                "rank": rank, "trigger": m.group(0), "source": "numeric_ordinal",
                "context": meta["context"], "position": m.start(), **meta,
            })

    for p in SUPERLATIVE_RANK1_PATTERNS:
        for m in re.finditer(p, normal):
            trigger = m.group(0)
            if re.search(r"\bilk\s+(?:kez|defa|yarı|çeyrek|ay|hafta|gün|model|ürün)\b", trigger):
                continue
            meta = _context_metadata(metin, m.start(), entities, locations)
            adaylar.append({
                "rank": 1, "trigger": trigger, "source": "superlative_rank1",
                "context": meta["context"], "position": m.start(), **meta,
            })

    unique, seen = [], set()
    for a in adaylar:
        key = (a["rank"], a["trigger"], a["position"])
        if key not in seen:
            unique.append(a)
            seen.add(key)
    return unique


def yuzde_adaylarini_cikar(metin, entities=None, locations=None):
    normal = metni_normalize_et(metin)
    entities = entities if entities is not None else entity_adaylarini_cikar(metin)
    locations = locations if locations is not None else lokasyonlari_cikar(metin, entities)
    out = []
    patterns = [r"%\s*(\d+(?:[.,]\d+)?)", r"\byüzde\s+(\d+(?:[.,]\d+)?)"]
    seen = set()
    for p in patterns:
        for m in re.finditer(p, normal):
            try:
                value = float(m.group(1).replace(",", "."))
            except Exception:
                continue
            key = (round(value, 9), m.start())
            if key in seen:
                continue
            seen.add(key)
            meta = _context_metadata(metin, m.start(), entities, locations)
            out.append({
                "value": value,
                "trigger": m.group(0),
                "position": m.start(),
                **meta,
            })
    return out


def diger_sayi_adaylarini_cikar(metin, entities=None, locations=None):
    normal = metni_normalize_et(metin)
    entities = entities if entities is not None else entity_adaylarini_cikar(metin)
    locations = locations if locations is not None else lokasyonlari_cikar(metin, entities)
    tmp = re.sub(r"%\s*\d+(?:[.,]\d+)?", " ", normal)
    tmp = re.sub(r"\byüzde\s+\d+(?:[.,]\d+)?", " ", tmp)
    tmp = re.sub(r"(?<!\d)(?:19\d{2}|20\d{2}|21\d{2})(?!\d)", " ", tmp)
    out = []
    for m in re.finditer(r"(?<!\w)(\d+(?:[.,]\d+)?)(?!\w)", tmp):
        after = tmp[m.end(): m.end() + 20]
        if re.match(r"\s*\.\s*(?:sıra|sırada|sıraya)", after):
            continue
        try:
            value = float(m.group(1).replace(",", "."))
        except Exception:
            continue
        meta = _context_metadata(metin, m.start(), entities, locations)
        out.append({"value": value, "position": m.start(), **meta})
        if len(out) >= 16:
            break
    return out


def diger_sayilari_cikar(metin):
    vals = []
    for c in diger_sayi_adaylarini_cikar(metin):
        v = c["value"]
        if not any(abs(v - u) < 1e-9 for u in vals):
            vals.append(v)
        if len(vals) >= 12:
            break
    return vals


def primary_rank_for_claim(claim_text, rank_candidates, subject=None):
    if not rank_candidates:
        return None
    source_priority = {"word_ordinal": 4, "numeric_ordinal": 4, "superlative_rank1": 2}
    subj = (subject or {}).get("canonical")
    def key(x):
        bound = x.get("bound_entities") or []
        subject_bonus = 1 if subj and subj in bound else 0
        return (
            subject_bonus,
            source_priority.get(x.get("source"), 0),
            token_overlap(claim_text, x.get("context", "")),
            -x.get("position", 0),
        )
    return max(rank_candidates, key=key)


def primary_percentage_for_claim(candidates, subject=None):
    if not candidates:
        return None
    subj = (subject or {}).get("canonical")
    def key(c):
        return (
            1 if subj and subj in (c.get("bound_entities") or []) else 0,
            1 if c.get("context_change_direction") else 0,
            -c.get("position", 0),
        )
    return max(candidates, key=key)


def predicate_type_cikar(metin, primary_rank=None, comparison=None, change=None, percentages=None, numbers=None):
    normal = metni_normalize_et(metin)
    if primary_rank is not None:
        return "RANK"
    if comparison is not None:
        return "COMPARISON"
    if change is not None:
        return "CHANGE"
    if any(re.search(p, normal) for p in PREDICATE_EVENT_HINTS):
        return "EVENT"
    if percentages or numbers:
        return "VALUE"
    if re.search(r"\b(?:var|yok|mevcut|bulunuyor|bulunmuyor)\b", normal):
        return "EXISTENCE"
    return "OTHER"


def topic_tokens_cikar(metin, entities=None, locations=None):
    tokens = icerik_tokenlari(metin)
    remove = set()
    for e in entities or []:
        remove.update(tokenlara_ayir(e.get("canonical", "")))
    for l in locations or []:
        remove.update(tokenlara_ayir(l.get("canonical", "")))
    remove.update(str(y) for y in yillari_cikar(metin))
    remove.update(ORDINAL_KELIME_HARITASI.keys())
    generic = {
        "yılında", "yıla", "göre", "önceki", "geçen", "sırada", "yer", "aldı",
        "yüzde", "şirket", "şirketin", "şirketinin", "marka", "markası", "markasının",
    }
    out = []
    for t in tokens:
        if t in remove or t in generic or re.fullmatch(r"\d+(?:[.,]\d+)?", t):
            continue
        if t not in out:
            out.append(t)
        if len(out) >= 18:
            break
    return out


def structured_schema_cikar(metin, source="claim"):
    entities = entity_adaylarini_cikar(metin)
    locations = lokasyonlari_cikar(metin, entities)
    subject = subject_cikar(metin, entities, locations)
    ranks = rank_adaylarini_cikar(metin, entities, locations)
    percentages_c = yuzde_adaylarini_cikar(metin, entities, locations)
    number_c = diger_sayi_adaylarini_cikar(metin, entities, locations)
    primary_rank = primary_rank_for_claim(metin, ranks, subject) if source == "claim" else None
    primary_percentage = primary_percentage_for_claim(percentages_c, subject) if source == "claim" else None
    comparison = comparison_operator_cikar(metin)
    change = change_direction_cikar(metin)
    percentages = []
    for c in percentages_c:
        if not any(abs(c["value"] - v) < 1e-9 for v in percentages):
            percentages.append(c["value"])
    numbers = []
    for c in number_c:
        if not any(abs(c["value"] - v) < 1e-9 for v in numbers):
            numbers.append(c["value"])
        if len(numbers) >= 12:
            break
    predicate = predicate_type_cikar(metin, primary_rank, comparison, change, percentages, numbers)
    topic_tokens = topic_tokens_cikar(metin, entities, locations)

    schema = {
        "schema_version": "v5_1_generic_frame_1",
        "source": source,
        "entities": entities,
        "subject": subject,
        "locations": locations,
        "predicate_type": predicate,
        "topic_tokens": topic_tokens,
        "years": yillari_cikar(metin),
        "percentages": percentages,
        "percentage_candidates": percentages_c,
        "primary_percentage": primary_percentage,
        "numbers": numbers,
        "number_candidates": number_c,
        "rank_candidates": ranks,
        "primary_rank": primary_rank,
        "ranking_scope_hint": ranking_scope_hint_cikar(metin),
        "comparison_operator": comparison,
        "change_direction": change,
        "negated": negation_cikar(metin),
        "modality": modality_cikar(metin),
        "time_scope": time_scope_cikar(metin),
        "relative_time": goreli_zaman_bilgisi(metin),
    }
    schema["fact_frame"] = {
        "subject": subject,
        "predicate": predicate,
        "locations": locations,
        "years": schema["years"],
        "time_scope": schema["time_scope"],
        "topic_tokens": topic_tokens,
        "rank": primary_rank,
        "percentage": primary_percentage,
        "comparison_operator": comparison,
        "change_direction": change,
        "negated": schema["negated"],
        "modality": schema["modality"],
    }
    return schema


def _merge_unique_dicts(base, extra, key="canonical"):
    out = [dict(x) for x in (base or [])]
    seen = {x.get(key) for x in out if x.get(key) is not None}
    for x in extra or []:
        k = x.get(key)
        if k not in seen:
            out.append(dict(x))
            seen.add(k)
    return out


def context_schema_merge(schema, inherited_context):
    """Fill missing semantic slots from the parent atomic-claim context."""
    if not inherited_context:
        return schema
    merged = dict(schema)
    if not merged.get("subject") and inherited_context.get("subject"):
        merged["subject"] = dict(inherited_context["subject"])
    if not merged.get("locations") and inherited_context.get("locations"):
        merged["locations"] = [dict(x) for x in inherited_context["locations"]]
    if not merged.get("years") and inherited_context.get("years"):
        merged["years"] = list(inherited_context["years"])
    merged["entities"] = _merge_unique_dicts(
        merged.get("entities"),
        [inherited_context["subject"]] if inherited_context.get("subject") else [],
    )
    merged["context_inherited"] = {
        "subject": bool(inherited_context.get("subject") and not schema.get("subject")),
        "locations": bool(inherited_context.get("locations") and not schema.get("locations")),
        "years": bool(inherited_context.get("years") and not schema.get("years")),
    }
    if merged.get("fact_frame"):
        ff = dict(merged["fact_frame"])
        ff["subject"] = merged.get("subject")
        ff["locations"] = merged.get("locations")
        ff["years"] = merged.get("years")
        merged["fact_frame"] = ff
    return merged


def external_schema_merge(fallback_schema, external_schema):
    if not external_schema:
        return fallback_schema
    merged = dict(fallback_schema)
    for key, value in external_schema.items():
        if value is not None:
            if key in {"entities", "locations"} and isinstance(value, list):
                merged[key] = _merge_unique_dicts(merged.get(key), value)
            elif key == "fact_frame" and isinstance(value, dict):
                ff = dict(merged.get("fact_frame") or {})
                ff.update(value)
                merged[key] = ff
            else:
                merged[key] = value
    merged["schema_source"] = "external_plus_fallback"
    return merged



def _frame_entity(value, role="ENTITY", entity_type="UNKNOWN"):
    """Normalize a future Person-1 ClaimFrame entity into V5.2's local schema."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return {
            "text": text,
            "canonical": metni_normalize_et(text),
            "type": entity_type,
            "role": role,
            "position": -1,
            "end_position": -1,
            "is_named": True,
        }
    if isinstance(value, dict):
        text = (value.get("text") or value.get("name") or value.get("value") or "").strip()
        if not text:
            return None
        return {
            "text": text,
            "canonical": value.get("canonical") or metni_normalize_et(text),
            "type": value.get("type") or entity_type,
            "role": value.get("role") or role,
            "position": value.get("position", -1),
            "end_position": value.get("end_position", -1),
            "is_named": value.get("is_named", True),
        }
    return None


def claim_frame_to_schema(claim_text, claim_frame=None):
    """Backward-compatible adapter for a richer future ClaimFrame.

    Today Person 1 returns only Claim{id,text,is_verifiable}; in that case this
    function simply returns the generic local fallback schema. If Person 1 later
    adds subject/location/time/predicate/constraints, retrieval can consume them
    without changing Person 3's current `search(claim.text)` integration.
    """
    fallback = structured_schema_cikar(claim_text, source="claim")
    if not isinstance(claim_frame, dict) or not claim_frame:
        return fallback

    ext = {}

    subject = _frame_entity(claim_frame.get("subject"), role="SUBJECT")
    if subject:
        ext["subject"] = subject
        ext.setdefault("entities", []).append({**subject, "role": "ENTITY"})

    raw_locations = claim_frame.get("locations")
    if raw_locations is None and claim_frame.get("location") is not None:
        raw_locations = [claim_frame.get("location")]
    elif raw_locations is not None and not isinstance(raw_locations, list):
        raw_locations = [raw_locations]

    locs = []
    for loc in raw_locations or []:
        ent = _frame_entity(loc, role="LOCATION", entity_type="LOCATION")
        if ent:
            locs.append({
                "text": ent["text"],
                "canonical": ent["canonical"],
                "type": "LOCATION",
                "position": ent.get("position", -1),
            })
    if locs:
        ext["locations"] = locs

    # Time may be represented as {year: 2024}, {years: [...]}, or top-level years.
    years = []
    raw_years = claim_frame.get("years") or []
    if isinstance(raw_years, (int, str)):
        raw_years = [raw_years]
    for y in raw_years:
        try:
            yi = int(y)
            if 1900 <= yi <= 2100:
                years.append(yi)
        except (TypeError, ValueError):
            pass
    time_obj = claim_frame.get("time") or {}
    if isinstance(time_obj, dict):
        for key in ("year", "start_year", "end_year"):
            if time_obj.get(key) is not None:
                try:
                    yi = int(time_obj[key])
                    if 1900 <= yi <= 2100:
                        years.append(yi)
                except (TypeError, ValueError):
                    pass
        if time_obj.get("scope"):
            ext["time_scope"] = str(time_obj["scope"]).upper()
    if years:
        ext["years"] = sorted(set(years))

    pred = claim_frame.get("predicate") or claim_frame.get("predicate_type")
    if pred:
        ext["predicate_type"] = str(pred).upper()

    # Common future fields can enrich search even if local extraction missed them.
    metric = claim_frame.get("metric") or claim_frame.get("topic")
    if metric:
        toks = icerik_tokenlari(str(metric))
        ext["topic_tokens"] = _unique_keep_order((fallback.get("topic_tokens") or []) + toks)

    percentages = []
    numbers = []
    rank_value = None
    change_direction = claim_frame.get("change_direction")
    comparison_operator = claim_frame.get("comparison_operator")

    rank_obj = claim_frame.get("rank") or claim_frame.get("ranking")
    if isinstance(rank_obj, dict):
        rank_value = rank_obj.get("value", rank_obj.get("rank"))
    elif rank_obj is not None:
        rank_value = rank_obj

    constraints = claim_frame.get("constraints") or []
    if isinstance(constraints, dict):
        constraints = [constraints]
    for c in constraints:
        if not isinstance(c, dict):
            continue
        ctype = str(c.get("type") or "").upper()
        value = c.get("value")
        if "PERCENT" in ctype:
            try:
                percentages.append(float(value))
            except (TypeError, ValueError):
                pass
            change_direction = change_direction or c.get("direction")
        elif "RANK" in ctype:
            rank_value = value if value is not None else rank_value
        elif value is not None:
            try:
                numbers.append(float(value))
            except (TypeError, ValueError):
                pass

    # Also accept a direct value object: {type: percentage, value: 20, direction: UP}.
    value_obj = claim_frame.get("value")
    if isinstance(value_obj, dict):
        vtype = str(value_obj.get("type") or "").upper()
        if "PERCENT" in vtype:
            try:
                percentages.append(float(value_obj.get("value")))
            except (TypeError, ValueError):
                pass
            change_direction = change_direction or value_obj.get("direction")

    if percentages:
        ext["percentages"] = sorted(set(percentages))
    if numbers:
        ext["numbers"] = sorted(set(numbers))
    if change_direction:
        ext["change_direction"] = str(change_direction).upper()
    if comparison_operator:
        ext["comparison_operator"] = str(comparison_operator).upper()

    if rank_value is not None:
        try:
            rank_int = int(rank_value)
        except (TypeError, ValueError):
            rank_int = None
        if rank_int is not None:
            candidate = {
                "rank": rank_int,
                "trigger": str(rank_value),
                "source": "external_claim_frame",
                "context": claim_text,
                "position": -1,
                "bound_entities": [subject["canonical"]] if subject else [],
                "bound_entity_details": [subject] if subject else [],
                "primary_bound_entity": subject["canonical"] if subject else None,
                "context_locations": [x["canonical"] for x in locs],
                "context_location_details": locs,
                "context_years": ext.get("years", []),
                "context_scope_hint": claim_frame.get("ranking_scope") or fallback.get("ranking_scope_hint", "UNKNOWN"),
                "context_change_direction": ext.get("change_direction"),
                "context_negated": bool(fallback.get("negated")),
                "context_modality": fallback.get("modality", "ASSERTED"),
            }
            ext["rank_candidates"] = [candidate]
            ext["primary_rank"] = candidate

    merged = external_schema_merge(fallback, ext)
    merged["schema_source"] = "claim_frame_plus_fallback"
    return merged


# ==========================================================
# GENERIC CONSTRAINT + SEMANTIC FRAME ALIGNMENT (V5.2)
# ==========================================================


def _numeric_relation(claim_values, evidence_values, near_abs=None, near_rel=0.05):
    if not claim_values:
        return {"status": "NOT_APPLICABLE", "claim_values": [], "evidence_values": evidence_values, "best_pair": None}
    if not evidence_values:
        return {"status": "MISSING", "claim_values": claim_values, "evidence_values": [], "best_pair": None}
    best = None
    for c in claim_values:
        for e in evidence_values:
            diff = abs(float(c) - float(e))
            scale = max(abs(float(c)), 1.0)
            rel = diff / scale
            cand = (diff, rel, c, e)
            if best is None or cand[:2] < best[:2]:
                best = cand
    diff, rel, c, e = best
    if diff <= 1e-9:
        status = "SAME"
    else:
        abs_limit = near_abs if near_abs is not None else max(0.5, abs(float(c)) * 0.01)
        status = "NEAR" if (diff <= abs_limit or rel <= near_rel) else "DIFFERENT"
    return {
        "status": status,
        "claim_values": claim_values,
        "evidence_values": evidence_values,
        "best_pair": {"claim": c, "evidence": e, "abs_diff": round(diff, 6), "relative_diff": round(rel, 6)},
    }


def _year_relation(claim_years, evidence_years):
    if not claim_years:
        return "NOT_APPLICABLE"
    if not evidence_years:
        return "MISSING"
    if set(claim_years) & set(evidence_years):
        return "SAME_OR_OVERLAP"
    return "DIFFERENT"


def _time_scope_relation(claim_scope, evidence_scope, year_relation):
    if claim_scope == "UNKNOWN":
        return "NOT_APPLICABLE"
    if evidence_scope == "UNKNOWN":
        return "UNKNOWN"
    if year_relation == "DIFFERENT":
        return "DIFFERENT"
    if claim_scope == evidence_scope:
        return "SAME"
    if claim_scope == "FULL_YEAR_OR_UNSPECIFIED_WITHIN_YEAR" and evidence_scope == "PARTIAL_PERIOD":
        return "PARTIAL"
    if claim_scope == "PARTIAL_PERIOD" and evidence_scope == "FULL_YEAR_OR_UNSPECIFIED_WITHIN_YEAR":
        return "BROADER_EVIDENCE"
    return "UNKNOWN"


def _canon_set(items):
    out = set()
    for x in items or []:
        if isinstance(x, dict):
            v = x.get("canonical") or x.get("text")
        else:
            v = x
        if v:
            out.add(metni_normalize_et(str(v)))
    return out


def _subject_relation(claim_schema, evidence_schema, local_bound=None, local_context=None):
    subj = claim_schema.get("subject") or {}
    can = metni_normalize_et(subj.get("canonical") or subj.get("text") or "")
    if not can:
        return "NOT_APPLICABLE"

    if subj.get("is_named") is False or subj.get("type") == "PHRASE":
        overlap = token_overlap(subj.get("canonical", ""), local_context or "")
        return "SAME" if overlap >= 0.5 else "UNKNOWN"

    if local_bound is not None:
        local = {metni_normalize_et(x) for x in local_bound if x}
        if can in local:
            return "SAME"
        if local:
            return "DIFFERENT"
        if local_context and can in metni_normalize_et(local_context):
            return "SAME"
        return "MISSING"

    ev_entities = _canon_set(evidence_schema.get("entities"))
    if can in ev_entities:
        return "SAME"
    if ev_entities:
        return "DIFFERENT"
    if can in metni_normalize_et(local_context or ""):
        return "SAME"
    return "MISSING"


def _location_relation(claim_schema, evidence_schema, local_locations=None):
    claim_locs = _canon_set(claim_schema.get("locations"))
    if not claim_locs:
        return "NOT_APPLICABLE"
    ev_locs = {metni_normalize_et(x) for x in (local_locations or []) if x} if local_locations is not None else _canon_set(evidence_schema.get("locations"))
    if not ev_locs:
        return "MISSING"
    if claim_locs & ev_locs:
        return "SAME"
    return "DIFFERENT"


def _scope_relation(claim_scope, evidence_scope):
    if not claim_scope or claim_scope == "UNKNOWN":
        return "NOT_APPLICABLE"
    if not evidence_scope or evidence_scope == "UNKNOWN":
        return "MISSING"
    cset = set(claim_scope.replace("AMBIGUOUS:", "").split("+"))
    eset = set(evidence_scope.replace("AMBIGUOUS:", "").split("+"))
    return "SAME" if cset & eset else "DIFFERENT"


def _predicate_relation(claim_schema, evidence_schema):
    cp = claim_schema.get("predicate_type") or "OTHER"
    ep = evidence_schema.get("predicate_type") or "OTHER"
    if cp == "OTHER":
        return "NOT_APPLICABLE"
    if cp == ep:
        return "SAME"
    # Rank evidence may be present even if whole-passage predicate is mixed.
    if cp == "RANK" and evidence_schema.get("rank_candidates"):
        return "SAME"
    if cp == "CHANGE" and evidence_schema.get("change_direction"):
        return "SAME"
    return "DIFFERENT" if ep != "OTHER" else "MISSING"


def _relation_quality(status, neutral=0.5):
    return {
        "SAME": 1.0, "SAME_OR_OVERLAP": 1.0,
        "NOT_APPLICABLE": 0.72,
        "PARTIAL": 0.78, "BROADER_EVIDENCE": 0.7,
        "UNKNOWN": neutral, "MISSING": 0.42,
        "DIFFERENT": 0.05,
    }.get(status, neutral)


def _candidate_binding_score(claim_text, claim_schema, candidate):
    subject_rel = _subject_relation(
        claim_schema, {},
        local_bound=candidate.get("bound_entities") or [],
        local_context=candidate.get("context", ""),
    )
    loc_rel = _location_relation(
        claim_schema, {},
        local_locations=candidate.get("context_locations") or [],
    )
    year_rel = _year_relation(
        claim_schema.get("years", []),
        candidate.get("context_years") or [],
    )
    scope_rel = _scope_relation(
        claim_schema.get("ranking_scope_hint", "UNKNOWN"),
        candidate.get("context_scope_hint", "UNKNOWN"),
    )
    topic = " ".join(claim_schema.get("topic_tokens") or [])
    lexical = token_overlap(topic or claim_text, candidate.get("context", ""))

    score = (
        0.42 * _relation_quality(subject_rel)
        + 0.20 * _relation_quality(loc_rel, 0.55)
        + 0.14 * _relation_quality(year_rel, 0.55)
        + 0.10 * _relation_quality(scope_rel, 0.6)
        + 0.14 * lexical
    )
    return score, {
        "subject_relation": subject_rel,
        "location_relation": loc_rel,
        "year_relation": year_rel,
        "scope_relation": scope_rel,
        "topic_overlap": round(lexical, 4),
    }


def _best_evidence_rank_candidate(claim_text, claim_schema, evidence_schema):
    candidates = evidence_schema.get("rank_candidates") or []
    if not candidates:
        return None
    scored = []
    for c in candidates:
        score, binding = _candidate_binding_score(claim_text, claim_schema, c)
        explicit = 0.04 if c.get("source") in {"word_ordinal", "numeric_ordinal"} else 0.0
        scored.append((score + explicit, c, binding))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best, binding = scored[0]
    out = dict(best)
    out["binding_score"] = round(best_score, 4)
    out["binding"] = binding
    return out


def _best_bound_numeric_candidate(claim_text, claim_schema, claim_candidate, evidence_candidates):
    if not evidence_candidates:
        return None
    scored = []
    for c in evidence_candidates:
        score, binding = _candidate_binding_score(claim_text, claim_schema, c)
        # Direction is relevance context, not truth value. SAME gets a small bonus,
        # opposite direction remains retrievable as potential counter-evidence.
        cc = claim_candidate.get("context_change_direction") if claim_candidate else claim_schema.get("change_direction")
        ec = c.get("context_change_direction")
        if cc and ec:
            score += 0.05 if cc == ec else 0.025
        scored.append((score, c, binding))
    scored.sort(key=lambda x: x[0], reverse=True)
    score, c, binding = scored[0]
    out = dict(c)
    out["binding_score"] = round(score, 4)
    out["binding"] = binding
    return out


def _percentage_alignment(claim_text, claim_schema, evidence_schema):
    claim_candidates = claim_schema.get("percentage_candidates") or []
    evidence_candidates = evidence_schema.get("percentage_candidates") or []
    if not claim_candidates:
        return _numeric_relation([], evidence_schema.get("percentages", []), near_abs=2.0, near_rel=0.05)

    bindings = []
    statuses = []
    for cc in claim_candidates:
        ec = _best_bound_numeric_candidate(claim_text, claim_schema, cc, evidence_candidates)
        if ec is None:
            bindings.append({"claim": cc, "evidence": None, "value_relation": "MISSING", "context_relation": "MISSING"})
            statuses.append("MISSING")
            continue
        num = _numeric_relation([cc["value"]], [ec["value"]], near_abs=2.0, near_rel=0.05)
        b = ec.get("binding") or {}
        context_bad = any(b.get(k) == "DIFFERENT" for k in ["subject_relation", "location_relation", "year_relation"])
        context_relation = "MISMATCH" if context_bad else "COMPATIBLE"
        bindings.append({
            "claim": cc,
            "evidence": ec,
            "value_relation": num.get("status"),
            "context_relation": context_relation,
            "binding": b,
        })
        statuses.append("CONTEXT_MISMATCH" if context_bad else num.get("status"))

    if all(x in {"SAME", "NEAR"} for x in statuses):
        summary = "SAME_OR_NEAR"
    elif any(x == "CONTEXT_MISMATCH" for x in statuses):
        summary = "CONTEXT_MISMATCH"
    elif any(x == "DIFFERENT" for x in statuses):
        summary = "DIFFERENT"
    elif any(x == "MISSING" for x in statuses):
        summary = "MISSING"
    else:
        summary = statuses[0] if statuses else "NOT_APPLICABLE"

    return {
        "status": summary,
        "claim_values": [x["value"] for x in claim_candidates],
        "evidence_values": [x["value"] for x in evidence_candidates],
        "bindings": bindings,
        "best_pair": None,
    }


def constraint_alignment(claim_text, claim_schema, evidence_text, evidence_schema=None):
    evidence_schema = evidence_schema or structured_schema_cikar(evidence_text, source="evidence")

    subject_rel = _subject_relation(claim_schema, evidence_schema, local_context=evidence_text)
    location_rel = _location_relation(claim_schema, evidence_schema)
    predicate_rel = _predicate_relation(claim_schema, evidence_schema)
    year_rel = _year_relation(claim_schema.get("years", []), evidence_schema.get("years", []))
    time_scope_rel = _time_scope_relation(
        claim_schema.get("time_scope", "UNKNOWN"),
        evidence_schema.get("time_scope", "UNKNOWN"),
        year_rel,
    )

    percentage_rel = _percentage_alignment(claim_text, claim_schema, evidence_schema)
    number_rel = _numeric_relation(
        claim_schema.get("numbers", []), evidence_schema.get("numbers", []),
        near_abs=None, near_rel=0.03,
    )

    claim_rank_obj = claim_schema.get("primary_rank")
    best_ev_rank = _best_evidence_rank_candidate(claim_text, claim_schema, evidence_schema)
    claim_rank_scope = claim_schema.get("ranking_scope_hint", "UNKNOWN")
    evidence_rank_scope = (best_ev_rank or {}).get("context_scope_hint") or evidence_schema.get("ranking_scope_hint", "UNKNOWN")

    rank_value_rel = "NOT_APPLICABLE"
    rank_binding_rel = "NOT_APPLICABLE"
    if claim_rank_obj is None:
        rank_rel = "NOT_APPLICABLE"
    elif best_ev_rank is None:
        rank_rel = "MISSING"
        rank_value_rel = "MISSING"
        rank_binding_rel = "MISSING"
    else:
        b = best_ev_rank.get("binding") or {}
        rank_value_rel = "SAME" if claim_rank_obj.get("rank") == best_ev_rank.get("rank") else "DIFFERENT"
        if b.get("subject_relation") == "DIFFERENT":
            rank_binding_rel, rank_rel = "ENTITY_MISMATCH", "ENTITY_MISMATCH"
        elif b.get("location_relation") == "DIFFERENT":
            rank_binding_rel, rank_rel = "LOCATION_MISMATCH", "LOCATION_MISMATCH"
        elif b.get("year_relation") == "DIFFERENT":
            rank_binding_rel, rank_rel = "TIME_MISMATCH", "TIME_MISMATCH"
        elif b.get("scope_relation") == "DIFFERENT":
            rank_binding_rel, rank_rel = "SCOPE_MISMATCH", "SCOPE_MISMATCH"
        elif b.get("subject_relation") == "MISSING" and (claim_schema.get("subject") or {}).get("is_named"):
            rank_binding_rel, rank_rel = "ENTITY_MISSING", rank_value_rel
        elif b.get("location_relation") == "MISSING" and claim_schema.get("locations"):
            rank_binding_rel, rank_rel = "LOCATION_MISSING", rank_value_rel
        elif b.get("year_relation") == "MISSING" and claim_schema.get("years"):
            rank_binding_rel, rank_rel = "TIME_MISSING", rank_value_rel
        else:
            rank_binding_rel, rank_rel = "COMPATIBLE", rank_value_rel

    claim_comp = claim_schema.get("comparison_operator")
    ev_comp = evidence_schema.get("comparison_operator")
    if claim_comp is None:
        comp_rel = "NOT_APPLICABLE"
    elif ev_comp is None:
        comp_rel = "MISSING"
    else:
        comp_rel = "SAME" if claim_comp == ev_comp else "DIFFERENT"

    claim_change = claim_schema.get("change_direction")
    ev_change = evidence_schema.get("change_direction")
    if claim_change is None:
        change_rel = "NOT_APPLICABLE"
    elif ev_change is None:
        change_rel = "MISSING"
    else:
        change_rel = "SAME" if claim_change == ev_change else "DIFFERENT"

    neg_rel = "SAME" if bool(claim_schema.get("negated")) == bool(evidence_schema.get("negated")) else "DIFFERENT"
    modality_rel = "SAME" if claim_schema.get("modality") == evidence_schema.get("modality") else "DIFFERENT"

    constraint_checks = []
    if claim_schema.get("years"):
        constraint_checks.append(1.0 if evidence_schema.get("years") else 0.0)
    if claim_schema.get("locations"):
        constraint_checks.append(1.0 if evidence_schema.get("locations") else 0.0)
    if claim_schema.get("subject") and claim_schema.get("subject", {}).get("is_named"):
        constraint_checks.append(1.0 if subject_rel == "SAME" else 0.0)
    if claim_schema.get("percentages"):
        constraint_checks.append(1.0 if evidence_schema.get("percentages") else 0.0)
    if claim_schema.get("numbers"):
        constraint_checks.append(1.0 if evidence_schema.get("numbers") else 0.0)
    if claim_rank_obj is not None:
        constraint_checks.append(1.0 if best_ev_rank is not None else 0.0)
    if claim_comp is not None:
        constraint_checks.append(1.0 if ev_comp is not None else 0.0)
    if claim_change is not None:
        constraint_checks.append(1.0 if ev_change is not None else 0.0)
    directness = sum(constraint_checks) / len(constraint_checks) if constraint_checks else 0.5

    structural_components = {
        "subject": _relation_quality(subject_rel, 0.5),
        "location": _relation_quality(location_rel, 0.58),
        "predicate": _relation_quality(predicate_rel, 0.55),
        "time": _relation_quality(year_rel, 0.58),
        "time_scope": _relation_quality(time_scope_rel, 0.58),
        "directness": directness,
    }
    structural_relevance = (
        0.28 * structural_components["subject"]
        + 0.19 * structural_components["location"]
        + 0.16 * structural_components["predicate"]
        + 0.13 * structural_components["time"]
        + 0.08 * structural_components["time_scope"]
        + 0.16 * structural_components["directness"]
    )

    gate = 1.0
    if subject_rel == "DIFFERENT":
        gate *= 0.38
    elif subject_rel == "MISSING" and (claim_schema.get("subject") or {}).get("is_named"):
        gate *= 0.72
    if location_rel == "DIFFERENT":
        gate *= 0.52
    elif location_rel == "MISSING" and claim_schema.get("locations"):
        gate *= 0.82
    if predicate_rel == "DIFFERENT":
        gate *= 0.72
    if year_rel == "DIFFERENT":
        gate *= 0.72
    if rank_rel in {"ENTITY_MISMATCH", "LOCATION_MISMATCH", "TIME_MISMATCH", "SCOPE_MISMATCH"}:
        gate *= 0.65
    elif rank_binding_rel in {"ENTITY_MISSING", "LOCATION_MISSING", "TIME_MISSING"}:
        gate *= 0.80
    if percentage_rel.get("status") == "CONTEXT_MISMATCH":
        gate *= 0.65
    gate = max(0.12, min(1.0, gate))

    differing_signals = sum(
        1 for status in [
            percentage_rel.get("status"), number_rel.get("status"), rank_value_rel,
            comp_rel, change_rel, neg_rel, modality_rel,
        ] if status == "DIFFERENT"
    )
    context_mismatches = sum(
        1 for status in [subject_rel, location_rel, predicate_rel, year_rel, time_scope_rel, rank_rel, percentage_rel.get("status")]
        if status in {"DIFFERENT", "ENTITY_MISMATCH", "LOCATION_MISMATCH", "TIME_MISMATCH", "SCOPE_MISMATCH", "CONTEXT_MISMATCH"}
    )

    return {
        "subject_relation": subject_rel,
        "location_relation": location_rel,
        "predicate_relation": predicate_rel,
        "year_relation": year_rel,
        "percentage_relation": percentage_rel,
        "number_relation": number_rel,
        "rank_relation": rank_rel,
        "rank_value_relation": rank_value_rel,
        "rank_binding_relation": rank_binding_rel,
        "claim_rank": claim_rank_obj,
        "evidence_rank": best_ev_rank,
        "claim_ranking_scope_hint": claim_rank_scope,
        "evidence_ranking_scope_hint": evidence_rank_scope,
        "comparison_relation": comp_rel,
        "change_direction_relation": change_rel,
        "negation_relation": neg_rel,
        "modality_relation": modality_rel,
        "time_scope_relation": time_scope_rel,
        "constraint_directness": round(directness, 4),
        "structural_components": {k: round(v, 4) for k, v in structural_components.items()},
        "structural_relevance": round(structural_relevance, 4),
        "structural_gate": round(gate, 4),
        "differing_signal_count": differing_signals,
        "context_mismatch_count": context_mismatches,
        "note": (
            "Bu metadata retrieval içindir. SAME/DIFFERENT final doğruluk kararı değildir; "
            "entity/location/time/predicate bağlamı verifier'a ayrı taşınır."
        ),
    }


# ==========================================================
# CONTEXT-AWARE FALLBACK ATOMIC CLAIM DECOMPOSITION (V5.1)
# ==========================================================


def _parcada_yuklem_var_mi(parca):
    normal = metni_normalize_et(parca)
    tokens = tokenlara_ayir(normal)
    if not tokens:
        return False
    explicit = [
        "oldu", "olmadı", "arttı", "azaldı", "yükseldi", "düştü",
        "açtı", "açıldı", "kurdu", "kuruldu", "kazandı", "kaybetti",
        "yer aldı", "ulaştı", "gerçekleşti", "başladı", "bitti",
        "gönderdi", "yasakladı", "yasaklandı", "lider", "birinci",
    ]
    if any(x in normal for x in explicit):
        return True
    return any(VERBISH_SUFFIX_PATTERN.search(t) for t in tokens[-4:])


def _safe_and_split(text):
    if not text:
        return []
    matches = list(re.finditer(r"\s+ve\s+", text, flags=re.IGNORECASE))
    if not matches:
        return [text.strip()]
    for m in matches:
        sol = text[:m.start()].strip(" ,.;")
        sag = text[m.end():].strip(" ,.;")
        if len(tokenlara_ayir(sol)) >= 4 and len(tokenlara_ayir(sag)) >= 4:
            if _parcada_yuklem_var_mi(sol) and _parcada_yuklem_var_mi(sag):
                return [sol, sag]
    return [text.strip()]


def _raw_atomic_parts(iddia):
    text = re.sub(r"\s+", " ", iddia or "").strip()
    if not text:
        return []
    parts = re.split(r"\s*(?:;|\bancak\b|\bama\b|\bfakat\b|\boysa\b)\s*", text, flags=re.IGNORECASE)
    parts = [p.strip(" ,.;") for p in parts if p.strip(" ,.;")]
    out = []
    for p in parts:
        out.extend(_safe_and_split(p))
    final, seen = [], set()
    for p in out:
        key = metni_normalize_et(p)
        if key and key not in seen:
            final.append(p.strip())
            seen.add(key)
    return final or [text]


def _has_generic_anaphora(text):
    normal = metni_normalize_et(text)
    return any(re.search(p, normal) for p, _ in GENERIC_ANAPHORA_PATTERNS)


def _resolve_anaphora_text(text, subject_text):
    if not subject_text:
        return text, False
    for p, repl in GENERIC_ANAPHORA_PATTERNS:
        m = re.search(p, metni_normalize_et(text))
        if m and m.start() == 0:
            # Apply same-length-insensitive replacement to original via beginning token span.
            original_match = re.match(p, text, flags=re.IGNORECASE)
            if original_match:
                return repl.format(subject=subject_text) + text[original_match.end():], True
            # Fallback: replace normalized prefix length approximately.
            return repl.format(subject=subject_text) + " " + text, True
    return text, False


def _context_from_schema(schema):
    return {
        "subject": schema.get("subject"),
        "locations": schema.get("locations") or [],
        "years": schema.get("years") or [],
    }


def fallback_atomic_claim_frames(iddia):
    raw_parts = _raw_atomic_parts(iddia)
    frames = []
    parent_context = {"subject": None, "locations": [], "years": []}

    for i, raw in enumerate(raw_parts):
        raw_schema = structured_schema_cikar(raw, source="claim")
        inherited = {"subject": None, "locations": [], "years": []}
        resolved = raw

        explicit_subject = raw_schema.get("subject")
        has_named_subject = bool(explicit_subject and explicit_subject.get("is_named"))
        anaphoric = _has_generic_anaphora(raw)

        if i > 0 and parent_context.get("subject") and (anaphoric or not has_named_subject):
            inherited["subject"] = parent_context["subject"]
            subject_text = parent_context["subject"].get("text") or parent_context["subject"].get("canonical")
            resolved, rewritten = _resolve_anaphora_text(raw, subject_text)
            if not rewritten and subject_text:
                # Generic context prefix: search/schema sees the inherited entity without
                # pretending a language-specific possessive form is always known.
                resolved = f"{subject_text} — {raw}"

        if i > 0 and not raw_schema.get("locations") and parent_context.get("locations"):
            inherited["locations"] = parent_context["locations"]
        if i > 0 and not raw_schema.get("years") and parent_context.get("years"):
            inherited["years"] = parent_context["years"]

        resolved_schema = structured_schema_cikar(resolved, source="claim")
        resolved_schema = context_schema_merge(resolved_schema, inherited)

        frame = {
            "raw_text": raw,
            "resolved_text": resolved,
            "inherited_context": inherited,
            "schema": resolved_schema,
        }
        frames.append(frame)

        # Explicit new subject/location/year updates discourse context; inherited slots
        # otherwise remain available for later clauses.
        current = _context_from_schema(resolved_schema)
        if current.get("subject"):
            parent_context["subject"] = current["subject"]
        if current.get("locations"):
            parent_context["locations"] = current["locations"]
        if current.get("years"):
            parent_context["years"] = current["years"]

    return frames


def fallback_atomic_claims(iddia):
    """Backward-compatible string view of V5.1 contextual atomic claims."""
    return [x["resolved_text"] for x in fallback_atomic_claim_frames(iddia)]


# ==========================================================
# GENERIC QUERY GENERATION
# ==========================================================


def _unique_keep_order(items):
    out = []
    seen = set()
    for x in items:
        x = re.sub(r"\s+", " ", (x or "").strip())
        key = metni_normalize_et(x)
        if x and key not in seen:
            out.append(x)
            seen.add(key)
    return out


def sorgulari_uret(iddia, claim_schema=None):
    schema = claim_schema or structured_schema_cikar(iddia, source="claim")
    base = re.sub(r"\s+", " ", iddia).strip().rstrip(".?!")

    structured = []
    subject = schema.get("subject") or {}
    if subject.get("text"):
        structured.append(subject["text"])

    structured.extend(l.get("text") or l.get("canonical") for l in schema.get("locations", []) if (l.get("text") or l.get("canonical")))
    structured.extend(str(y) for y in schema.get("years", []))
    structured.extend(schema.get("topic_tokens", [])[:12])

    primary_rank = schema.get("primary_rank")
    if primary_rank and primary_rank.get("trigger"):
        structured.append(str(primary_rank["trigger"]))

    for v in schema.get("percentages", []):
        structured.append(f"yüzde {v:g}")

    rel = schema.get("relative_time") or {}
    if rel.get("has_relative_time"):
        if rel.get("start"):
            structured.append(rel["start"])
        if rel.get("end") and rel.get("end") != rel.get("start"):
            structured.append(rel["end"])

    structural_core = " ".join(_unique_keep_order([str(x) for x in structured if x]))
    lexical_core = " ".join(_unique_keep_order(icerik_tokenlari(base)[:22]))

    queries = [
        base,
        structural_core,
        f"{structural_core} {lexical_core}".strip(),
        f"{structural_core} resmi açıklama".strip(),
        f"{structural_core} haber doğrulama".strip(),
    ]
    return _unique_keep_order(queries)[:5]


# ==========================================================
# SEARCH PROVIDER ABSTRACTION
# ==========================================================


class TavilySearchProvider:
    name = "tavily"

    def search(self, query, max_results=HER_SORGU_ICIN_SONUC):
        client = get_tavily_client()
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_raw_content=False,
        )

        results = response.get("results", []) if isinstance(response, dict) else []
        out = []
        for r in results:
            url = (r.get("url") or "").strip()
            if not url:
                continue
            out.append({
                "title": (r.get("title") or "").strip(),
                "url": url,
                "content": (r.get("content") or "").strip(),
                "tavily_score": float(r.get("score") or 0.0),
                "published_date": r.get("published_date"),
            })
        return out


def search_provider_getir():
    provider = os.getenv("RETRIEVAL_SEARCH_PROVIDER", "tavily").strip().lower()
    if provider == "tavily":
        return TavilySearchProvider()
    raise RuntimeError(f"Desteklenmeyen RETRIEVAL_SEARCH_PROVIDER: {provider}")


def coklu_internet_aramasi(iddia, claim_schema=None):
    provider = search_provider_getir()
    queries = sorgulari_uret(iddia, claim_schema)

    print("\n====================================")
    print("ÜRETİLEN ARAMA SORGULARI")
    print("====================================")
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q}")

    all_results = []
    seen = set()

    for q in queries:
        try:
            results = provider.search(q, HER_SORGU_ICIN_SONUC)
        except Exception as exc:
            print(f"[UYARI] Arama hatası ({provider.name}): {type(exc).__name__}: {exc}")
            continue

        for r in results:
            canon = canonical_url(r["url"])
            if canon in seen:
                continue
            seen.add(canon)
            rr = dict(r)
            rr["query"] = q
            rr["provider"] = provider.name
            all_results.append(rr)

    return all_results


def arama_sonuclarini_filtrele(iddia, sonuclar):
    scored = []

    for r in sonuclar:
        url = r.get("url", "")
        if sosyal_medya_mi(url) or pdf_mi(url):
            continue

        haystack = f"{r.get('title', '')} {r.get('content', '')}"
        lexical = token_overlap(iddia, haystack)
        tavily_score = float(r.get("tavily_score") or 0.0)
        relevance = 0.45 * lexical + 0.55 * tavily_score

        rr = dict(r)
        rr["lexical_search_score"] = lexical
        rr["arama_benzerligi"] = relevance
        scored.append(rr)

    scored.sort(key=lambda x: x["arama_benzerligi"], reverse=True)
    return scored[:ON_FILTRE_LIMITI]


# ==========================================================
# WEB FETCHING
# ==========================================================


def html_metin_temizle(html):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup([
        "script", "style", "noscript", "svg", "canvas", "form",
        "nav", "footer", "header", "aside", "iframe",
    ]):
        tag.decompose()

    preferred = soup.find("article") or soup.find("main") or soup.body or soup
    text = preferred.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def requests_ile_metin_cek(url):
    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/124 Safari/537.36"
        )
    }

    try:
        r = requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        r.raise_for_status()

        content_type = (r.headers.get("Content-Type") or "").lower()
        if "html" not in content_type and "text" not in content_type:
            return ""

        return html_metin_temizle(r.text)
    except Exception:
        return ""


def playwright_ile_metin_cek(url):
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=PLAYWRIGHT_TIMEOUT)
            page.wait_for_timeout(1200)
            html = page.content()
            browser.close()
            return html_metin_temizle(html)
    except Exception:
        return ""


def web_sayfasindan_metin_cek(url, tavily_content=""):
    text = requests_ile_metin_cek(url)
    if len(text) >= REQUESTS_MIN_KARAKTER:
        return text, "requests"

    pw = playwright_ile_metin_cek(url)
    if len(pw) >= PLAYWRIGHT_MIN_KARAKTER:
        return pw, "playwright"

    fallback = re.sub(r"\s+", " ", tavily_content or "").strip()
    if len(fallback) >= TAVILY_FALLBACK_MIN_KARAKTER:
        return fallback, "tavily_snippet"

    if text:
        return text, "requests_short"
    if pw:
        return pw, "playwright_short"
    return "", None


# ==========================================================
# SENTENCE-AWARE CHUNKING
# ==========================================================


def cumlelere_ayir(metin):
    text = re.sub(r"\s+", " ", metin or "").strip()
    if not text:
        return []

    # Protect decimal/thousand separators before sentence splitting.
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", text)
    protected = re.sub(r"(?<=\d),(?=\d)", "<COMMA>", protected)

    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9])", protected)
    out = []
    for p in parts:
        p = p.replace("<DOT>", ".").replace("<COMMA>", ",").strip()
        if p:
            out.append(p)
    return out


def gurultulu_chunk_mi(metin):
    normal = metni_normalize_et(metin)
    if len(normal) < 60:
        return True

    noisy_markers = [
        "çerez", "cookie", "gizlilik politikası", "üyelik", "giriş yap",
        "reklam", "telif hakkı", "copyright", "abone ol",
    ]
    noisy_hits = sum(1 for x in noisy_markers if x in normal)
    if noisy_hits >= 3:
        return True

    alpha = sum(ch.isalpha() for ch in normal)
    if alpha / max(1, len(normal)) < 0.45:
        return True

    return False


def metni_parcala(metin):
    sentences = cumlelere_ayir(metin)
    if not sentences:
        return []

    chunks = []
    current = []
    current_words = 0

    for sentence in sentences:
        wc = len(sentence.split())

        if current and current_words + wc > CHUNK_MAX_KELIME:
            chunk_text = " ".join(current).strip()
            if not gurultulu_chunk_mi(chunk_text):
                chunks.append(chunk_text)

            overlap = current[-CHUNK_OVERLAP_CUMLE:] if CHUNK_OVERLAP_CUMLE > 0 else []
            current = list(overlap)
            current_words = sum(len(x.split()) for x in current)

        current.append(sentence)
        current_words += wc

        if current_words >= CHUNK_HEDEF_KELIME:
            chunk_text = " ".join(current).strip()
            if not gurultulu_chunk_mi(chunk_text):
                chunks.append(chunk_text)

            overlap = current[-CHUNK_OVERLAP_CUMLE:] if CHUNK_OVERLAP_CUMLE > 0 else []
            current = list(overlap)
            current_words = sum(len(x.split()) for x in current)

    if current:
        chunk_text = " ".join(current).strip()
        if not gurultulu_chunk_mi(chunk_text):
            chunks.append(chunk_text)

    # Deduplicate exact normalized chunks.
    out = []
    seen = set()
    for i, text in enumerate(chunks):
        key = metni_normalize_et(text)
        if key and key not in seen:
            out.append({"index": i, "metin": text})
            seen.add(key)

    return out



def evidence_focus_passage(claim_text, passage, claim_schema=None, max_sentences=3):
    """Select a short candidate-local sentence window instead of scoring a whole blob.

    This is deliberately generic: it rewards claim topic/entity/location/time/number cues,
    then keeps the winning sentence plus adjacent context. It reduces accidental binding of
    a rank/percentage belonging to another sentence or entity while preserving counter-evidence.
    """
    sentences = cumlelere_ayir(passage)
    if len(sentences) <= max_sentences:
        return passage.strip()

    schema = claim_schema or structured_schema_cikar(claim_text, source="claim")
    subject = (schema.get("subject") or {}).get("canonical")
    locations = _canon_set(schema.get("locations"))
    years = {str(y) for y in schema.get("years", [])}
    pct_vals = schema.get("percentages", [])
    claim_tokens = set(icerik_tokenlari(claim_text))

    scored = []
    for i, sentence in enumerate(sentences):
        normal = metni_normalize_et(sentence)
        stoks = set(icerik_tokenlari(sentence))
        overlap = len(claim_tokens & stoks) / max(1, len(claim_tokens))
        score = overlap
        if subject and subject in normal:
            score += 0.42
        if locations and any(loc in normal for loc in locations):
            score += 0.24
        if years and any(y in normal for y in years):
            score += 0.18
        if pct_vals and ("%" in sentence or re.search(r"\byüzde\b", normal)):
            score += 0.14
        if schema.get("primary_rank") and re.search(r"\b(?:birinci|ikinci|üçüncü|dördüncü|beşinci|altıncı|yedinci|sekizinci|dokuzuncu|onuncu|lider|ilk\s+sırada|en\s+(?:çok|fazla|büyük|yüksek))\b", normal):
            score += 0.14
        if schema.get("change_direction") and change_direction_cikar(sentence) is not None:
            score += 0.10
        scored.append((score, i))

    _, best_i = max(scored, key=lambda x: x[0])
    # Keep one neighboring sentence when possible; max 3 sentences total.
    indices = [best_i]
    if best_i > 0:
        indices.insert(0, best_i - 1)
    if len(indices) < max_sentences and best_i + 1 < len(sentences):
        indices.append(best_i + 1)
    indices = sorted(set(indices))[:max_sentences]
    return " ".join(sentences[i] for i in indices).strip()


def lexical_chunk_skoru(iddia, chunk):
    return 0.72 * token_overlap(iddia, chunk) + 0.28 * jaccard_overlap(iddia, chunk)


def kaynak_chunklarini_sinirla(iddia, chunks):
    if len(chunks) <= MAX_CHUNK_PER_SOURCE:
        return chunks

    scored = []
    for c in chunks:
        score = lexical_chunk_skoru(iddia, c["metin"])
        cc = dict(c)
        cc["lexical_source_score"] = score
        scored.append(cc)

    scored.sort(key=lambda x: x["lexical_source_score"], reverse=True)
    selected = scored[:MAX_CHUNK_PER_SOURCE]
    selected.sort(key=lambda x: x["index"])
    return selected


def adjacent_pairs(chunks):
    if not ADJACENT_PAIR_ENABLED:
        return []

    pairs = []
    for i in range(len(chunks) - 1):
        a = chunks[i]
        b = chunks[i + 1]
        combined = f"{a['metin']} {b['metin']}".strip()
        if len(combined.split()) > 260:
            continue
        pairs.append({
            "metin": combined,
            "chunk_indices": [a["index"], b["index"]],
        })
    return pairs


# ==========================================================
# BI-ENCODER / CROSS-ENCODER
# ==========================================================


def cosine_scores(query_vector, matrix):
    import numpy as np

    q = np.asarray(query_vector, dtype=float)
    m = np.asarray(matrix, dtype=float)

    q_norm = np.linalg.norm(q)
    m_norm = np.linalg.norm(m, axis=1)
    denom = np.maximum(q_norm * m_norm, 1e-12)
    return (m @ q) / denom


def embedding_skorlarini_hesapla(iddia, adaylar):
    if not adaylar:
        return

    model = get_embedding_model()
    texts = [a["ranking_text"] for a in adaylar]

    matrix = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    query = model.encode(
        [iddia],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )[0]

    scores = cosine_scores(query, matrix)

    for a, s in zip(adaylar, scores):
        a["embedding_score"] = float(s)


def cross_encoder_skorlarini_hesapla(iddia, adaylar):
    model = get_cross_encoder_model()
    if model is None or not adaylar:
        for a in adaylar:
            a["cross_encoder_raw"] = None
            a["cross_encoder_score"] = None
        return False

    pairs = [(iddia, a["ranking_text"]) for a in adaylar]

    try:
        raw_scores = model.predict(
            pairs,
            batch_size=8,
            show_progress_bar=True,
        )

        # Handle logits, 1-D arrays or nested outputs.
        for a, raw in zip(adaylar, raw_scores):
            try:
                if hasattr(raw, "tolist"):
                    raw = raw.tolist()
                if isinstance(raw, (list, tuple)):
                    if len(raw) == 1:
                        raw = raw[0]
                    else:
                        # If a classifier returns multiple logits, use the last class.
                        raw = raw[-1]
                raw = float(raw)
            except Exception:
                raw = 0.0

            a["cross_encoder_raw"] = raw
            a["cross_encoder_score"] = sigmoid(raw)

        return True
    except Exception as exc:
        global _cross_encoder_failed_reason
        _cross_encoder_failed_reason = f"predict failed: {type(exc).__name__}: {exc}"
        print(f"[UYARI] Cross-encoder predict başarısız: {_cross_encoder_failed_reason}")
        for a in adaylar:
            a["cross_encoder_raw"] = None
            a["cross_encoder_score"] = None
        return False


# ==========================================================
# FINAL GENERIC RETRIEVAL SCORING
# ==========================================================


def evidence_schema_for_candidate(candidate):
    if "evidence_schema" not in candidate:
        # V5.2 binds constraints to the focused passage first. The title stays available
        # for ranking, but no longer dominates entity/number binding inside the passage.
        evidence_text = candidate.get("focus_text") or candidate["metin"]
        candidate["evidence_schema"] = structured_schema_cikar(evidence_text, source="evidence")
    return candidate["evidence_schema"]


def score_candidate(iddia, claim_schema, aday, cross_available):
    evidence_schema = evidence_schema_for_candidate(aday)
    alignment = constraint_alignment(
        iddia,
        claim_schema,
        aday["ranking_text"],
        evidence_schema,
    )

    lexical = token_overlap(iddia, aday["ranking_text"])
    embedding = max(0.0, min(1.0, float(aday.get("embedding_score") or 0.0)))
    search_score = max(0.0, min(1.0, float(aday.get("arama_benzerligi") or 0.0)))
    directness = float(alignment.get("constraint_directness") or 0.0)
    structural = float(alignment.get("structural_relevance") or 0.0)
    gate = float(alignment.get("structural_gate") or 1.0)

    if cross_available and aday.get("cross_encoder_score") is not None:
        cross = max(0.0, min(1.0, float(aday["cross_encoder_score"])))
        pre_gate = (
            0.29 * cross
            + 0.25 * embedding
            + 0.09 * search_score
            + 0.08 * lexical
            + 0.08 * directness
            + 0.21 * structural
        )
    else:
        cross = None
        pre_gate = (
            0.39 * embedding
            + 0.11 * search_score
            + 0.11 * lexical
            + 0.11 * directness
            + 0.28 * structural
        )

    # Structural gating does not inspect whether values AGREE with the claim. It only
    # asks whether the evidence is about the same subject/location/time/predicate frame.
    # Therefore direct counter-evidence remains retrievable.
    final = pre_gate * (0.58 + 0.42 * gate)

    return final, {
        "embedding_score": round(embedding, 4),
        "cross_encoder_score": round(cross, 4) if cross is not None else None,
        "search_score": round(search_score, 4),
        "lexical_overlap": round(lexical, 4),
        "structural_relevance": round(structural, 4),
        "structural_gate": round(gate, 4),
        "pre_gate_score": round(pre_gate, 4),
        "constraint_alignment": alignment,
        "scoring_mode": "cross_encoder_plus_structural_gate" if cross is not None else "embedding_plus_structural_gate",
    }


def source_diverse_topk(adaylar, k=TOP_KANIT_SAYISI):
    selected = []
    used_domains = set()

    for a in sorted(adaylar, key=lambda x: x["retrieval_score"], reverse=True):
        d = a.get("domain") or ""
        if d and d in used_domains:
            continue
        selected.append(a)
        if d:
            used_domains.add(d)
        if len(selected) >= k:
            break

    return selected


# ==========================================================
# ATOMIC CLAIM RETRIEVAL
# ==========================================================


def atomic_kanit_bul(iddia, claim_schema=None, max_sources=TOP_KANIT_SAYISI, yazdir=True):
    fallback_schema = structured_schema_cikar(iddia, source="claim")
    schema = external_schema_merge(fallback_schema, claim_schema)

    search_results = coklu_internet_aramasi(iddia, schema)

    if not search_results:
        return {
            "iddia": iddia,
            "claim_schema": schema,
            "durum": "arama_sonucu_yok",
            "kanit_sayisi": 0,
            "kanitlar": [],
            "diagnostics": {},
        }

    unique_url_count = len(search_results)
    search_results = arama_sonuclarini_filtrele(iddia, search_results)

    print(f"\nToplam benzersiz URL: {unique_url_count}")
    print(f"Ön filtre sonrası kaynak: {len(search_results)}")

    print("\n====================================")
    print("KULLANILACAK KAYNAKLAR")
    print("====================================")
    for i, r in enumerate(search_results, 1):
        print(f"\n{i}. {r['title']}")
        print(r["url"])
        print(f"Arama benzerliği: {r['arama_benzerligi']:.4f}")

    candidates = []
    single_count = 0
    pair_count = 0

    for i, r in enumerate(search_results, 1):
        print(f"\n[{i}/{len(search_results)}] {r['title']}")
        text, method = web_sayfasindan_metin_cek(r["url"], r.get("content", ""))

        if not text:
            print("→ Metin alınamadı.")
            continue

        print(f"→ {method} ile alındı.")

        chunks = metni_parcala(text)
        original_count = len(chunks)
        chunks = kaynak_chunklarini_sinirla(iddia, chunks)

        if not chunks:
            print("→ Kullanılabilir chunk yok.")
            continue

        if original_count > len(chunks):
            print(f"→ {original_count} chunk oluştu, {len(chunks)} tutuldu.")
        else:
            print(f"→ {len(chunks)} chunk")

        domain = domain_bul(r["url"])

        for c in chunks:
            passage = c["metin"]
            focus_text = evidence_focus_passage(iddia, passage, schema)
            candidates.append({
                "metin": focus_text,
                "raw_metin": passage,
                "focus_text": focus_text,
                "ranking_text": f"{r['title']}\n{focus_text}",
                "kaynak": r["title"],
                "url": r["url"],
                "domain": domain,
                "published_date": r.get("published_date"),
                "alma_yontemi": method,
                "candidate_type": "single_chunk",
                "chunk_indices": [c["index"]],
                "arama_benzerligi": r["arama_benzerligi"],
                "tavily_score": r.get("tavily_score", 0.0),
            })
            single_count += 1

        for p in adjacent_pairs(chunks):
            passage = p["metin"]
            focus_text = evidence_focus_passage(iddia, passage, schema)
            candidates.append({
                "metin": focus_text,
                "raw_metin": passage,
                "focus_text": focus_text,
                "ranking_text": f"{r['title']}\n{focus_text}",
                "kaynak": r["title"],
                "url": r["url"],
                "domain": domain,
                "published_date": r.get("published_date"),
                "alma_yontemi": method,
                "candidate_type": "adjacent_pair",
                "chunk_indices": p["chunk_indices"],
                "arama_benzerligi": r["arama_benzerligi"],
                "tavily_score": r.get("tavily_score", 0.0),
            })
            pair_count += 1

    if not candidates:
        return {
            "iddia": iddia,
            "claim_schema": schema,
            "durum": "yeterli_kanit_yok",
            "kanit_sayisi": 0,
            "kanitlar": [],
            "diagnostics": {
                "benzersiz_url_sayisi": unique_url_count,
                "on_filtre_sonrasi_kaynak": len(search_results),
            },
        }

    print(f"\nTek chunk adayları: {single_count}")
    print(f"Adjacent pair adayları: {pair_count}")
    print(f"Toplam embedding adayı: {len(candidates)}")

    embedding_skorlarini_hesapla(iddia, candidates)

    # Keep semantically plausible passages, but always retain some top passages.
    candidates.sort(key=lambda x: x.get("embedding_score", 0.0), reverse=True)
    thresholded = [
        c for c in candidates
        if c.get("embedding_score", 0.0) >= MIN_EMBEDDING_BENZERLIGI
    ]

    if len(thresholded) < min(12, len(candidates)):
        thresholded = candidates[:min(max(12, len(thresholded)), len(candidates))]

    bi_pool = thresholded[:BI_ENCODER_TOP_N]

    print(f"Minimum skor sonrası aday evidence: {len(thresholded)}")
    print(f"Bi-encoder rerank havuzu: {len(bi_pool)}")

    cross_pool = bi_pool[:CROSS_ENCODER_TOP_N]
    cross_available = cross_encoder_skorlarini_hesapla(iddia, cross_pool)

    # Candidates outside cross-pool still receive fallback scoring.
    scored = []
    for c in bi_pool:
        use_cross = cross_available and c in cross_pool
        score, details = score_candidate(
            iddia,
            schema,
            c,
            cross_available=use_cross,
        )
        c["retrieval_score"] = score
        c["rerank_detaylari"] = details
        scored.append(c)

    selected = source_diverse_topk(scored, max_sources)

    evidence_out = []
    for i, c in enumerate(selected, 1):
        evidence_out.append({
            "sira": i,
            "candidate_type": c["candidate_type"],
            "chunk_indices": c["chunk_indices"],
            "metin": c["metin"],
            "kaynak": c["kaynak"],
            "domain": c["domain"],
            "url": c["url"],
            "alma_yontemi": c["alma_yontemi"],
            "published_date": c.get("published_date"),
            "embedding_skoru": round(float(c.get("embedding_score") or 0.0), 4),
            "cross_encoder_raw": (
                round(float(c["cross_encoder_raw"]), 4)
                if c.get("cross_encoder_raw") is not None
                else None
            ),
            "cross_encoder_skoru": (
                round(float(c["cross_encoder_score"]), 4)
                if c.get("cross_encoder_score") is not None
                else None
            ),
            "retrieval_skoru": round(float(c["retrieval_score"]), 4),
            "rerank_detaylari": c["rerank_detaylari"],
            "evidence_schema": c.get("evidence_schema"),
        })

    result = {
        "iddia": iddia,
        "claim_schema": schema,
        "durum": "kanit_bulundu" if evidence_out else "yeterli_kanit_yok",
        "kanit_sayisi": len(evidence_out),
        "kanitlar": evidence_out,
        "diagnostics": {
            "surum": SURUM,
            "search_provider": os.getenv("RETRIEVAL_SEARCH_PROVIDER", "tavily"),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "cross_encoder_enabled": CROSS_ENCODER_ENABLED,
            "cross_encoder_model": CROSS_ENCODER_MODEL_NAME,
            "cross_encoder_available": bool(cross_available),
            "cross_encoder_fallback_reason": _cross_encoder_failed_reason,
            "benzersiz_url_sayisi": unique_url_count,
            "on_filtre_sonrasi_kaynak": len(search_results),
            "tek_chunk_aday_sayisi": single_count,
            "adjacent_pair_aday_sayisi": pair_count,
            "toplam_embedding_adayi": len(candidates),
            "threshold_sonrasi_aday": len(thresholded),
            "bi_encoder_pool": len(bi_pool),
            "cross_encoder_pool": len(cross_pool) if cross_available else 0,
            "query_count": len(sorgulari_uret(iddia, schema)),
            "architecture": {
                "generic_semantic_frame_schema": True,
                "person1_atomic_claim_compatible": True,
                "backward_compatible_search_api": True,
                "optional_claim_frame_adapter": True,
                "candidate_local_focus_passage": True,
                "context_aware_atomic_decomposition": True,
                "entity_bound_constraints": True,
                "location_binding": True,
                "structural_relevance_gating": True,
                "search_provider_abstraction": True,
                "bi_encoder_retrieval": True,
                "cross_encoder_reranking": True,
                "structured_constraint_alignment": True,
                "source_diversity": True,
                "final_truth_decision_in_retrieval": False,
            },
        },
    }

    if yazdir:
        print("\n====================================")
        print("RETRIEVAL SONUCU")
        print("====================================")
        print(f"\nDurum: {result['durum']}")
        print(f"Kanıt sayısı: {len(evidence_out)}")

        print("\nClaim schema:")
        print(json.dumps(schema, ensure_ascii=False, indent=2))

        for e in evidence_out:
            print(f"\n{e['sira']}. KANIT")
            print(f"Candidate type: {e['candidate_type']}")
            print(f"Chunk indices: {e['chunk_indices']}")
            print(f"Embedding: {e['embedding_skoru']}")
            print(f"Cross-encoder: {e['cross_encoder_skoru']}")
            print(f"Retrieval score: {e['retrieval_skoru']}")
            print(f"Kaynak: {e['kaynak']}")
            print(f"Domain: {e['domain']}")
            print(f"Alma yöntemi: {e['alma_yontemi']}")
            print(f"URL: {e['url']}")

            print("\nStructured alignment:")
            print(json.dumps(
                e["rerank_detaylari"]["constraint_alignment"],
                ensure_ascii=False,
                indent=2,
            ))

            print("\nMetin:")
            print(e["metin"])
            print("\n------------------------------------")

    return result


# ==========================================================
# FULL CLAIM ORCHESTRATION
# ==========================================================


def kanit_bul(iddia, atomic_claims=None, claim_schemas=None, yazdir=True):
    """Generic integration entry point with discourse-context preservation."""
    if atomic_claims is None:
        frames = fallback_atomic_claim_frames(iddia)
        decomposition_mode = "fallback_contextual_v5_2"
    else:
        clean = [x.strip() for x in atomic_claims if x and x.strip()]
        frames = [
            {
                "raw_text": x,
                "resolved_text": x,
                "inherited_context": {},
                "schema": structured_schema_cikar(x, source="claim"),
            }
            for x in clean
        ]
        decomposition_mode = "external"

    if not frames:
        frames = [{
            "raw_text": iddia,
            "resolved_text": iddia,
            "inherited_context": {},
            "schema": structured_schema_cikar(iddia, source="claim"),
        }]

    if claim_schemas is None:
        external_schemas = [None] * len(frames)
    elif isinstance(claim_schemas, dict) and len(frames) == 1:
        external_schemas = [claim_schemas]
    elif isinstance(claim_schemas, list):
        external_schemas = list(claim_schemas) + [None] * max(0, len(frames) - len(claim_schemas))
    else:
        external_schemas = [None] * len(frames)

    for i, ext in enumerate(external_schemas[:len(frames)]):
        if ext:
            frames[i]["schema"] = external_schema_merge(frames[i]["schema"], ext)

    atomic_texts = [f["resolved_text"] for f in frames]

    if yazdir:
        print("\n====================================")
        print("TAM İDDİA")
        print("====================================")
        print(iddia)
        print("\n====================================")
        print("ATOMIC CLAIMS")
        print("====================================")
        print(f"Decomposition mode: {decomposition_mode}")
        for i, f in enumerate(frames, 1):
            print(f"{i}. {f['resolved_text']}")
            inherited = f.get("inherited_context") or {}
            if any(inherited.get(k) for k in ["subject", "locations", "years"]):
                printable = {
                    "subject": (inherited.get("subject") or {}).get("text") if inherited.get("subject") else None,
                    "locations": [x.get("text") or x.get("canonical") for x in inherited.get("locations", [])],
                    "years": inherited.get("years", []),
                }
                print(f"   ↳ inherited context: {printable}")

    alt_results = []
    for i, f in enumerate(frames, 1):
        claim = f["resolved_text"]
        schema = f["schema"]
        if yazdir:
            print("\n\n####################################")
            print(f"ALT İDDİA {i}")
            print("####################################")
            print(claim)
        alt_results.append(atomic_kanit_bul(claim, claim_schema=schema, yazdir=yazdir))

    return {
        "surum": SURUM,
        "iddia": iddia,
        "decomposition_mode": decomposition_mode,
        "parcalandi": len(frames) > 1,
        "alt_iddia_sayisi": len(frames),
        "alt_iddia_metinleri": atomic_texts,
        "atomic_claim_frames": frames,
        "alt_iddialar": alt_results,
    }



# ==========================================================
# PUBLIC INTEGRATION API (PERSON 2 -> PERSON 3 / PERSON 1)
# ==========================================================


def _source_payloads_from_result(result, max_sources=TOP_KANIT_SAYISI):
    """Convert internal evidence objects to the repo's stable Source contract.

    Required by Person 1 today: id, url, text.
    Additive metadata is safe because Person 3 currently picks only those three
    fields when constructing ai_engine.Source, while the frontend can inspect the
    extra fields from raw_sources if desired.
    """
    out = []
    for i, ev in enumerate((result or {}).get("kanitlar", [])[:max_sources], 1):
        out.append({
            "id": f"{SOURCE_ID_PREFIX}_{i}",
            "url": ev.get("url"),
            "text": ev.get("metin") or "",
            "title": ev.get("kaynak") or "",
            "domain": ev.get("domain") or "",
            "published_date": ev.get("published_date"),
            "retrieval_score": ev.get("retrieval_skoru"),
            "candidate_type": ev.get("candidate_type"),
            "metadata": {
                "retrieval_version": SURUM,
                "api_version": RETRIEVAL_API_VERSION,
                "embedding_score": ev.get("embedding_skoru"),
                "cross_encoder_score": ev.get("cross_encoder_skoru"),
                "alignment": (ev.get("rerank_detaylari") or {}).get("constraint_alignment"),
            },
        })
    return out


def search_detailed(claim_text: str, max_sources: int = 5, claim_frame=None, debug: bool = False):
    """Detailed retrieval service for integration tests/benchmarking.

    `claim_frame` is optional and future-facing. Current Person 1 only sends an
    atomic claim string; when a richer frame becomes available, the same function
    can consume it without breaking the current backend.
    """
    if not isinstance(claim_text, str) or not claim_text.strip():
        return {
            "version": SURUM,
            "api_version": RETRIEVAL_API_VERSION,
            "claim": claim_text,
            "status": "invalid_claim",
            "sources": [],
            "retrieval": None,
        }

    try:
        max_sources = int(max_sources)
    except (TypeError, ValueError):
        max_sources = 5
    max_sources = max(1, min(max_sources, 10))

    schema = claim_frame_to_schema(claim_text.strip(), claim_frame)
    result = atomic_kanit_bul(
        claim_text.strip(),
        claim_schema=schema,
        max_sources=max_sources,
        yazdir=debug,
    )
    sources = _source_payloads_from_result(result, max_sources=max_sources)
    return {
        "version": SURUM,
        "api_version": RETRIEVAL_API_VERSION,
        "claim": claim_text.strip(),
        "status": result.get("durum"),
        "sources": sources,
        "retrieval": result,
    }


def search(claim_text: str, max_sources: int = 5, claim_frame=None):
    """DROP-IN contract used by the repo's current backend/main.py.

    Existing call remains valid:
        raw_sources = search(claim.text, max_sources=5)

    Returns a list of dicts. Every item always contains `id`, `url`, `text`, so
    Person 3's existing Source(...) conversion and Person 1's verify_claim() keep
    working unchanged. Additional metadata is additive and optional for frontend.
    """
    detailed = search_detailed(
        claim_text=claim_text,
        max_sources=max_sources,
        claim_frame=claim_frame,
        debug=False,
    )
    return detailed["sources"]


# ==========================================================
# BENCHMARK SUPPORT
# ==========================================================


def _load_benchmark(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("Benchmark JSON bir liste olmalı.")
        return data

    items = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def _gold_match(url, gold_urls):
    cu = canonical_url(url)
    for g in gold_urls:
        cg = canonical_url(g)
        if cu == cg:
            return True
        # Also accept same domain+path prefix for tracking/query variants.
        pu = urlparse(cu)
        pg = urlparse(cg)
        if pu.netloc == pg.netloc and (
            pu.path == pg.path
            or pu.path.startswith(pg.path.rstrip("/") + "/")
            or pg.path.startswith(pu.path.rstrip("/") + "/")
        ):
            return True
    return False


def benchmark_calistir(path, output_path=None):
    items = _load_benchmark(path)
    if not items:
        raise ValueError("Benchmark boş.")

    results = []
    recall_hits = 0
    reciprocal_ranks = []
    gold_count = 0

    for idx, item in enumerate(items, 1):
        claim = item.get("claim") or item.get("iddia")
        if not claim:
            continue

        print("\n" + "=" * 70)
        print(f"BENCHMARK {idx}/{len(items)}")
        print(claim)

        result = atomic_kanit_bul(
            claim,
            claim_schema=item.get("claim_schema"),
            yazdir=False,
        )

        gold_urls = item.get("gold_urls") or []
        metric = None

        if gold_urls:
            gold_count += 1
            first_rank = None
            for rank, ev in enumerate(result.get("kanitlar", []), 1):
                if _gold_match(ev.get("url", ""), gold_urls):
                    first_rank = rank
                    break

            hit = first_rank is not None
            if hit:
                recall_hits += 1
                reciprocal_ranks.append(1.0 / first_rank)
            else:
                reciprocal_ranks.append(0.0)

            metric = {
                "gold_hit_at_k": hit,
                "first_gold_rank": first_rank,
                "reciprocal_rank": (1.0 / first_rank) if first_rank else 0.0,
            }

        results.append({
            "claim": claim,
            "gold_urls": gold_urls,
            "metric": metric,
            "retrieval": result,
        })

    summary = {
        "version": SURUM,
        "benchmark_items": len(results),
        "items_with_gold": gold_count,
        "recall_at_k": (recall_hits / gold_count) if gold_count else None,
        "mrr": (sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else None,
        "k": TOP_KANIT_SAYISI,
    }

    out = {"summary": summary, "results": results}

    output_path = output_path or "benchmark_v5_2_sonuc.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n====================================")
    print("BENCHMARK SUMMARY")
    print("====================================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Kaydedildi: {output_path}")

    return out


# ==========================================================
# LOCAL SELF TESTS (NO WEB / NO MODEL)
# ==========================================================


def v5_2_local_testleri():
    tests = []

    # A - Generic numeric/change schema
    c1 = "Türkiye'de 2025 yılında tamamen elektrikli otomobil satışları yüzde 80 arttı."
    s1 = structured_schema_cikar(c1, source="claim")
    tests.append({"test": "A_generic_numeric_schema", "basarili": s1["years"] == [2025] and s1["percentages"] == [80.0] and s1["predicate_type"] == "CHANGE", "schema": s1})

    # B - Named subject + location + rank, no named hardcode.
    c2 = "Orion 2024 yılında Atlas'ta telefon satışlarında ikinci sırada yer aldı."
    s2 = structured_schema_cikar(c2, source="claim")
    tests.append({"test": "B_generic_subject_location_rank", "basarili": (s2.get("subject") or {}).get("canonical") == "orion" and "atlas" in _canon_set(s2.get("locations")) and (s2.get("primary_rank") or {}).get("rank") == 2, "schema": s2})

    # C - Context inheritance/coreference across atomic clauses.
    compound = "Orion 2024 yılında Atlas'ta telefon satışlarında birinci sırada yer aldı ve şirketin Atlas'taki satışları bir önceki yıla göre yüzde 20 arttı."
    frames = fallback_atomic_claim_frames(compound)
    second = frames[1] if len(frames) > 1 else {}
    second_schema = second.get("schema") or {}
    tests.append({"test": "C_context_inheritance", "basarili": len(frames) == 2 and (second_schema.get("subject") or {}).get("canonical") == "orion" and second_schema.get("years") == [2024] and "atlas" in _canon_set(second_schema.get("locations")) and second_schema.get("percentages") == [20.0], "frame": second})

    # D - Rank must bind to matching entity, not a nearby different entity.
    claim = "Orion 2024 yılında telefon satışlarında birinci sırada yer aldı."
    ev = "Orion ve Vega 2024 yılında liderliği paylaştı. Nova ise üçüncü sırada yer aldı."
    cs = structured_schema_cikar(claim, source="claim")
    es = structured_schema_cikar(ev, source="evidence")
    al = constraint_alignment(claim, cs, ev, es)
    tests.append({"test": "D_entity_bound_rank", "basarili": al["rank_relation"] == "SAME" and (al.get("evidence_rank") or {}).get("rank") == 1 and (al.get("evidence_rank") or {}).get("binding", {}).get("subject_relation") == "SAME", "alignment": al})

    # E - If only another entity has a rank, do not compare rank values as if same subject.
    ev2 = "Nova 2024 yılında telefon satışlarında üçüncü sırada yer aldı."
    es2 = structured_schema_cikar(ev2, source="evidence")
    al2 = constraint_alignment(claim, cs, ev2, es2)
    tests.append({"test": "E_rank_entity_mismatch", "basarili": al2["rank_relation"] == "ENTITY_MISMATCH", "alignment": al2})

    # F - Same percentage but wrong entity should be context mismatch, not supporting numeric evidence.
    growth_claim = "Orion 2025 yılında Atlas'taki satışlarını yüzde 20 artırdı."
    wrong_ev = "Nova 2025 yılında Atlas'taki kârını yüzde 20,4 artırdı."
    gcs = structured_schema_cikar(growth_claim, source="claim")
    ges = structured_schema_cikar(wrong_ev, source="evidence")
    gal = constraint_alignment(growth_claim, gcs, wrong_ev, ges)
    tests.append({"test": "F_percentage_context_binding", "basarili": gal["subject_relation"] == "DIFFERENT" and gal["structural_gate"] < 0.6, "alignment": gal})

    # G - Direct counter-evidence keeps same frame and differing value.
    counter = "Orion 2024 yılında Atlas'ta telefon satışlarında dördüncü sırada yer aldı."
    ces = structured_schema_cikar(counter, source="evidence")
    cal = constraint_alignment(c2, s2, counter, ces)
    tests.append({"test": "G_counter_evidence_same_frame", "basarili": cal["rank_value_relation"] == "DIFFERENT" and cal["rank_binding_relation"] == "COMPATIBLE" and cal["structural_gate"] >= 0.9, "alignment": cal})

    # H - Brand/model scope generic protection.
    brand_claim = "Nova 2025 yılında telefon pazarının lider markası oldu."
    model_ev = "Nova X1 2025 yılında en çok satan telefon modeli oldu."
    bcs = structured_schema_cikar(brand_claim, source="claim")
    bes = structured_schema_cikar(model_ev, source="evidence")
    bal = constraint_alignment(brand_claim, bcs, model_ev, bes)
    tests.append({"test": "H_generic_brand_model_scope", "basarili": bcs["ranking_scope_hint"] == "BRAND" and bes["ranking_scope_hint"] == "MODEL" and bal["rank_relation"] in {"SCOPE_MISMATCH", "ENTITY_MISMATCH"}, "alignment": bal})

    # I - Different year is structural mismatch for a year-specific rank claim.
    old_ev = "Orion 2023 yılında Atlas'ta telefon satışlarında birinci sırada yer aldı."
    oes = structured_schema_cikar(old_ev, source="evidence")
    oal = constraint_alignment(c2, s2, old_ev, oes)
    tests.append({"test": "I_rank_time_mismatch", "basarili": oal["rank_relation"] == "TIME_MISMATCH", "alignment": oal})

    # J - Generic comparison.
    comp = "Şirket A'nın geliri Şirket B'den daha hızlı büyüdü."
    comps = structured_schema_cikar(comp, source="claim")
    tests.append({"test": "J_generic_comparison", "basarili": comps["comparison_operator"] == "FASTER" and comps["predicate_type"] == "COMPARISON", "schema": comps})

    # K - Safe compound split with explicit second subject.
    explicit_compound = "Orion 2025 yılında birinci oldu ve Nova 2025 yılında ikinci oldu."
    parts = fallback_atomic_claim_frames(explicit_compound)
    tests.append({"test": "K_explicit_subject_not_overwritten", "basarili": len(parts) == 2 and (parts[1]["schema"].get("subject") or {}).get("canonical") == "nova", "parts": parts})

    # L - Word boundary relative time regression.
    rel = goreli_zaman_bilgisi("Orion 2025 yılında dördüncü sırada yer aldı.", now=datetime(2026, 10, 3, 12, 0))
    tests.append({"test": "L_dorduncu_not_yesterday", "basarili": rel["has_relative_time"] is False, "relative_time": rel})

    # M - Real yesterday still works.
    rel2 = goreli_zaman_bilgisi("Şirket dün yeni fabrikasını açtı.", now=datetime(2026, 10, 3, 12, 0))
    tests.append({"test": "M_real_yesterday", "basarili": rel2["has_relative_time"] and rel2["start"] == "2026-10-02", "relative_time": rel2})

    # N - Query generator uses inherited structural context, not only surface child text.
    qs = sorgulari_uret(second.get("resolved_text", ""), second_schema)
    joined = " | ".join(metni_normalize_et(x) for x in qs)
    tests.append({"test": "N_structured_query_context", "basarili": all(x in joined for x in ["orion", "atlas", "2024", "yüzde 20"]), "queries": qs})

    # O - Full-year vs partial evidence metadata.
    fc = "Orion 2025 yılında 100 milyon gelir elde etti."
    fe = "Orion 2025'in ilk altı ayında 60 milyon gelir elde etti."
    fcs = structured_schema_cikar(fc, source="claim")
    fes = structured_schema_cikar(fe, source="evidence")
    fal = constraint_alignment(fc, fcs, fe, fes)
    tests.append({"test": "O_time_scope_partial", "basarili": fal["time_scope_relation"] == "PARTIAL", "alignment": fal})

    # P - Context-independent entity names demonstrate no Apple/Togg/Türkiye hardcode.
    pclaim = "Helios 2026 yılında Borealis'te enerji satışlarında üçüncü sırada yer aldı."
    pev = "Helios 2026 yılında Borealis'te enerji satışlarında ikinci sırada yer aldı."
    pcs = structured_schema_cikar(pclaim, source="claim")
    pes = structured_schema_cikar(pev, source="evidence")
    pal = constraint_alignment(pclaim, pcs, pev, pes)
    tests.append({"test": "P_unseen_names_generalize", "basarili": pal["rank_value_relation"] == "DIFFERENT" and pal["rank_binding_relation"] == "COMPATIBLE", "alignment": pal})


    # Q - Current backend contract: Source dicts must contain id/url/text and stable IDs.
    fake_result = {
        "kanitlar": [
            {"url": "https://example.com/a", "metin": "kanıt A", "kaynak": "A", "domain": "example.com", "retrieval_skoru": 0.9},
            {"url": "https://example.org/b", "metin": "kanıt B", "kaynak": "B", "domain": "example.org", "retrieval_skoru": 0.8},
        ]
    }
    srcs = _source_payloads_from_result(fake_result, max_sources=2)
    tests.append({"test": "Q_backend_source_contract", "basarili": len(srcs) == 2 and srcs[0]["id"] == "source_1" and all(all(k in x for k in ("id", "url", "text")) for x in srcs), "sources": srcs})

    # R - Future ClaimFrame adapter must enrich fallback without requiring Person 1 today.
    frame = {
        "subject": {"name": "Orion", "type": "ORGANIZATION"},
        "location": "Atlas",
        "time": {"year": 2027, "scope": "FULL_YEAR"},
        "predicate": "CHANGE",
        "metric": "telefon satışları",
        "constraints": [{"type": "PERCENTAGE_CHANGE", "value": 12, "direction": "UP"}],
    }
    fs = claim_frame_to_schema("Orion satışları arttı.", frame)
    tests.append({"test": "R_future_claim_frame_adapter", "basarili": (fs.get("subject") or {}).get("canonical") == "orion" and "atlas" in _canon_set(fs.get("locations")) and fs.get("years") == [2027] and fs.get("percentages") == [12.0] and fs.get("predicate_type") == "CHANGE", "schema": fs})

    # S - Candidate-local focus should prefer the sentence containing claim frame cues.
    fp_claim = "Orion 2025 yılında Atlas'ta satışlarını yüzde 20 artırdı."
    fp_schema = structured_schema_cikar(fp_claim, source="claim")
    noisy = "Nova başka bir pazarda üçüncü oldu. Orion 2025 yılında Atlas'ta satışlarını yüzde 20 artırdı. Vega yeni ürün tanıttı. Dördüncü bir şirket yatırım yaptı."
    focused = evidence_focus_passage(fp_claim, noisy, fp_schema, max_sentences=3)
    tests.append({"test": "S_candidate_local_focus", "basarili": "Orion 2025" in focused and "yüzde 20" in focused, "focused": focused})

    return tests


# ==========================================================
# SAVE / CLI
# ==========================================================


def json_kaydet(data, path=JSON_DOSYA_ADI):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nJSON sonucu kaydedildi: {path}")


def main():
    claim = iddia_al()
    if not claim:
        print("[HATA] İddia boş bırakılamaz.")
        return

    # Standalone smoke test mirrors the real repo integration: Person 1 is expected
    # to have already produced one atomic claim before calling search().
    result = search_detailed(claim, max_sources=TOP_KANIT_SAYISI, debug=True)
    json_kaydet(result)

    print("\n====================================")
    print("RETRIEVAL V5.2 TAMAMLANDI")
    print("====================================")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        print("\n====================================")
        print("V5.2 LOCAL SELF TESTS")
        print("====================================")

        results = v5_2_local_testleri()
        failed = []

        for t in results:
            status = "PASS" if t.get("basarili") else "FAIL"
            print(f"\n[{status}] {t['test']}")
            print(json.dumps(t, ensure_ascii=False, indent=2, default=str))
            if not t.get("basarili"):
                failed.append(t)

        if failed:
            raise SystemExit(1)

        print(f"\nTüm V5.2 local testleri geçti: {len(results)}/{len(results)}")

    elif "--benchmark" in sys.argv:
        idx = sys.argv.index("--benchmark")
        if idx + 1 >= len(sys.argv):
            raise SystemExit("Kullanım: python search_retrieval_v5_2.py --benchmark benchmark.jsonl")
        benchmark_calistir(sys.argv[idx + 1])

    else:
        main()
