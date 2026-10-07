import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from ai_engine import FactCheckEngine, Source
from search_retrieval import search


app = FastAPI(
    title="Verify Backend",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

allowed_origins = [
    "https://verify-test-swart.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173",
]

frontend_origin = os.getenv("FRONTEND_ORIGIN")

if frontend_origin and frontend_origin not in allowed_origins:
    allowed_origins.append(frontend_origin)


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# AI ENGINE
# ============================================================

engine = FactCheckEngine()


# ============================================================
# REQUEST MODEL
# ============================================================

class VerifyRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000
    )


# ============================================================
# VALIDATION ERROR
# ============================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "error": "VALIDATION_ERROR",
            "message": "Gönderilen veri geçersiz."
        }
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Backend çalışıyor!"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "verify-backend"
    }


# ============================================================
# VERIFY
# ============================================================

@app.post("/verify")
def verify(request: VerifyRequest):

    text = request.text.strip()

    if not text:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error": "EMPTY_TEXT",
                "message": "Doğrulanacak metin boş olamaz."
            }
        )

    try:

        # 1. CLAIM EXTRACTION
        extraction = engine.extract_claims(text)

        results = []

        # 2. HER CLAIM İÇİN İŞLEM
        for claim in extraction.claims:

            # Doğrulanabilir olmayan claim
            if not claim.is_verifiable:

                results.append({
                    "claim_id": claim.id,
                    "claim": claim.text,
                    "is_verifiable": False,
                    "verdict": None,
                    "explanation": None,
                    "used_sources": [],
                    "sources": []
                })

                continue

            # 3. ARDA'NIN RETRIEVAL V5.2
            raw_sources = search(
                claim.text,
                max_sources=5
            )

            # 4. RETRIEVAL SOURCE -> AI ENGINE SOURCE
            sources_for_ai = [
                Source(
                    id=source["id"],
                    url=source.get("url"),
                    text=source["text"]
                )
                for source in raw_sources
            ]

            # 5. VERDICT
            verdict = engine.verify_claim(
                claim.text,
                sources_for_ai
            )

            # 6. FRONTEND'E UYGUN SOURCE FORMAT
            frontend_sources = []

            for source in raw_sources:

                frontend_sources.append({
                    "id": source["id"],
                    "title": (
                        source.get("title")
                        or source.get("domain")
                        or "Kaynak"
                    ),
                    "site_name": source.get("domain"),
                    "domain": source.get("domain"),
                    "url": source.get("url"),
                    "published_date": source.get("published_date"),
                    "retrieval_score": source.get("retrieval_score"),
                    "text": source.get("text")
                })

            # 7. RESULT
            results.append({
                "claim_id": claim.id,
                "claim": claim.text,
                "is_verifiable": True,
                "verdict": verdict.verdict,
                "explanation": verdict.explanation,
                "used_sources": verdict.used_sources,
                "sources": frontend_sources
            })

        return {
            "status": "ok",
            "results": results
        }

    except Exception as e:

        print(
            f"[VERIFY ERROR] "
            f"{type(e).__name__}: {e}"
        )

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": "VERIFY_FAILED",
                "message": (
                    "Doğrulama işlemi sırasında "
                    "bir hata oluştu."
                )
            }
        )