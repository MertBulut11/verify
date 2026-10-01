from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from ai_engine import FactCheckEngine, Source
from search_retrieval import search


app = FastAPI()

engine = FactCheckEngine()


class VerifyRequest(BaseModel):
    text: str


@app.get("/")
def ana_sayfa():
    return {"mesaj": "Backend çalışıyor!"}


@app.post("/verify")
def verify(request: VerifyRequest):

    # 1. Mert'in AI'ı metinden iddiaları çıkarıyor
    extraction = engine.extract_claims(request.text)

    results = []

    # 2. Her iddia için işlem yap
    for claim in extraction.claims:

        # Doğrulanabilir değilse geç
        if not claim.is_verifiable:
            results.append({
                "claim_id": claim.id,
                "claim": claim.text,
                "is_verifiable": False
            })
            continue

        # 3. Arda'nın sistemi internette kaynak arıyor
        raw_sources = search(
            claim.text,
            max_sources=5
        )

        # 4. Arda'nın döndürdüğü dictionary'leri
        # Mert'in Source nesnelerine çeviriyoruz
        sources = [
            Source(
                id=source["id"],
                url=source["url"],
                text=source["text"]
            )
            for source in raw_sources
        ]

        # 5. Mert'in AI'ı iddiayı kaynaklara göre değerlendiriyor
        verdict = engine.verify_claim(
            claim.text,
            sources
        )

        # 6. Sonucu hazırlıyoruz
        results.append({
            "claim_id": claim.id,
            "claim": claim.text,
            "is_verifiable": claim.is_verifiable,
            "verdict": verdict.verdict,
            "explanation": verdict.explanation,
            "used_sources": verdict.used_sources,
            "sources": raw_sources
        })

    return {
        "results": results
    }