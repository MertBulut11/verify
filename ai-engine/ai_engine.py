"""
ai_engine.py
------------
Person 1 (AI / LLM / Claim Verification) module for the fact-checking
system.

Responsibilities of this module (per the team's task split):
    1. Claim Extraction & Decomposition
       Break a raw user-submitted text into atomic, independently
       verifiable claims.
    2. Verdict Generation
       Given a claim and a set of labelled evidence snippets (produced
       by Person 2's search/retrieval module), decide whether the claim
       is SUPPORTED, REFUTED, NOT ENOUGH EVIDENCE, or CONFLICTING, with
       an explanation that cites the evidence it relied on.

This module exposes exactly two functions Person 3 (Backend) needs:

    extract_claims(text: str) -> ExtractionResult
    verify_claim(claim_text: str, sources: list[Source]) -> VerdictResult

Everything else (prompt text, model choice, temperature, JSON schema
enforcement) is an internal detail Backend should not have to touch.

Setup
-----
    pip install -r requirements.txt
    export GEMINI_API_KEY="..."          # see .env.example

    python ai_engine.py                  # runs the smoke tests below
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import google.generativeai as genai
from pydantic import BaseModel, Field


# ============================================================
# 1. CONFIG
# ============================================================

# Cost-effective, low-latency model chosen for this project (validated
# in the Google AI Studio playground before this module was written).
# Kept as a constant/env override so it's a one-line change if a newer
# model should be swapped in later.
DEFAULT_MODEL_NAME = os.environ.get("FACTCHECK_MODEL_NAME", "gemini-3.8-flash")

# temperature=0.0: this system must behave like a rule-following
# classifier, not a creative writer. Halucination-suppression, not
# style, is the goal.
DEFAULT_TEMPERATURE = 0.0


# ============================================================
# 2. PYDANTIC SCHEMAS (the data contract with Backend / Person 3)
# ============================================================

class Claim(BaseModel):
    """A single atomic, independently verifiable claim."""

    id: str
    text: str
    is_verifiable: bool = Field(
        description=(
            "False for predictions, opinions, or vague/unmeasurable "
            "statements that cannot be fact-checked against evidence."
        )
    )


class ExtractionResult(BaseModel):
    claims: List[Claim]


class Source(BaseModel):
    """One labelled evidence snippet handed to us by Person 2 (Search)."""

    id: str  # e.g. "kaynak_1" / "source_1" — must match what appears in prompts
    url: Optional[str] = None
    text: str


class VerdictResult(BaseModel):
    claim_id: str
    verdict: str = Field(
        description="One of: SUPPORTED, REFUTED, NOT ENOUGH EVIDENCE, CONFLICTING"
    )
    explanation: str
    used_sources: List[str] = Field(
        default_factory=list,
        description="IDs of only the sources actually relied on in `explanation`.",
    )


VALID_VERDICTS = {"SUPPORTED", "REFUTED", "NOT ENOUGH EVIDENCE", "CONFLICTING"}


# ============================================================
# 3. SYSTEM PROMPTS (validated in the AI Studio playground)
# ============================================================

EXTRACTION_SYSTEM_PROMPT = """\
Sen uzman bir fact-checking veri ayrıştırıcısısın. Görevin, sana verilen
metindeki doğrulanabilir, nesnel ve somut iddiaları (claims) bulmak ve
bunları tekil, bağımsız cümleler halinde parçalamaktır.

Kurallar:
- Öznel yargıları, yorumları ve duyguları tamamen yoksay.
- Bir cümlede birden fazla iddia varsa, bunları ayrı ayrı maddelere böl.
- Zamirleri (o, bu, şu) bağlama göre asıl isimlerle değiştir.
- Bir kurumun veya kişinin açıklaması aktarılıyorsa, iddiayı o açıklamayı
  yapan kurum/kişinin adıyla birlikte yaz (ör. "X kurumu ... açıklamıştır"),
  çünkü bu iddianın kendisi aranacak olgu haline gelir.
- Gelecek tahminlerini, öznel yargıları ve ölçülemez/muğlak ifadeleri
  is_verifiable=false olarak işaretle (yine de listeye ekleyebilirsin).
- Sadece verilen JSON şemasında çıktı ver, başka hiçbir açıklama ekleme.
"""

VERDICT_SYSTEM_PROMPT = """\
Sen kesinlikle tarafsız bir doğruluk kontrol (fact-checking) yargıcısın.
Sana bir 'İddia' ve referans numaralarıyla birlikte 'Kanıtlar' verilecek.

Kurallar:
- Kendi içsel bilgini KESİNLİKLE kullanma. Sadece sana verilen kanıtlara
  dayanarak karar ver.
- Kararını (verdict) şu 4 kategoriden birine oturt:
  - SUPPORTED: Kanıt, iddiayı tamamen doğruluyor.
  - REFUTED: Kanıt, iddiayı yalanlıyor veya aksini söylüyor. İddianın bir
    kısmı doğru olsa bile, geri kalanı kanıtla çelişiyorsa bütünü REFUTED say.
  - NOT ENOUGH EVIDENCE: Kanıt, iddiayı doğrulamak/yalanlamak için yeterli
    detay içermiyor.
  - CONFLICTING: Kanıt metinlerinin kendi içinde çelişkili ifadeleri var.
- 'explanation' kısmında, hangi bilgiyi hangi kanıttan aldıysan, o cümlenin
  sonuna [kaynak_id] şeklinde köşeli parantez içinde atıf yap.
- Konuyla alakasız kanıtları (tuzak kanıtları) tamamen yoksay ve
  'used_sources' listesine ekleme.
- 'used_sources' listesine SADECE açıklama içinde gerçekten atıf yaptığın
  kaynakların ID'lerini ekle.
- Sadece verilen JSON şemasında çıktı ver, başka hiçbir açıklama ekleme.
"""


# ============================================================
# 4. ENGINE
# ============================================================

class FactCheckEngine:
    """Thin, stateless wrapper around the two Gemini calls this module owns."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEY in your "
                "environment (see .env.example) or pass api_key=... explicitly."
            )
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature

    # -- internal helper --------------------------------------------------

    def _model(self, system_instruction: str, response_schema) -> genai.GenerativeModel:
        return genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction,
            generation_config={
                "temperature": self.temperature,
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            },
        )

    # -- public API (this is the contract Backend/Person 3 calls) --------

    def extract_claims(self, text: str) -> ExtractionResult:
        """Split raw user text into atomic, verifiable claims."""
        model = self._model(EXTRACTION_SYSTEM_PROMPT, ExtractionResult)
        response = model.generate_content(f"Metin: {text}")
        data = json.loads(response.text)
        return ExtractionResult.model_validate(data)

    def verify_claim(self, claim_text: str, sources: List[Source]) -> VerdictResult:
        """Judge a single claim against labelled evidence snippets."""
        evidence_block = "\n".join(f"[{s.id}]: {s.text}" for s in sources)
        prompt = f"İddia: {claim_text}\n\nKanıtlar:\n{evidence_block}"

        model = self._model(VERDICT_SYSTEM_PROMPT, VerdictResult)
        response = model.generate_content(prompt)
        data = json.loads(response.text)
        result = VerdictResult.model_validate(data)

        if result.verdict not in VALID_VERDICTS:
            raise ValueError(
                f"Model returned an unexpected verdict label: {result.verdict!r}"
            )
        return result


# ============================================================
# 5. SMOKE TEST (run this file directly: `python ai_engine.py`)
# ============================================================

if __name__ == "__main__":
    engine = FactCheckEngine()

    print("--- CLAIM EXTRACTION ---")
    extraction = engine.extract_claims(
        "Apple Türkiye'de fabrika açtı ve bence bu harika bir karar."
    )
    print(extraction.model_dump_json(indent=2))

    print("\n--- VERDICT (Porsche example from the design chat) ---")
    verdict = engine.verify_claim(
        claim_text="Porsche 911 GT3 RS modeli V8 motora sahiptir.",
        sources=[
            Source(
                id="kaynak_1",
                text=(
                    "Porsche 911 GT3 RS, arka kısma yerleştirilmiş 4.0 litrelik "
                    "atmosferik 6 silindirli (flat-six) boxer motor kullanmaktadır. "
                    "Model serisinde V8 motor seçeneği hiçbir zaman sunulmamıştır."
                ),
            )
        ],
    )
    print(verdict.model_dump_json(indent=2))
