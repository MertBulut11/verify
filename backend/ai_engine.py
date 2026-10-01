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

Provider: Groq (https://groq.com) — OpenAI-compatible Chat Completions
API with "strict" JSON-schema structured output, called via the
official `groq` Python package. (Previously this module used Gemini;
the team switched to Groq.)

Setup
-----
    pip install -r requirements.txt
    cp .env.example .env         # then put your real key in .env
    python ai_engine.py          # runs the smoke tests below
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ConfigDict, Field

# Pick up GROQ_API_KEY (and any other overrides) from a local .env
# file if one exists. No-op if there isn't one / the vars are already
# set in the environment.
load_dotenv()


# ============================================================
# 1. CONFIG
# ============================================================

# Both of these support Groq's "strict" structured-output mode (the
# response is *guaranteed* to match our JSON schema). gpt-oss-20b is
# the faster/cheaper of the two — good for iterating during
# development; swap to openai/gpt-oss-120b for higher quality later
# if needed, via the FACTCHECK_MODEL_NAME env var.
DEFAULT_MODEL_NAME = os.environ.get("FACTCHECK_MODEL_NAME", "openai/gpt-oss-20b")

# temperature=0.0: this system must behave like a rule-following
# classifier, not a creative writer. Hallucination-suppression, not
# style, is the goal.
DEFAULT_TEMPERATURE = 0.0


# ============================================================
# 2. PYDANTIC SCHEMAS (the data contract with Backend / Person 3)
# ============================================================
#
# NOTE: these also double as the JSON schemas sent to Groq's "strict"
# structured-output mode, which requires every field to be required
# (no defaults) and every object to forbid extra properties. That's
# why `model_config = ConfigDict(extra="forbid")` is set on each one,
# and why `used_sources` below has no default — don't remove either
# without re-reading https://console.groq.com/docs/structured-outputs.

class Claim(BaseModel):
    """A single atomic, independently verifiable claim."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    is_verifiable: bool = Field(
        description=(
            "False for predictions, opinions, or vague/unmeasurable "
            "statements that cannot be fact-checked against evidence."
        )
    )


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: List[Claim]


class Source(BaseModel):
    """One labelled evidence snippet handed to us by Person 2 (Search).

    This is only ever an INPUT we build ourselves (never something the
    model has to return), so it doesn't need the strict-schema config
    above.
    """

    id: str  # e.g. "kaynak_1" / "source_1" — must match what appears in prompts
    url: Optional[str] = None
    text: str


class VerdictResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    verdict: str = Field(
        description="One of: SUPPORTED, REFUTED, NOT ENOUGH EVIDENCE, CONFLICTING"
    )
    explanation: str
    used_sources: List[str] = Field(
        description="IDs of only the sources actually relied on in `explanation`. "
        "Use an empty list if none were used."
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
- Eğer hiçbir doğrulanabilir iddia yoksa, boş bir claims listesi döndür.
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
- 'used_sources' alanına SADECE açıklama içinde gerçekten atıf yaptığın
  kaynakların ID'lerini ekle; hiçbiri kullanılmadıysa boş liste ([]) döndür.
- Sadece verilen JSON şemasında çıktı ver, başka hiçbir açıklama ekleme.
"""


# ============================================================
# 4. ENGINE
# ============================================================

def _strict_response_format(model: type[BaseModel], name: str) -> dict:
    """Build the {"type": "json_schema", ...} block Groq's strict mode expects."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": model.model_json_schema(),
        },
    }


class FactCheckEngine:
    """Thin, stateless wrapper around the two Groq calls this module owns."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No Groq API key found. Create a .env file (see "
                ".env.example) with GROQ_API_KEY=..., or pass "
                "api_key=... explicitly. Get a key at "
                "https://console.groq.com/keys"
            )
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature

    # -- internal helper --------------------------------------------------

    def _generate(self, system_prompt: str, user_prompt: str, schema_model, schema_name: str):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=_strict_response_format(schema_model, schema_name),
        )
        content = completion.choices[0].message.content
        return schema_model.model_validate(json.loads(content))

    # -- public API (this is the contract Backend/Person 3 calls) --------

    def extract_claims(self, text: str) -> ExtractionResult:
        """Split raw user text into atomic, verifiable claims."""
        return self._generate(
            EXTRACTION_SYSTEM_PROMPT,
            f"Metin: {text}",
            ExtractionResult,
            "extraction_result",
        )

    def verify_claim(self, claim_text: str, sources: List[Source]) -> VerdictResult:
        """Judge a single claim against labelled evidence snippets."""
        evidence_block = "\n".join(f"[{s.id}]: {s.text}" for s in sources)
        prompt = f"İddia: {claim_text}\n\nKanıtlar:\n{evidence_block}"

        result = self._generate(
            VERDICT_SYSTEM_PROMPT, prompt, VerdictResult, "verdict_result"
        )

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