"""
test_ai_engine.py
------------------
Runs the exact edge cases worked out and validated in the Google AI
Studio playground session, now against the real ai_engine.py module.

This is a manual/exploratory test script, not a strict unit test suite:
LLM output isn't byte-for-byte deterministic even at temperature=0, so
each case prints the result for you to eyeball, plus a couple of soft
sanity checks (e.g. "did it correctly ignore the trap source?").

Usage:
    export GEMINI_API_KEY="..."
    python test_ai_engine.py
"""

from ai_engine import FactCheckEngine, Source


def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_extraction_cases(engine: FactCheckEngine) -> None:
    cases = {
        "Nested claims (inflation + minimum wage)": (
            "Enflasyonun %60'ı aştığı Türkiye'de, dün asgari ücret "
            "17.000 TL olarak açıklandı."
        ),
        "Opinion + fact mix": (
            "Bence Apple'ın Türkiye'de fabrika açıp 5000 kişiyi işe "
            "alması büyük bir hata."
        ),
        "Implicit/pronoun reference": ("O adam dün istifa etti."),
        "Pronouns + implicit subject (Elon Musk / Tesla-like)": (
            "Şirketin CEO'su Elon Musk dün yaptığı açıklamada, yeni "
            "araçlarının 2 saniyenin altında 100 km hıza çıkacağını ve "
            "fiyatının 50 bin doların altında olacağını belirtti. Bence "
            "bu imkansız."
        ),
        "Reported speech (WHO / news sites)": (
            "Dünya Sağlık Örgütü kahvenin kanser yapmadığını açıklasa "
            "da, bazı yerel haber siteleri günde 3 fincandan fazla "
            "içmenin kalp krizini %40 artırdığını iddia ediyor."
        ),
        "Numeric-data trap (3 independent claims expected)": (
            "Son 10 yılın en sıcak yazını geçiren İstanbul'da, dün baraj "
            "doluluk oranları %20'nin altına düştü ve şehrin sadece 30 "
            "günlük suyu kaldı."
        ),
        "Fully unverifiable text (expect empty or all is_verifiable=false)": (
            "Gelecek yıl borsa kesinlikle çökecek ve herkes çok "
            "fakirleşecek. Zaten dünyanın en güzel şehri İstanbul'dur "
            "ama son zamanlarda insanların enerjisi çok düştü."
        ),
    }

    for title, text in cases.items():
        header(f"EXTRACTION — {title}")
        result = engine.extract_claims(text)
        print(result.model_dump_json(indent=2))


def run_verdict_cases(engine: FactCheckEngine) -> None:
    header("VERDICT — Source-citation trap (Tesla / unrelated source_4)")
    verdict = engine.verify_claim(
        claim_text=(
            "Tesla, 2025 yılında Türkiye'de batarya fabrikası kurmuş ve "
            "bu fabrikada 10.000 kişiyi işe almıştır."
        ),
        sources=[
            Source(
                id="kaynak_1",
                text=(
                    "Tesla'nın 2025 yılı faaliyet raporuna göre, "
                    "Avrupa'daki tek batarya üretim tesisi Almanya'nın "
                    "Berlin şehrinde bulunmaktadır."
                ),
            ),
            Source(
                id="kaynak_2",
                text=(
                    "Türkiye Sanayi Bakanlığı'nın dün yaptığı "
                    "açıklamada, Tesla'nın Türkiye'de sadece "
                    "Supercharger istasyonları kurduğu ve herhangi bir "
                    "fabrika yatırımının bulunmadığı vurgulandı."
                ),
            ),
            Source(
                id="kaynak_3",
                text=(
                    "Şirketin Türkiye ofisinde şu anda sadece 50 kişi "
                    "müşteri hizmetleri, satış ve teknik servis "
                    "departmanlarında çalışmaktadır."
                ),
            ),
            Source(
                id="kaynak_4",
                text=(
                    "Elon Musk, geçen ay Mars misyonu için SpaceX'in "
                    "roket fırlatmalarını artıracağını duyurdu."
                ),
            ),
        ],
    )
    print(verdict.model_dump_json(indent=2))
    assert verdict.verdict == "REFUTED", "Expected REFUTED for the Tesla trap case"
    assert "kaynak_4" not in verdict.used_sources, (
        "Model cited the irrelevant SpaceX/Mars source — it fell for the trap"
    )
    print("✓ Trap source (kaynak_4) correctly ignored.")

    header("VERDICT — Mixed true/false claim (Ankara population)")
    verdict = engine.verify_claim(
        claim_text=(
            "Türkiye'nin başkenti Ankara'dır ve şehrin nüfusu 2025 yılı "
            "itibarıyla 18 milyonu aşarak Avrupa'nın en kalabalık şehri "
            "unvanını almıştır."
        ),
        sources=[
            Source(id="kaynak_1", text="Türkiye Cumhuriyeti'nin başkenti Ankara'dır."),
            Source(
                id="kaynak_2",
                text=(
                    "Resmi istatistik kurumunun 2025 verilerine göre "
                    "Ankara'nın güncel nüfusu yaklaşık 5.8 milyondur."
                ),
            ),
            Source(
                id="kaynak_3",
                text=(
                    "Avrupa'nın sınırları içerisindeki en kalabalık "
                    "şehri, yaklaşık 16 milyonluk nüfusuyla İstanbul'dur."
                ),
            ),
            Source(
                id="kaynak_4",
                text="Ankara, Türkiye'nin İç Anadolu Bölgesi'nde yer alan tarihi bir şehirdir.",
            ),
        ],
    )
    print(verdict.model_dump_json(indent=2))
    assert verdict.verdict in ("REFUTED", "CONFLICTING"), (
        "A partly-false claim must not come back as SUPPORTED"
    )
    print("✓ Partial truth correctly did not get a clean SUPPORTED verdict.")

    header("VERDICT — Not enough evidence")
    verdict = engine.verify_claim(
        claim_text="Porsche 911 GT3 RS modeli V8 motora sahiptir.",
        sources=[
            Source(
                id="kaynak_1",
                text="Porsche 911 GT3 RS, aerodinamik yapısıyla öne çıkan bir performans otomobilidir.",
            )
        ],
    )
    print(verdict.model_dump_json(indent=2))
    assert verdict.verdict == "NOT ENOUGH EVIDENCE"
    print("✓ Correctly refused to guess without engine-relevant evidence.")


if __name__ == "__main__":
    engine = FactCheckEngine()
    run_extraction_cases(engine)
    run_verdict_cases(engine)
    header("ALL TESTS COMPLETED")
