import os
import re
import json
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tavily import TavilyClient
from playwright.sync_api import sync_playwright

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from urllib.parse import urlparse
from difflib import SequenceMatcher


# ==========================================================
# 1. GENEL AYARLAR
# ==========================================================

ARAMA_SONUCU_SAYISI = 8
TOP_KANIT_SAYISI = 5

CHUNK_BOYUTU = 100
CHUNK_OVERLAP = 20

DUPLICATE_ESIGI = 0.88

JSON_DOSYA_ADI = "retrieval_sonucu.json"


# ==========================================================
# 2. .ENV DOSYASINI YÜKLE
# ==========================================================

load_dotenv()

TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY"
)


if not TAVILY_API_KEY:

    print(
        "\n[HATA] TAVILY_API_KEY bulunamadı."
    )

    print(
        ".env dosyasını kontrol et."
    )

    exit()


# ==========================================================
# 3. TAVILY CLIENT
# ==========================================================

try:

    tavily = TavilyClient(
        api_key=TAVILY_API_KEY
    )

except Exception as hata:

    print(
        "\n[HATA] Tavily başlatılamadı:"
    )

    print(
        hata
    )

    exit()


# ==========================================================
# 4. EMBEDDING MODELİ
# ==========================================================

print(
    "\nEmbedding modeli yükleniyor..."
)

try:

    model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

except Exception as hata:

    print(
        "\n[HATA] Embedding modeli yüklenemedi:"
    )

    print(
        hata
    )

    exit()


# ==========================================================
# 5. KULLANICIDAN İDDİA AL
# ==========================================================



# ==========================================================
# 6. HTML TEMİZLEME
# ==========================================================

def html_metin_temizle(html):

    if not html:

        return None

    try:

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        # Gereksiz HTML alanları
        for etiket in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "noscript",
            "svg"
        ]):

            etiket.decompose()


        paragraflar = soup.find_all(
            "p"
        )


        temiz_paragraflar = []


        for paragraf in paragraflar:

            metin = paragraf.get_text(
                separator=" ",
                strip=True
            )

            if len(metin) >= 40:

                temiz_paragraflar.append(
                    metin
                )


        tam_metin = "\n".join(
            temiz_paragraflar
        )


        if not tam_metin.strip():

            return None


        return tam_metin


    except Exception as hata:

        print(
            f"[UYARI] HTML temizlenemedi: {hata}"
        )

        return None


# ==========================================================
# 7. REQUESTS İLE SAYFA ÇEKME
# ==========================================================

def requests_ile_metin_cek(url):

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/136.0 Safari/537.36"
        )
    }


    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )


        response.raise_for_status()


        return html_metin_temizle(
            response.text
        )


    except requests.Timeout:

        print(
            "[UYARI] Requests zaman aşımına uğradı."
        )

        return None


    except requests.HTTPError as hata:

        print(
            f"[UYARI] HTTP hatası: {hata}"
        )

        return None


    except requests.RequestException as hata:

        print(
            f"[UYARI] Requests hatası: {hata}"
        )

        return None


    except Exception as hata:

        print(
            f"[UYARI] Beklenmeyen requests hatası: {hata}"
        )

        return None


# ==========================================================
# 8. PLAYWRIGHT İLE SAYFA ÇEKME
# ==========================================================

def playwright_ile_metin_cek(url):

    print(
        "→ Playwright deneniyor..."
    )


    try:

        with sync_playwright() as p:

            browser = p.chromium.launch(
                headless=True
            )


            try:

                page = browser.new_page(
                    viewport={
                        "width": 1280,
                        "height": 900
                    }
                )


                page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=30000
                )


                page.wait_for_timeout(
                    2000
                )


                html = page.content()


                return html_metin_temizle(
                    html
                )


            finally:

                browser.close()


    except Exception as hata:

        print(
            f"[UYARI] Playwright başarısız: {hata}"
        )

        return None


# ==========================================================
# 9. WEB SAYFASINDAN METİN ÇEK
# ==========================================================

def web_sayfasindan_metin_cek(url):

    if not url:

        return None


    # Önce requests kullan
    metin = requests_ile_metin_cek(
        url
    )


    # Yeterli içerik geldiyse kullan
    if metin and len(metin) >= 500:

        return metin


    print(
        "→ Requests ile yeterli metin alınamadı."
    )


    # Sonra Playwright dene
    metin = playwright_ile_metin_cek(
        url
    )


    if metin and len(metin) >= 100:

        return metin


    return None


# ==========================================================
# 10. CHUNKING + OVERLAP
# ==========================================================

def metni_parcala(
    metin,
    parca_boyutu=100,
    overlap=20
):

    if not metin:

        return []


    if overlap >= parca_boyutu:

        raise ValueError(
            "Overlap, parça boyutundan küçük olmalıdır."
        )


    kelimeler = metin.split()

    parcalar = []

    adim = parca_boyutu - overlap


    for i in range(
        0,
        len(kelimeler),
        adim
    ):

        parca = kelimeler[
            i:i + parca_boyutu
        ]


        # Çok küçük parçaları alma
        if len(parca) >= 30:

            parcalar.append(
                " ".join(parca)
            )


    return parcalar


# ==========================================================
# 11. TAVILY WEB SEARCH
# ==========================================================

def internette_ara(
    sorgu,
    sonuc_sayisi=8
):

    print(
        "\nİnternette kaynaklar aranıyor..."
    )


    try:

        cevap = tavily.search(
            query=sorgu,
            search_depth="basic",
            max_results=sonuc_sayisi,
            include_answer=False
        )


        sonuclar = cevap.get(
            "results",
            []
        )


        if not sonuclar:

            print(
                "[UYARI] Tavily hiçbir sonuç döndürmedi."
            )


        return sonuclar


    except Exception as hata:

        print(
            f"\n[HATA] Web araması başarısız: {hata}"
        )

        return []


# ==========================================================
# 12. DOMAIN BUL
# ==========================================================

def domain_bul(url):

    try:

        domain = urlparse(
            url
        ).netloc.lower()


        if domain.startswith(
            "www."
        ):

            domain = domain[4:]


        if not domain:

            return "bilinmeyen-kaynak"


        return domain


    except Exception:

        return "bilinmeyen-kaynak"


# ==========================================================
# 13. METNİ NORMALİZE ET
# ==========================================================

def metni_normalize_et(metin):

    metin = metin.lower()


    metin = re.sub(
        r"\s+",
        " ",
        metin
    )


    metin = re.sub(
        r"[^\w\s]",
        "",
        metin
    )


    return metin.strip()


# ==========================================================
# 14. İKİ METNİN BENZERLİĞİ
# ==========================================================

def metin_benzerligi(
    metin1,
    metin2
):

    metin1 = metni_normalize_et(
        metin1
    )

    metin2 = metni_normalize_et(
        metin2
    )


    return SequenceMatcher(
        None,
        metin1,
        metin2
    ).ratio()


# ==========================================================
# 15. DUPLICATE FİLTRELE
# ==========================================================

def duplicate_filtrele(
    sonuclar,
    esik=0.88
):

    benzersiz_sonuclar = []


    for aday in sonuclar:

        duplicate_mi = False


        for secilmis in benzersiz_sonuclar:

            oran = metin_benzerligi(
                aday["metin"],
                secilmis["metin"]
            )


            if oran >= esik:

                duplicate_mi = True

                break


        if not duplicate_mi:

            benzersiz_sonuclar.append(
                aday
            )


    return benzersiz_sonuclar


# ==========================================================
# 16. KAYNAK ÇEŞİTLİLİĞİ
# ==========================================================

def cesitli_kaynaklari_sec(
    sonuclar,
    limit=5
):

    secilenler = []

    kullanilan_domainler = set()


    # İlk tur:
    # Her domain'den bir tane al
    for sonuc in sonuclar:

        domain = sonuc[
            "domain"
        ]


        if domain not in kullanilan_domainler:

            secilenler.append(
                sonuc
            )

            kullanilan_domainler.add(
                domain
            )


        if len(secilenler) >= limit:

            return secilenler


    # Yeterince farklı domain yoksa
    # kalan en iyi sonuçlarla tamamla
    for sonuc in sonuclar:

        if sonuc not in secilenler:

            secilenler.append(
                sonuc
            )


        if len(secilenler) >= limit:

            break


    return secilenler


# ==========================================================
# 17. JSON DOSYASINA KAYDET
# ==========================================================

def json_kaydet(
    iddia,
    kanitlar,
    dosya_adi
):

    veri = {
        "iddia": iddia,
        "kanit_sayisi": len(
            kanitlar
        ),
        "kanitlar": []
    }


    for sira, kanit in enumerate(
        kanitlar,
        start=1
    ):

        veri["kanitlar"].append(
            {
                "sira": sira,
                "metin": kanit["metin"],
                "kaynak": kanit["kaynak"],
                "domain": kanit["domain"],
                "url": kanit["url"],
                "benzerlik_skoru": round(
                    kanit["skor"],
                    4
                )
            }
        )


    try:

        with open(
            dosya_adi,
            "w",
            encoding="utf-8"
        ) as dosya:

            json.dump(
                veri,
                dosya,
                ensure_ascii=False,
                indent=4
            )


        print(
            f"\nJSON sonucu kaydedildi: {dosya_adi}"
        )


    except Exception as hata:

        print(
            f"\n[UYARI] JSON dosyası kaydedilemedi: {hata}"
        )

print(
    "===================================="
)

def search(claim_text: str, max_sources: int = 5):
    """
    Verilen iddia için internette kaynak arar,
    ilgili metinleri bulur ve backend'e döndürür.
    """

    # 1. İnternette ara
    arama_sonuclari = internette_ara(
        claim_text,
        sonuc_sayisi=ARAMA_SONUCU_SAYISI
    )

    if not arama_sonuclari:
        return []

    # 2. Kaynaklardan metinleri topla
    tum_parcalar = []

    for sonuc in arama_sonuclari:

        url = sonuc.get("url")

        if not url:
            continue

        baslik = sonuc.get(
            "title",
            "Başlık yok"
        )

        metin = web_sayfasindan_metin_cek(url)

        if not metin:
            continue

        parcalar = metni_parcala(
            metin,
            parca_boyutu=CHUNK_BOYUTU,
            overlap=CHUNK_OVERLAP
        )

        domain = domain_bul(url)

        for parca in parcalar:
            tum_parcalar.append({
                "metin": parca,
                "kaynak": baslik,
                "url": url,
                "domain": domain
            })

    if not tum_parcalar:
        return []

    # 3. Embedding oluştur
    metinler = [
        parca["metin"]
        for parca in tum_parcalar
    ]

    parca_embeddingleri = model.encode(
        metinler,
        show_progress_bar=False
    )

    iddia_embeddingi = model.encode(
        [claim_text]
    )

    # 4. Benzerlik hesapla
    benzerlikler = cosine_similarity(
        iddia_embeddingi,
        parca_embeddingleri
    )[0]

    for i in range(len(tum_parcalar)):
        tum_parcalar[i]["skor"] = float(
            benzerlikler[i]
        )

    # 5. En alakalı parçaları üste getir
    tum_parcalar.sort(
        key=lambda x: x["skor"],
        reverse=True
    )

    # 6. Aynı içerikleri temizle
    benzersiz_parcalar = duplicate_filtrele(
        tum_parcalar,
        esik=DUPLICATE_ESIGI
    )

    # 7. Farklı kaynaklardan sonuç seç
    en_iyi_kanitlar = cesitli_kaynaklari_sec(
        benzersiz_parcalar,
        limit=max_sources
    )

    # 8. Backend'in anlayacağı formata çevir
    kaynaklar = []

    for i, kanit in enumerate(
        en_iyi_kanitlar,
        start=1
    ):
        kaynaklar.append({
            "id": f"source_{i}",
            "url": kanit["url"],
            "text": kanit["metin"]
        })

    return kaynaklar