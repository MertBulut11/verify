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

iddia = input(
    "\nKontrol etmek istediğiniz iddiayı girin:\n> "
).strip()


if not iddia:

    print(
        "\n[HATA] İddia boş bırakılamaz."
    )

    exit()


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


# ==========================================================
# 18. WEB'DE ARA
# ==========================================================

arama_sonuclari = internette_ara(
    iddia,
    sonuc_sayisi=ARAMA_SONUCU_SAYISI
)


if not arama_sonuclari:

    print(
        "\nProgram sonlandırıldı."
    )

    exit()


print(
    "\n===================================="
)

print(
    "BULUNAN KAYNAKLAR"
)

print(
    "===================================="
)


for sira, sonuc in enumerate(
    arama_sonuclari,
    start=1
):

    print(
        f"\n{sira}. "
        f"{sonuc.get('title', 'Başlık yok')}"
    )

    print(
        sonuc.get(
            "url",
            "URL yok"
        )
    )


# ==========================================================
# 19. TÜM KAYNAKLARDAN CHUNK TOPLA
# ==========================================================

tum_parcalar = []


for sira, sonuc in enumerate(
    arama_sonuclari,
    start=1
):

    url = sonuc.get(
        "url"
    )


    baslik = sonuc.get(
        "title",
        "Başlık yok"
    )


    if not url:

        print(
            "[UYARI] URL bulunamadı, atlanıyor."
        )

        continue


    print(
        f"\n[{sira}/{len(arama_sonuclari)}] "
        f"Kaynak inceleniyor:"
    )

    print(
        baslik
    )

    print(
        url
    )


    try:

        metin = web_sayfasindan_metin_cek(
            url
        )


        if not metin:

            print(
                "[UYARI] Bu kaynaktan metin alınamadı."
            )

            continue


        print(
            f"Metin alındı: "
            f"{len(metin)} karakter"
        )


        parcalar = metni_parcala(
            metin,
            parca_boyutu=CHUNK_BOYUTU,
            overlap=CHUNK_OVERLAP
        )


        if not parcalar:

            print(
                "[UYARI] Bu kaynaktan chunk oluşturulamadı."
            )

            continue


        print(
            f"{len(parcalar)} chunk oluşturuldu."
        )


        domain = domain_bul(
            url
        )


        for parca in parcalar:

            tum_parcalar.append(
                {
                    "metin": parca,
                    "kaynak": baslik,
                    "url": url,
                    "domain": domain
                }
            )


    except Exception as hata:

        # Bir kaynak hata verdi diye
        # tüm programı durdurmuyoruz
        print(
            f"[UYARI] Kaynak işlenirken hata oluştu: {hata}"
        )

        continue


# ==========================================================
# 20. CHUNK KONTROLÜ
# ==========================================================

if not tum_parcalar:

    print(
        "\n[HATA] Hiçbir kaynaktan "
        "kullanılabilir metin alınamadı."
    )

    exit()


print(
    "\n===================================="
)

print(
    f"TOPLAM CHUNK: {len(tum_parcalar)}"
)

print(
    "===================================="
)


# ==========================================================
# 21. EMBEDDING
# ==========================================================

print(
    "\nEmbedding'ler oluşturuluyor..."
)


metinler = [
    parca["metin"]
    for parca in tum_parcalar
]


try:

    parca_embeddingleri = model.encode(
        metinler,
        show_progress_bar=True
    )


    iddia_embeddingi = model.encode(
        [iddia]
    )


except Exception as hata:

    print(
        f"\n[HATA] Embedding oluşturulamadı: {hata}"
    )

    exit()


# ==========================================================
# 22. COSINE SIMILARITY
# ==========================================================

try:

    benzerlikler = cosine_similarity(
        iddia_embeddingi,
        parca_embeddingleri
    )[0]


except Exception as hata:

    print(
        f"\n[HATA] Benzerlik hesaplanamadı: {hata}"
    )

    exit()


# ==========================================================
# 23. SKORLARI KAYDET
# ==========================================================

for i in range(
    len(tum_parcalar)
):

    tum_parcalar[i][
        "skor"
    ] = float(
        benzerlikler[i]
    )


# ==========================================================
# 24. SKORA GÖRE SIRALA
# ==========================================================

tum_parcalar.sort(
    key=lambda x: x["skor"],
    reverse=True
)


# ==========================================================
# 25. DUPLICATE TEMİZLE
# ==========================================================

print(
    "\nDuplicate içerikler temizleniyor..."
)


duplicate_oncesi = len(
    tum_parcalar
)


benzersiz_parcalar = duplicate_filtrele(
    tum_parcalar,
    esik=DUPLICATE_ESIGI
)


duplicate_sonrasi = len(
    benzersiz_parcalar
)


print(
    f"Duplicate öncesi: "
    f"{duplicate_oncesi}"
)

print(
    f"Duplicate sonrası: "
    f"{duplicate_sonrasi}"
)


# ==========================================================
# 26. KAYNAKLARI ÇEŞİTLENDİR
# ==========================================================

en_iyi_kanitlar = cesitli_kaynaklari_sec(
    benzersiz_parcalar,
    limit=TOP_KANIT_SAYISI
)


if not en_iyi_kanitlar:

    print(
        "\n[HATA] Kullanılabilir kanıt bulunamadı."
    )

    exit()


# ==========================================================
# 27. TERMINALE YAZDIR
# ==========================================================

print(
    "\n\n===================================="
)

print(
    "İDDİA"
)

print(
    "===================================="
)

print(
    iddia
)


print(
    "\n===================================="
)

print(
    "EN ALAKALI VE ÇEŞİTLİ KANITLAR"
)

print(
    "===================================="
)


for sira, sonuc in enumerate(
    en_iyi_kanitlar,
    start=1
):

    print(
        f"\n{sira}. KANIT"
    )

    print(
        f"Benzerlik Skoru: "
        f"{sonuc['skor']:.4f}"
    )

    print(
        f"\nDomain:"
        f"\n{sonuc['domain']}"
    )

    print(
        f"\nKaynak:"
        f"\n{sonuc['kaynak']}"
    )

    print(
        f"\nURL:"
        f"\n{sonuc['url']}"
    )

    print(
        "\nMetin:"
    )

    print(
        sonuc["metin"]
    )

    print(
        "\n------------------------------------"
    )


# ==========================================================
# 28. JSON DOSYASINA KAYDET
# ==========================================================

json_kaydet(
    iddia,
    en_iyi_kanitlar,
    JSON_DOSYA_ADI
)


print(
    "\n===================================="
)

print(
    "RETRIEVAL TAMAMLANDI"
)

print(
    "===================================="
)