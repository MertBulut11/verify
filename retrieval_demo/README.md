# Retrieval Demo

Bu modül, verilen bir iddia için internetten ilgili kaynakları bulur
ve iddiayla en alakalı kanıt parçalarını seçer.

## Özellikler

- Tavily ile otomatik web araması(Tavily demo için sonradan değişecek)
- Requests ile web sayfası çekme
- Playwright fallback
- BeautifulSoup ile HTML temizleme
- Chunking ve overlap
- Türkçe / çok dilli embedding
- Cosine similarity
- Duplicate içerik filtreleme
- Farklı kaynaklardan kanıt seçme
- JSON çıktı üretme
- Temel hata yönetimi

## Kurulum

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt