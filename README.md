<div align="center">

# 🧠 LeWM-Finans · JEPA Financial World Model

**BIST için fiyat barlarından öğrenen dünya modeli — latent piyasa analizi, anomali radarı ve tarihsel senaryo araştırması.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white) ![JEPA](https://img.shields.io/badge/Architecture-JEPA-7353BA) ![Market](https://img.shields.io/badge/Market-BIST-1B998B) ![Data](https://img.shields.io/badge/Data-Local%20OHLCV-555)

**[📖 Genel Bakış](#-genel-bakış) · [✨ Özellikler](#-özellikler) · [🚀 Kurulum](#-kurulum) · [🎮 Kullanım](#-kullanım) · [📊 Çıktılar](#-çıktılar) · [🧪 Doğrulama](#-doğrulama) · [📚 Referanslar](#-referanslar)**

</div>
![LeWM-Finans Sistem Mimarisi](images/lewm-mimari.png)
## 📖 Genel Bakış

**LeWM-Finans**, [LeWorldModel](https://arxiv.org/abs/2603.19312) çalışmasındaki Joint-Embedding Predictive Architecture (JEPA) yaklaşımını yerel **MetaStock CSV / OHLCV** finansal zaman serilerine uyarlayan Python araştırma paketidir.

Video kareleri yerine **fiyat barları** kodlanır. Transformer encoder, piyasa geçmişini kompakt bir latent temsile dönüştürür; predictor gelecekteki pencerenin latent temsilini tahmin etmeye çalışır. **SIGReg**, gömmelerin tek noktaya çökmesini önlemeyi amaçlar.

```text
📁 LOCAL OHLCV CSV
       ↓
🧮 6 BAR FEATURES + ROBUST NORMALIZATION
       ↓
🧩 PATCH EMBEDDING → TRANSFORMER ENCODER
       ↓
🧠 JEPA PREDICTOR + SIGReg
       ↓
🌐 LATENT MARKET SPACE
       ├── 🚨 Surprise / Anomaly Detection
       ├── 🧭 Market Crash Radar
       ├── 🔎 k-NN Scenario Discovery
       ├── 📋 AL Candidate Screening
       └── 🧪 Hindsight / Probe Validation
```

> [!IMPORTANT]
> Bu sistem **kesin fiyat tahmini** üretmez. Latent benzerlik, anomali ve kural tabanlı tarama çıktıları araştırma amaçlıdır. Geleceği bilen bir model olarak yorumlanmamalıdır.

## ✨ Özellikler

### 🧠 JEPA Dünya Modeli

- Transformer encoder + latent predictor mimarisi
- Gelecek `--tgt` barın gömmesini tahmin etme
- **SIGReg** (rastgele projeksiyon + Epps–Pulley istatistiği)
- EMA ve stop-gradient gerektirmeyen eğitim yaklaşımı
- Tek GPU, AMP desteği, checkpoint ile eğitime devam

### 📈 OHLCV Özellik Mühendisliği

Bar başına **6 özellik**: log getiri, `log(C/O)`, `log(H/L)`, kapanış konumu `(C−L)/(H−L)`, önceki kapanışa göre gap `log(O/Cönce)` ve log hacim. Normalizasyon, bağlam penceresinin **medyan/MAD** istatistiklerine dayanır.

Desteklenen periyotlar: `gunluk`, `60dk`, `240dk`, `haftalik`, `aylik`.

### 🚨 Anomali ve Piyasa Radarı

- Latent tahmin hatasından **robust sigma (σ)** skoru
- Kayan 252 bar medyan/MAD kalibrasyonu
- `|σ| > 3` gibi anomali işaretleri
- Piyasadaki yüksek-σ hisse oranı ve 5-bar medyan getiri ile rejim radarı
- **GZ (gerçek zamanlı)** görünüm: skorun gerçekten bilinebildiği tarihe kaydırılması
- Çöküş yakalama ve yanlış alarm değerlendirmesi

### 🔎 Latent k-NN Senaryo Analizi

- Tahmin edilen latent hedefe benzeyen **25 tarihsel hedef pencere**
- Tarihsel komşuların gerçekleşmiş getirilerinden medyan, p10–p90 ve yükselen oranı
- Tüm hisselerde ortak referans havuzuyla senaryo taraması
- PCA 2B embedding haritası

### 📋 AL Aday Listesi

- **K1:** GZ sigma risk filtresi
- **K2–K6:** MA20/MA50 trendi, göreli güç, 5-bar getiri ve hacim teyitleri
- **6 puan:** GÜÇLÜ AL · **5 puan:** AL · **4 puan:** İZLE
- Piyasa ALARM durumunda isteğe bağlı listeyi durdurma
- Tarihsel doğrulama ve tarih kırpmalı hindsight değerlendirmesi

> [!CAUTION]
> **AL** etiketleri yazılımın kural tabanlı sınıflandırmasıdır; emir, yatırım tavsiyesi veya doğrulanmış getiri garantisi değildir.

### 🧪 Kontrol Deneyleri

- Dondurulmuş embedding ile **8-bar yön probe** (lojistik regresyon)
- Aynı probe ile **ham fiyat/hacim istatistikleri baseline** karşılaştırması
- Zamansal doğrulama ve geçmişe dönük senaryo değerlendirmesi
- Girdi sızıntısı ile **checkpoint/ağırlık sızıntısı** ayrımının açık raporlanması

## 🚀 Kurulum

### 1. Bağımlılıkları yükleyin

```powershell
pip install -r requirements.txt
```

NVIDIA GPU için PyTorch'un CUDA sürümünü [resmî kurulum sayfasından](https://pytorch.org/get-started/locally/) sisteminize uygun olarak seçin. CPU ile de çalışır; eğitim daha yavaştır.

### 2. Proje yapısı

```text
LeWM-Finans/
├── veri_periyot.py          # Mevcut yerel CSV okuyucu (değiştirilmez)
├── lewm_sigreg.py           # SIGReg / Epps–Pulley
├── lewm_model.py            # Transformer encoder + predictor
├── lewm_veri.py             # OHLCV pencere veri seti
├── lewm_egitim.py           # Eğitim / GPU / AMP / checkpoint
├── lewm_uygula.py           # Sigma, embedding, k-NN, probe
├── lewm_geriye_donuk.py     # Tarih kırpmalı hindsight testi
├── lewm_piyasa_radari.py    # Kesitsel piyasa rejimi radarı
├── lewm_al_listesi.py       # AL aday taraması / doğrulama
├── lewm_rapor.py            # Grafik ve CSV raporları
├── lewm_cli.py              # Ana komut satırı girişi
├── requirements.txt
└── demo_ciktilar/           # Sentetik veri örnekleri
```

Tüm `lewm_*` dosyaları `veri_periyot.py` ile **aynı klasörde** bulunmalıdır. Varsayılan CSV dizinleri `veri_periyot.py` içinden okunur; istenirse komut satırından değiştirilir.

## 🎮 Kullanım

### 🎓 Model eğitimi

```powershell
# Günlük veri, tüm hisseler
python lewm_cli.py egit --periyot gunluk --epoch 30

# Belirli hisseler
python lewm_cli.py egit --hisseler THYAO,ASELS,GARAN --epoch 30
```

### 🔍 Analiz ve embedding haritası

```powershell
python lewm_cli.py uygula --hisse THYAO --harita --probe

# Eğitim ve uygulama birlikte
python lewm_cli.py hepsi --hisse THYAO
```

### ⏯️ Kaldığı yerden eğitim

```powershell
python lewm_cli.py egit --devam --epoch 12 --derinlik 4 --emb 192 --batch 128
```

Her epoch sonunda checkpoint güncellenir. `Ctrl+C` ile durdurup **aynı komutla** devam edebilirsiniz. `--epoch` ek epoch değil, **toplam hedef epoch** sayısıdır. Model mimarisi değişirse devam işlemi hata verir.

### ⏱️ Farklı periyotlar ve veri dizinleri

```powershell
python lewm_cli.py hepsi --periyot 60dk --ctx 256 --tgt 9 --hisse THYAO
python lewm_cli.py hepsi --periyot haftalik --ctx 64 --tgt 4 --hisse THYAO
python lewm_cli.py egit --gunluk "D:\farkli\yol\gun-csv" --saatlik "D:\farkli\yol\60dk-csv"
```

### 🌐 Piyasa geneli senaryo taraması

```powershell
python lewm_cli.py uygula --tum --senaryo-tarama --senaryo-ref 500
```

### 🚨 Piyasa radarı

```powershell
python lewm_piyasa_radari.py --son-bar 60 --cokus-esik -5
```

### 📋 AL aday listesi

```powershell
python lewm_al_listesi.py
python lewm_al_listesi.py --esik-skor 6 --top 20
python lewm_al_listesi.py --senaryo --piyasa-onay
```

### 🕰️ Geçmiş tarih simülasyonu

```powershell
python lewm_geriye_donuk.py --son-tarih 2026-10-16 --kontrol-tarih 2026-10-23 --top 20
python lewm_al_listesi.py --son-tarih 2026-10-16 --kontrol-tarih 2026-10-23
```

## ⚙️ Model Parametreleri

| Parametre | Varsayılan | Açıklama |
|---|---:|---|
| `--periyot` | `gunluk` | `gunluk`, `60dk`, `240dk`, `haftalik`, `aylik` |
| `--hisseler` | Tümü | Virgülle ayrılmış semboller |
| `--epoch` | `30` | Toplam hedef epoch |
| `--batch` | `256` | Batch boyutu |
| `--ctx` | `64` | Bağlam penceresi (bar) |
| `--tgt` | `8` | Gelecek hedef pencere (bar) |
| `--lam` | `0.1` | SIGReg kayıp ağırlığı |
| `--emb` | `256` | Latent gömme boyutu |
| `--derinlik` | `6` | Transformer katman sayısı |
| `--kafa` | `8` | Attention head sayısı |
| `--patch` | `4` | Patch bar uzunluğu |
| `--lr` | `3e-4` | Warmup + cosine öğrenme oranı |
| `--cikti` | `lewm_cikti` | Çıktı klasörü |

`--patch`, `ctx` ve `tgt` değerlerini tam bölmezse ortak bölenlerine göre otomatik küçültülür.

## 📊 Çıktılar

Çıktılar varsayılan olarak `lewm_cikti/` klasörüne yazılır.

| Dosya | Açıklama |
|---|---|
| `lewm_model_eniyi.pt` | En iyi doğrulama kaybına sahip checkpoint |
| `lewm_model_son.pt` | Son epoch checkpoint |
| `egitim_eğrileri.png` | Eğitim ve doğrulama kayıpları |
| `surprise_<HISSE>.csv` | Fiyat, sürpriz ve robust sigma |
| `surprise_<HISSE>.png` | Fiyat ve anomali grafiği |
| `tarama_son_gommeler.csv` | Son latent gömmeler |
| `embedding_tumu.csv` | Tarihsel embedding pencereleri |
| `embedding_haritasi.png` | PCA 2B latent harita |
| `embedding_pca.csv` | PC1 / PC2 koordinatları |
| `tahmin_<HISSE>.csv` | k-NN tarihsel senaryolar |
| `senaryo_tarama.csv` | Piyasa geneli senaryo sıralaması |
| `piyasa_radari.csv`, `piyasa_radari.png` | Rejim radarı |
| `piyasa_radari_yakalama.csv` | Çöküş/alarm eşleşmeleri |
| `al_listesi.csv` | Güncel aday listesi |
| `al_dogrulama.csv` | Tarihsel kural değerlendirmesi |
| `al_degerlendirme.csv` | Hindsight AL değerlendirmesi |
| `geriye_donuk_rapor.csv` | Geçmiş tarih sürpriz taraması |

### 📐 Sigma nasıl yorumlanır?

`σ`, latent tahmin hatasının kayan medyan/MAD referansına göre standartlaştırılmış sapmasıdır. `|σ| > 3`, **modele göre olağandışı** bir gözlemi işaret eder; tek başına fiyatın yükseleceği veya düşeceği anlamına gelmez.

### 🎯 Senaryo tablosu nasıl yorumlanır?

Modelin tahmin ettiği gelecek gömmesine en yakın tarihsel hedef gömmeleri bulunur. Bunların **geçmişte gerçekleşmiş** getirileri özetlenir: medyan, p10–p90 ve pozitif getiri oranı. Bu değerler olasılık kalibrasyonu yapılmış bir gelecek fiyat tahmini değildir.

## 🧪 Doğrulama ve Araştırma Disiplini

### 🔬 Probe / baseline karşılaştırması

Aynı 8-bar yön sınıflandırması, **(A) öğrenilmiş embedding** ve **(B) ham fiyat/hacim istatistikleri** ile ayrı ayrı eğitilir. Ham baseline; 8/16/32-bar getiri, volatilite, hacim rejimi ve 64-bar yol konumu gibi değişkenler kullanır.

```text
[PROBE] 8-bar yön tahmini — örnek çıktı
Embedding           : %50.7
Ham baseline        : %52.4
Şans / sınıf tabanı : %53.2
Sonuç               : Ek bilgi gösterilemedi
```

Örnekteki yüzdeler **temsili çıktıdır**; doğrulanmış canlı performans değildir. Projedeki yorum kuralı: embedding baseline'ı **2 puandan fazla** geçerse `VAR`, ±2 puan içindeyse `SINIRDA`, belirgin gerideyse `YOK`. Bu eşik bir araştırma sezgisidir; istatistiksel anlamlılık kanıtı değildir.

### ⏳ Girdi ve ağırlık sızıntısı

- **Bağlam normalizasyonu** yalnızca bağlam istatistiklerini kullanır.
- **Zamansal validation** her hissenin son %10'luk bölgesine dayanır.
- **Hindsight** modu CSV'leri belirtilen son tarihte fiziksel olarak kırpar.
- **Önemli sınır:** Checkpoint bu tarihten sonraki verilerle eğitildiyse **ağırlık sızıntısı** devam eder. Gerçek bir walk-forward değerlendirmesi için her tarih öncesinde modelin yeniden eğitilmesi gerekir.

### 🕒 Gerçek zamanlı sigma gecikmesi

`t` tarihli bağlamın tahmin hatası, takip eden `tgt` bar gerçekleşmeden bilinemez. Bu nedenle canlı değerlendirmede:

```text
gz_sigma(D) = sigma_tarama(D − tgt)
gz_fiyat(D) = fiyat(D)
```

**Tahmin tarihine yazılan sigma**, aynı tarihte biliniyormuş gibi kullanılmamalıdır. Radarın erken uyarı performansı yalnız **GZ görünümünde** değerlendirilmelidir.

### 📉 Piyasa radarı kuralları

| Bileşen | TEDBİR | ALARM |
|---|---:|---:|
| `|σ| > 3` olan hisselerin oranı | `> %1` | `> %4` |
| Piyasa medyan 5-bar getirisi | `< −%3` | `< −%7` |

Bu eşikler yazılımın başlangıç kurallarıdır; istatistiksel olarak evrensel sınırlar değildir. Çöküş olayları varsayılan olarak ileri-5-bar piyasa medyan getirisi `≤ −%5` olan dönemler üzerinden incelenir.

### 🧾 AL listesi puanlaması

| Koşul | Kural |
|---|---|
| K1 — Anomali | `gz_sigma ≤ +1` ve `|gz_sigma| ≤ 3` |
| K2 — Trend | `Close > MA20` |
| K3 — Trend devamı | `MA20 > MA50` |
| K4 — Göreli güç | 20-bar güç > piyasa medyanı |
| K5 — Momentum | `0 < 5-bar getiri ≤ %15` |
| K6 — Hacim | `5-bar hacim ort. / 60-bar hacim ort. ≥ 1` |

Geçmiş günlerin aday listesi ileri-5-bar getirileri ile piyasa medyanına karşı sınanır. Listenin piyasa altında kaldığı dönemler de raporlanmalıdır.

## 🗂️ Demo Verileri

`demo_ciktilar/`, **6 sentetik hisse × 1.500 gün** ve bilerek eklenen yaklaşık **%10–13 fiyat sıçramaları** ile üretilen örnek grafik ve raporları içerir. Bunlar çıktı biçimini göstermek içindir; **gerçek BIST verisinde daha yüksek performans elde edileceğini kanıtlamaz.**

## 🛠️ Pratik İpuçları

- **GPU belleği:** `--batch`, `--emb` ve `--derinlik` değerlerini donanımınıza göre azaltın.
- **SIGReg ağırlığı:** `--lam` değişikliğini tahmin kaybı ve embedding dağılımıyla birlikte değerlendirin.
- **Hedef ufku:** Küçük `--tgt` kısa vadeli anomaliler, daha büyük `--tgt` daha uzun rejimler için denenebilir.
- **Windows Türkçe karakterler:** Gerekirse terminalde `chcp 65001` çalıştırın.
- **Hız / kapsam:** `--senaryo-ref` küçüldükçe k-NN havuzu daralır ve tarama hızlanabilir.

## 🗺️ Yol Haritası

- [ ] Tam walk-forward yeniden eğitim ve değerlendirme
- [ ] İşlem maliyeti / slippage içeren portföy backtest'i
- [ ] Rejim bazlı model karşılaştırmaları
- [ ] Probe için güven aralıkları ve istatistiksel anlamlılık testleri
- [ ] Canlı veri akışı ve gecikme denetimleri
- [ ] Daha kapsamlı otomatik test ve CI

## 🤝 Katkıda Bulunma

Hata bildirimleri, yeniden üretilebilir deneyler ve pull request'ler memnuniyetle karşılanır. Büyük değişikliklerden önce bir issue açılması önerilir.

## 📚 Referanslar

1. Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., Balestriero, R. **LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels** (2026). [arXiv:2603.19312](https://arxiv.org/abs/2603.19312) · [Resmî kod](https://github.com/lucas-maes/le-wm)
2. Balestriero, R., LeCun, Y. **LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics** (2025). [arXiv:2511.08544](https://arxiv.org/abs/2511.08544)
3. Assran, M. ve diğerleri. **I-JEPA / V-JEPA** araştırma ailesi.

## ⚠️ Yasal Uyarı

Bu proje yalnızca **araştırma ve eğitim** amaçlıdır; yatırım tavsiyesi değildir. Geçmiş performans gelecekteki getiriyi garanti etmez. Model, veri, uygulama ve piyasa riskleri kullanıcı tarafından değerlendirilmelidir.

---

<div align="center">

**⭐ Projeyi faydalı bulduysanız GitHub'da yıldız verebilirsiniz.**

Made with 🧠 + 📈 for quantitative BIST research.

</div>
