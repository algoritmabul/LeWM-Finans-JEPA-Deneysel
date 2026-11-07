# LeWM-Finans — LeWorldModel (JEPA) ile Hisse Senedi Analizi

`arXiv:2603.19312` numaralı **"LeWorldModel: Stable End-to-End Joint-Embedding
Predictive Architecture from Pixels"** (Maes, Le Lidec, Scieur, LeCun,
Balestriero) makalesindeki yöntemi, MetaStock CSV verilerinize (günlük / 60dk /
240dk / haftalık / aylık OHLCV) uygulayan Python paketi.

Pikseller yerine **fiyat barları** kodlanır; amaç aynıdır: piyasa dinamiklerini
kompakt bir latent (gizli) uzayda öğrenen, uçtan uca kararlı eğitilen bir dünya
modeli kurmak.

---

## 1. Makaledeki yöntem → bu paket

| LeWorldModel (orijinal)               | LeWM-Finans (bu paket)                          |
|---------------------------------------|-------------------------------------------------|
| Ham piksel karesi (video karesi)      | Bar penceresi (örn. 64 günlük bar × 6 özellik)  |
| ViT patch'i                           | `--patch` bar'lık dilim + lineer projeksiyon    |
| Encoder (Transformer)                 | Aynı — pre-LN Transformer, ortalama havuzlama   |
| Predictor (gelecek karenin gömmesi)   | Gelecek `--tgt` barın latent gömmesini tahmin   |
| Kayıp = tahmin MSE + **SIGReg**       | Aynı iki terim, EMA/stop-gradient **yok**       |
| Sürpriz = latent tahmin hatası        | Piyasa anomalisi tespiti (σ skoru)              |
| ~15M param, tek GPU                   | Varsayılan ~5M–15M param, tek GPU               |

**SIGReg nedir?** (LeJEPA, arXiv:2511.08544 — Algoritma 1 bu pakette birebir
uygulanmıştır: `lewm_sigreg.py`) Gömmeler her adımda taze rastgele doğrultulara
projelenir ve her doğrultuda **Epps-Pulley istatistiği** (ampirik karakteristik
fonksiyon ile normal dağılım CF'i arasındaki ağırlıklı L2 uzaklık) hesaplanır.
Gömme bulutu izotropik Gauss'a çekildikçe "her şeyi aynı vektöre yığan" çökmüş
çözümler cezalanır; EMA veya stop-gradient gereksinimi ortadan kalkar.

**Özellik seti (bar başına 6):** log getiri, gövde log(C/O), aralık log(H/L),
kapanış konumu (C−L)/(H−L), gap log(O/Cönce), log hacim. Pencereler bağlamın
**robut (medyan/MAD)** istatistikleriyle normalize edilir — tek bir ani zıplama
anomaliyi seyreltemez ve geleceğe bilgi sızıntısı olmaz.

---

## 2. Dosyalar

```
veri_periyot.py    Sizin mevcut veri okuyucunuz (değişiklik yapılır mı? HAYIR)
lewm_sigreg.py     SIGReg kaybı (Epps-Pulley + rastgele projeksiyon)
lewm_model.py      Encoder + Predictor (Transformer)
lewm_veri.py       VeriPeriyot tabanlı pencere veri seti
lewm_egitim.py     GPU/AMP destekli eğitim döngüsü
lewm_uygula.py     Sürpriz skoru, embedding, latent k-NN gelecek tahmini, probe
lewm_geriye_donuk.py  Hindsight what-if testi (son tarih kırpılmış veriyle)
lewm_piyasa_radari.py Piyasa çöküş radarı (kesitsel sigma + fiyat, GZ görünüm)
lewm_al_listesi.py    "AL" durumdaki hisseler (sızıntısız koşul + doğrulama)
lewm_harman.py        Kural/model harmanı laboratuvarı (10 varyantın 250-bar kıyası)
lewm_rapor.py      Grafikler ve CSV raporlar
lewm_cli.py        KOMUT SATIRI GİRİŞİ (bunu çalıştırırsınız)
requirements.txt   Gerekli paketler
```

Tüm `lewm_*` dosyaları `veri_periyot.py` ile **aynı klasöre** konur
(örn. `D:\new\yeniden\`).

---

## 3. Kurulum

```bat
pip install -r requirements.txt
```

PyTorch GPU sürümü için (NVIDIA kartınız varsa):

```bat
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

CPU sürümü de çalışır (daha yavaş).

---

## 4. Hızlı Başlangıç

`veri_periyot.py`'nın yanındaki klasörde komut istemini açın:

```bat
:: 1) EĞİTİM — günlük veri, klasördeki TÜM hisseler
python lewm_cli.py egit --periyot gunluk --epoch 30

:: 2) UYGULAMA — sürpriz skoru + embedding + gelecek tahmini
python lewm_cli.py uygula --hisse THYAO --harita --probe

:: İkisi birden:
python lewm_cli.py hepsi --hisse THYAO
```

Varsayılan dizinler `veri_periyot.py`'nizdeki gibidir
(`D:\new\yeniden\metastock-gun-csv` ve `...\metastock-60dk-csv`).

### Parçalı eğitim (aşırı ısınma / uzun süreli eğitim için)

Her epoch sonunda checkpoint güncellenir. İstediğiniz an `Ctrl+C` ile
durdurup bilgisayarı serindestikten sonra kaldığı yerden devam
ettirebilirsiniz:

```bat
:: ilk oturum (örn. 3-4 epoch çalışır, sonra Ctrl+C)
python lewm_cli.py egit --devam --epoch 12 --derinlik 4 --emb 192 --batch 128

:: sonraki oturumlar — AYNI komut; kaldığı epoch'tan devam eder
python lewm_cli.py egit --devam --epoch 12 --derinlik 4 --emb 192 --batch 128
```

Notlar:
- `--epoch` her zaman TOPLAM hedefi gösterir (12 = 12. epoch'ta durur).
- `--devam` mimari uyuşmazlığını denetler: farklı `--emb/--derinlik/...`
  ile devam ederseniz net bir hata verir.

### Farklı veri dizini

Her iki sıra da geçerlidir: seçenekler `egit`'ten önce de sonra da yazılabilir.

```bat
python lewm_cli.py egit --gunluk "D:\farkli\yol\gun-csv" --saatlik "D:\farkli\yol\60dk-csv"
```

### 60dk / 240dk / haftalık periyotları

```bat
python lewm_cli.py hepsi --periyot 60dk --ctx 256 --tgt 9 --hisse THYAO
python lewm_cli.py hepsi --periyot haftalik --ctx 64 --tgt 4 --hisse THYAO
```

Not: intraday (60dk) veride ctx=256 bar ≈ 1 ay, tgt=9 bar ≈ 1 gün eder;
günlükte ctx=64 bar ≈ 3 ay, tgt=2 bar ≈ 2 gün.

---

## 5. Parametreler

| Parametre    | Varsayılan | Açıklama                                       |
|--------------|-----------:|------------------------------------------------|
| `--periyot`  | `gunluk`   | gunluk / 60dk / 240dk / haftalik / aylik       |
| `--hisseler` | tümü       | Virgüllü: `THYAO,ASELS,GARAN`                  |
| `--epoch`    | 30         | Epoch sayısı (GPU'da 30–50 önerilir)           |
| `--batch`    | 256        | Batch boyutu (GPU RAM'e göre 256–1024)         |
| `--ctx`      | 64         | Bağlam penceresi (bar)                         |
| `--tgt`      | 8          | Tahmin edilecek ileri pencere (bar)            |
| `--lam`      | 0.1        | SIGReg ağırlığı — makalenin tek hiperparametresi |
| `--emb`      | 256        | Latent gömme boyutu (D)                        |
| `--derinlik` | 6          | Transformer katman sayısı                      |
| `--kafa`     | 8          | Dikkat kafası sayısı                           |
| `--patch`    | 4          | Patch uzunluğu (ctx ve tgt'yi tam bölmeli)     |
| `--lr`       | 3e-4       | Öğrenme oranı (warmup + cosine)                |
| `--cikti`    | lewm_cikti | Çıktı klasörü                                  |

`--patch` otomatik güvenlik içerir: ctx/tgt'yi tam bölmezse gcd'ye küçültülür.

---

## 6. Çıktılar (`lewm_cikti/` klasörü)

| Dosya                          | İçerik                                                  |
|--------------------------------|---------------------------------------------------------|
| `lewm_model_eniyi.pt`          | Doğrulama kaybı en düşük checkpoint (uygulama bunu kullanır) |
| `lewm_model_son.pt`            | Son epoch modeli                                        |
| `egitim_eğrileri.png`          | Toplam / tahmin / SIGReg kayıpları + doğrulama          |
| `surprise_<HISSE>.csv`         | Tarih, kapanış, ham sürpriz, **sigma_skoru**            |
| `surprise_<HISSE>.png`         | Fiyat + ±3σ anomali bantları grafik                     |
| `tarama_son_gommeler.csv`      | Her hissenin SON pencere gömmesi + en benzer tarihsel karşılık |
| `embedding_tumu.csv`           | Son 400 pencere/gömme (harita ve k-NN için)             |
| `embedding_haritasi.png`       | PCA 2B embedding haritası — benzer hisseler yakın       |
| `embedding_pca.csv`            | PC1/PC2 koordinatları                                   |
| `tahmin_<HISSE>.csv`           | Latent k-NN senaryo tablosu (benzerlik + gerçekleşen getiri) |
| `senaryo_tarama.csv`           | Tüm piyasada k-NN senaryo taraması: hisse başına medyan senaryo getirisi, p10–p90, yükselen oranı (medyana göre azalan sıralı) |

**Sürpriz (σ) nasıl okunur?** `sigma_skoru` = o anki latent tahmin hatasının,
son 252 günün kayan medyan/MAD'ine göre kaç robust standart sapma uzakta
olduğu. `|σ| > 3` → modelin bu hareketi "görmediği", olağan dışı piyasa
davranışı (makaledeki "surprise detects implausible events" karşılığı).

**Gelecek tahmini nasıl okunur?** Predictor son bağlam penceresinden gelecek
pencerenin gömmesini öngörür; bu tahmin noktasının latent uzaydaki en yakın
25 tarihsel hedef-gömme listelenir ve o anlarda gerçekte gerçekleşen getiri
dağılımı raporlanır (medyan senaryo, %10–%90 aralık, yükselen oranı).
Bu bir **senaryo/benkzerlik analizi**dir; doğrudan fiyat kehaneti değildir.

**Piyasa geneli senaryo taraması** (`uygula --tum --senaryo-tarama`): yukarıdaki
k-NN senaryo analizini TÜM hisseler için tek seferde çalıştırır. Maliyet hilesi:
k-NN referans havuzu hisse başına yeniden kurulmaz, BİR KEZ kurulup tüm
hisselerle paylaşılır — 666 hisse ~1 hissenin havuz maliyetiyle taranır
(hisse başına kalan iş: son pencereyi gömüp en yakın komşulara bakmak).
`--senaryo-ref N` havuz boyutunu hisse başına N örnek'e indirir (default 500;
küçültmek hızlandırır, komşu havuzu daraltır). Çıktı `senaryo_tarama.csv`:
`medyan_senaryo_getiri_%` azalan sıralı; `p10_%`/`p90_%` senaryo aralığı,
`yukselen_orani_%` komşularda yükselen payı. Dürüstlük notu: bu kolonlar
senaryo keşfi içindir — AL listesindeki gibi bir filtre zinciriyle
birleşmeden tek başına alım kararı üretmez (modelin yön iddiası zayıftır).

**Probe nedir?** Dondurulmuş gömmelerle 8-bar yön sınıflandırması (lojistik
regresyon) + **ham öznitelik kontrol deneyi**. Aynı pencerelerden iki girdi
seti çıkarılıp AYNI probe'la eğitilir: (A) embedding gömmesi, (B) ham
istatistikler (son 8/16/32 bar getiri, 8/16 bar volatilite, hacim rejimi
değişimi, 64-bar yol konumu). Çıktı üçlü karşılaştırmadır:

```
[PROBE] 8-bar yön tahmini (lojistik probe) — kontrol deneyi:
   embedding gömmesi      : %50.7
   ham öznitelik baseline : %52.4  (getiri+vol+hacim, aynı probe)
   şans tabanı            : %53.2
   embedding ek bilgisi   : YOK — ham baseline 1.7 puan daha iyi ...
```

Yargı: embedding, ham baseline'ı **+2 puanın üzerinde** geçiyorsa model basit
trend tekrarının ötesinde bilgi taşıyor demektir (VAR); gerideyse modelin
ekleme değeri yoktur (YOK); ±2 puan içindeyse SINIRDA. Şans tabanına karşı
değil, ham baseline'a karşı okumak doğru kontrol yöntemidir.

---

## 7. Demo çıktıları (`demo_ciktilar/`)

Paket, **sentetik veriyle** (6 sahte hisse × 1500 gün, içine bilinçli %10–13'lük
zıplamalar enjekte edilmiş) çalıştırılmış örnek çıktıları içerir — gerçek
verinizle üretilen tabloların biçimini görmeniz içindir. Not: sentetik veri
yalnızca 8,6k eğitim penceresi içerir; modelin öngörü gücü (ve anomali ayrımı)
gerçek BIST verisinde çok daha yüksek olacaktır. Grafikleri açıp
biçimlendirmeyi inceleyebilirsiniz: `egitim_eğrileri.png`,
`surprise_ORN01.png`, `embedding_haritasi.png`.

---

## 8. İpuçları

- **Eğitim süresi:** ~10M parametre model, 100+ hisse × 20+ yıl günlük veri
  ile GPU'da epoch başına ~1–3 dk. CPU'da önce `--emb 128 --derinlik 3
  --hisseler` ile küçük başlayın.
- **`--lam` ayarı:** SIGReg kaybı eğitim grafiğinde hızla ~0'a düşüyorsa ve
  tahmin MSE sapıyorsa lam'ı düşürün; kayıp dalgalanıyorsa/çökme (tahmin MSE
  hızla 0'a çöküyor) görülüyorsa lam'ı yükseltin (0.5–1.0).
- **`--tgt` seçimi:** Anomali tespiti için küçük tgt (1–2 bar) daha keskin
  sinyal verir; genel "rejim öğrenimi" için 5–8 bar daha dengelidir.
- **Veri sızıntısı yok:** doğrulama bölmesi her hissenin SON %10'u üzerinden
  zaman sırasıyla ayrılır; pencere normalizasyonu yalnızca bağlam istatistiği
  kullanır.
- Windows konsolunda Türkçe karakter sorunu yaşarsanız
  `chcp 65001` komutunu çalıştırın.

---

## 9. Geriye dönük test ("ne yapsaydık, ne olurdu?")

`lewm_geriye_donuk.py` — tarayıcının geçmişte ne söyleseydi ve söyledikleri
gerçekleşmiş miydi sorusunu ölçer:

```powershell
python lewm_geriye_donuk.py --son-tarih 2026-10-16 --kontrol-tarih 2026-10-23 --top 20
```

1. Günlük CSV'ler fiziksel olarak SON TARİH'e kadar kırpılır (sonrası modelin
   erişiminin tamamen dışında — girdi sızıntısı imkânsız)
2. `uygula --tum` ile AYNI kod yoluyla surprise taraması yapılır
3. TOP-N listesi |σ| ile üretilir
4. ORİJİNAL CSV'lerden referans (son tarih) ve kontrol (kontrol tarihi)
   kapanışları okunur; her hissenin gerçek getirişi hesaplanır
5. Özet: liste / ±3σ alarm kümesi / piyasa medyan karşılaştırması +
   |σ| ↔ getiri korelasyonları
6. Rapor: `geriye_donuk_rapor.csv` (tüm hisseler, `listede_mi` bayrağıyla)

Not: mevcut checkpoint son tarihten sonraki verilerle de eğitilmiş olabilir;
betik girdi sızıntısını engeller ama ağırlık sızıntısını içerir. Tam temiz
walk-forward için modelin yalnızca son tarihe kadar eğitilmesi gerekir.
Ayrıca surprise skoru, hedef pencere (tgt bar) tamamlandığı için son
skorlanabilir bağlam tarihi son tarihin ~tgt bar öncesidir — bu modelin
tanımının doğal sonucudur, canlı taramada da böyledir.

---

## 10. Piyasa radarı (çöküş yakalama, gerçek-zamanlı görünüm)

`lewm_piyasa_radari.py` — bireysel sürprizleri **kesitsel** okuyup piyasa
rejimini verir. Çöküş günlerinde çok sayıda hisse AYNI anda ±3σ üstüne
çıkar; bu toplu sapma piyasa alarmıdır:

- **Sigma radarı**: |σ|>3 hisse oranı (Gauss beklentisi ~%0.3;
  >%1 TEDBİR, >%4 ALARM)
- **Fiyat radarı**: piyasa medyan son-5-bar getiri (anlık;
  <-%3 TEDBİR, <-%7 ALARM)

```powershell
python lewm_piyasa_radari.py --son-bar 60 --cokus-esik -5
```

**Bildirim gecikmesi (v2'de dürüstçe modellendi):** sigma satırı t, t
anında kurulan bağlamla İZLEYEN tgt barın gerçekleşen yolu arasındaki
sapmadır; değeri ancak t+tgt kapanışında bilinir. Bu yüzden tablodaki
sigma kolonları **tahmin-tarihlidir**; o gün canlıda görülen okuma
**GZ (gerçek-zamanlı)** kolonlarıdır: `gz_sigma(D) = sigma_tarama(D−tgt)`,
`gz_fiyat(D) = fiyat(D)`. "Radar çöküşten önce uyarmış" görüntüsü tahmin-
tarihli kolonlardan okunursa yanıltıcıdır — canlıda olan GZ'dir.

**Yakalama analizi** (GZ görünüm üzerinden, ayrıca
`piyasa_radari_yakalama.csv`):

- *Çöküş olayları*: piyasa medyan ileri-5-bar getiri ≤ `--cokus-esik`
  (varsayılan −%5) olan gün koşuları. Her olay için radarın ALARM zamanı:
  ÖNCEDEN / BAŞLANGICINDA / SÜRERKEN / GEÇ / KAÇIRDI.
- *ALARM bölümleri*: GZ ALARM'ın kesintisiz döneleri; her bölümün
  zirvesi ve bölüm+işleme ufku içindeki en kötü ileri getiri ile
  GERÇEK / YANLIŞ / AÇIK değerlendirmesi.

Tek model geçişiyle TÜM geçmişin radar serisi üretilir (kayan kalibrasyon
shift(1) yalnız geçmiş sürprizleri kullandığı için girdi sızıntısı yok —
her gün ayrı tarama gerekmez). Fiyat radarı veri sonuna kadar çalışır;
sigma radarı doğası gereği son tgt barı göremez. Çıktılar:
`piyasa_radari.csv` + `piyasa_radari.png`. Model ağırlıkları bugünkü
checkpoint olduğu için ağırlık sızıntısı notu geçerlidir. Eşikler
betiğin başındaki sabitlerle ayarlanır.

---

## 11. AL listesi ("gerçekten al" durumdaki hisseler)

`lewm_al_listesi.py` — piyasa radarının karşı ucu: hangi hisseler AL
durumda? Tanım kanıta dayalıdır ve üç katmandır:

- **K1 — model (risk filtresi):** gz sigma ≤ +1 ve |gz sigma| ≤ 3.
  Geriye dönük test yüksek pozitif σ kuyruğunun piyasa ALTINDA getiri
  verdiğini gösterdi (bölüm 9) — modelin bugünkü dürüst katkısı yön
  iddiası değil, ANOMALİ elemedir. gz kaydırma (`gz(D)=σ(D−tgt)`) ile
  sızıntısız.
- **K2..K6 — fiyat/hacim teyitleri:** Close>MA20, MA20>MA50, 20-bar göreli
  güç > piyasa medyanı, 0 < 5-bar getiri ≤ +%15 (aşırı uzama değil),
  hacim teyidi (5-bar ort / 60-bar ort ≥ 1×).
- **Piyasa katmanı:** radarla aynı eşiklerle GZ rejim; ALARM'da uyarı
  (`--piyasa-onay` ile liste üretimi durdurulabilir).

**SKOR** = geçilen koşul sayısı: 6 → GÜÇLÜ AL, 5 → AL, 4 → İZLE
(`--esik-skor`, varsayılan 5). Ek filtreler: `--min-fiyat` (kuruşluk
eleme), `--top`, `--senaryo` (listedekilere k-NN senaryo kolonu — yavaş).

```powershell
python lewm_al_listesi.py
python lewm_al_listesi.py --esik-skor 6 --top 20
python lewm_al_listesi.py --senaryo --piyasa-onay
```

**Tarihsel doğrulama (dürüstlük):** aynı kural geçmiş her günde yalnız o
gün bilinenlerle uygulanır; listenin ileri-5-bar getirisi piyasa medyanıyla
kıyaslanır (`al_dogrulama.csv` + ÖZET satırları: medyan fark, liste>piyasa
gün oranı, GZ-ALARM günlerinde ayrı fark). Kural tarihsel olarak piyasa
altıysa bunu açıkça gösterir — liste ancak bu tablo destekliyorsa anlamlıdır.
Çıktılar: `al_listesi.csv`, `al_dogrulama.csv`.

**Geriye dönük (hindsight) mod:** veriyi FİZİKSEL olarak son tarihe kadar
kırpıp listeyi o günü bilerek üretir, sonra listedekileri ORİJİNAL CSV'lerin
kontrol-tarihi kapanışlarıyla ölçer (`lewm_geriye_donuk` ile aynı dürüstlük
deseni; kırpılan geçici veri sonunda silinir, `--sakla` ile kalır):

```powershell
python lewm_al_listesi.py --son-tarih 2026-10-16 --kontrol-tarih 2026-10-23
```

DEĞERLENDİRME bölümünde liste üyelerinin 16.10 → 23.10 gerçek getirileri,
PİYASA (tüm hisseler) medyanı, kategori bazlı döküm ve "listedekilerin
piyasayı geçme oranı" verilir; `al_degerlendirme.csv`'ye yazılır. Liste boş
çıkarsa bu da bir karardır (kural o gün AL dememiş) — piyasa özeti yine de
gösterilir. Not: girdi sızıntısı yoktur; ağırlık sızıntısı notu geçerlidir.

### Kural/model harmanı laboratuvarı — `lewm_harman.py`

AL listesinin isabet/fark oranını yükseltmek için kural harmanını SİSTEMATİK
aramak gereken yer: 10 varyant, aynı 250-bar pencerede, aynı doğrulama kodu
ile sızıntısız kıyaslanır (TABAN skor 4/5/6; K1 sert gzσ≤0 / sakin |gzσ|≤1;
7. koşul K7 |gzσ|≤1; hacim ≥1.5×; uzama ≤%10; günlük rel20 ilk 15;
ALARM-gün filtresi). İlk koşu tüm hisselerde sigma geçişi yapar ve
`harman_onbellek.pkl.gz` önbelleği yazar; sonraki taramalar
`--onbellek-kullan` ile model/GPU'suz saniyeler içinde döner:

```powershell
python lewm_harman.py                              # ilk koşu: sigma geçişi + 10 varyant
python lewm_harman.py --onbellek-kullan            # hızlı yeniden tarama
python lewm_harman.py --onbellek-kullan --dogrula-bar 500 --dogrula-ufuk 10
python lewm_harman.py --cikti lewm_cikti_buyuk     # büyük modeli aynı tezgâhta sına
```

Çıktı: `harman_karsilastirma.csv` (varyant × n-gün, boyut, medyan fark%,
pozitif gün%, isabet%, hisse-gün) + fark% sıralı konsol tablosu. Okuma:
TABAN_skor5 satırı mevcut kuraldır; ondan iyi ve boyut_ort≥5 + hisse_gun≥200
şartını geçen varyant adaydır. Kazananı farklı `--dogrula-bar` /
`--dogrula-ufuk` ile yeniden sınayın (çoklu-test riski): pencere değişince
sınıfta kalan varyantı benimsemeyin. Ağırlık sızıntısı TÜM varyantlarda
ortaktır — varyantlar arası kıyas adildir.

---

## 12. Çıktı sözlüğü (tabloları okuma rehberi)

**Senaryo taraması** (`lewm_cli.py uygula --tum --senaryo-tarama`):
Her hissenin son 64 barlık görünümü, geçmiş ~300k pencere içinde latent
uzayda en benzer **k=25** pencereyle eşlenir. Tablodaki sayılar o 25
komşunun **sonraki 8 barlık gerçek getiri dağılımıdır**:

- `medyan` — 25 komşunun ortanca getirişi (modelin ortanca beklentisi)
- `p10` / `p90` — kötü/iyi senaryo kuyruğu (en kötü/en iyi %10'luk dilim)
- `yükselen %` — kârlı kapanan komşu oranı
- Kolay okuma: `medyan>0` + `p10 sığ` + `p90−p10 bandı dar` + `yükselen yüksek`
  = modelin en net olumlu görüşü. Geniş bant = model "her şey olabilir" diyor.
- Dürüstlük: benzerlik ≠ gelecek garantisi; k=25 küçük örneklem; keşif aracıdır.

**Sürpriz taraması** (`surprise_tarama.csv`):
- `sigma` — modelin "bu bar beklediğimden farklı" düzeyi (σ biriminde,
  kayan medyan/MAD'a göre normalize); `|sigma| > 3` = OLAĞANDIŞI
- `skor` — 0-1 arası normalize olağandışılık

**AL listesi** (`al_listesi.csv`):
- `skor` — 0-6 (K1-K6 koşulları; 5 = AL, 6 = GÜÇLÜ AL)
- `gzσ` — model risk filtresi (gerçek-zamanlı, +8 bar gecikmeli sigma;
  `|gzσ| > 3` = ALARM bandı, liste elenir)
- `r5%` / `r20%` — son 5/20 bar getiri; `rel%` = r20 − piyasa medyanı
- `hacim` — 5-bar/60-bar hacim oranı (≥1 = hacim desteği var)

**Doğrulama tablosu** (`al_dogrulama.csv`):
- `liste%` — listedeki hisselerin ufuk-bar ileri getirisi; `piyasa%` —
  tüm hisselerin medyanı; `fark%` = liste − piyasa
- `isabet%` — piyasa medyanını geçen liste hisselerinin oranı
- `—` — ufuk henüz dolmadı (gelecek gerçekleşmedi); birkaç bar sonra
  aynı koşu bu satırları otomatik doldurur

---

## 13. Referanslar

1. Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., Balestriero, R.
   *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture
   from Pixels.* arXiv:2603.19312, 2026. Resmî kod: github.com/lucas-maes/le-wm
2. Balestriero, R., LeCun, Y. *LeJEPA: Provable and Scalable Self-Supervised
   Learning Without the Heuristics.* arXiv:2511.08544, 2025.
   (SIGReg + Epps-Pulley istatistiği)
3. Assran, M. vd. *I-JEPA / V-JEPA* ailesi — JEPA mimarisi bağlamı.

Bu paket eğitim amaçlı bir uygulamadır; yatırım tavsiyesi değildir.
