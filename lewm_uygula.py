# -*- coding: utf-8 -*-
"""
lewm_uygula.py
==============
Eğitilmiş LeWM modelinin finansal uygulamaları — makaledeki üç kullanımın
pazar verisine uyarlaması:

1) SÜRPRİZ SKORU  (makale: "surprise detects physically implausible events")
   Predictor'ın öngördüğü gelecek gömme ile gerçek gömme arasındaki latent
   MSE. Robut istatistikle sigma birimine çevrilir; |sigma| > 3 => olağan
   dışı piyasa davranışı.

2) EMBEDDING ÜRETİMİ
   Her hisse/andan latent vektör; PCA haritası, benzer grafik eşleştirme
   veya başka ML modellerine girdi olarak kullanılabilir.

3) GELECEK TAHMİNİ (latent k-NN senaryo analizi)
   Predictor gelecek pencerenin gömmesini öngörür; bu noktanın latent
   uzaydaki en yakın TARİHSEL hedef gömmelerine bakılır — o anlarda gerçekte
   ne olmuşsa, olasılıklı senaryo olarak raporlanır (medyan getiri, % yükseliş).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from lewm_model import LeWMDunyaModeli
from lewm_veri import (LeWMVeriSeti, OZELLIK_SAYISI, ozellik_hesapla,
                       hisseleri_bul, pencere_normalle)

KAYIT_MAD = 1.4826  # MAD -> sigma dönüşüm katsayısı (normal dağılım varsayımı)


def _modeli_yukle(model_dosyasi: str | Path, aygit: str = "cpu") -> tuple:
    """Checkpoint'ten (model, cfg) yükler."""
    paket = torch.load(model_dosyasi, map_location=aygit, weights_only=False)
    cfg_d = paket["yapilandirma"]
    model = LeWMDunyaModeli(
        ozellik_sayisi=OZELLIK_SAYISI,
        ctx_uzunluk=cfg_d["ctx_uzunluk"], tgt_uzunluk=cfg_d["tgt_uzunluk"],
        patch_uzunluk=cfg_d["patch_uzunluk"], gomme_boyut=cfg_d["gomme_boyut"],
        derinlik=cfg_d["derinlik"], kafa=cfg_d["kafa"], dropout=0.0,
    ).to(aygit)
    model.encoder.load_state_dict(paket["encoder_durum"])
    model.predictor.load_state_dict(paket["predictor_durum"])
    model.eval()
    return model, cfg_d


# --------------------------------------------------------------------------
# 1) SÜRPRİZ SKORU
# --------------------------------------------------------------------------
@torch.no_grad()
def surprise_skorlari(model: LeWMDunyaModeli, vp, hisse: str, periyot: str,
                      ctx_uzunluk: int, tgt_uzunluk: int,
                      sigma_pencere: int = 252,
                      aygit: str = "cpu") -> pd.DataFrame:
    """Bir hisse için tüm tarih boyunca sürpriz skoru ve sigma serisi üretir.

    Döner: DataFrame[tarih, kapanis, surprise, sigma_skoru]
    (tarih = bağlam penceresinin son barı; skor onu izleyen tgt barların
    gerçekte ne kadar 'beklenmedik' olduğudur)
    """
    df = vp.get(hisse, periyot.lower())
    if df is None:
        raise FileNotFoundError(f"{hisse} için veri bulunamadı.")
    oz = ozellik_hesapla(df)
    arr = oz.to_numpy(dtype=np.float32)
    tarih = oz.index
    kapanis = df["Close"].reindex(tarih).to_numpy(dtype=float)
    L = len(arr)
    if L < ctx_uzunluk + tgt_uzunluk:
        raise ValueError(f"{hisse}: pencere için yetersiz bar ({L}).")

    # tüm başlangıçlar için pencere tensörleri (vektörize, hızlı)
    n = L - ctx_uzunluk - tgt_uzunluk + 1
    idx = np.arange(n)[:, None] + np.arange(ctx_uzunluk + tgt_uzunluk)[None, :]
    pencereler = arr[idx]                                  # [n, ctx+tgt, F]
    ctx = pencereler[:, :ctx_uzunluk]                      # [n, ctx, F]
    tgt = pencereler[:, ctx_uzunluk:]                      # [n, tgt, F]
    ctx_n = pencere_normalle(ctx, ctx)
    tgt_n = pencere_normalle(tgt, ctx)

    ctx_t = torch.from_numpy(ctx_n.astype(np.float32)).to(aygit)
    tgt_t = torch.from_numpy(tgt_n.astype(np.float32)).to(aygit)

    skorlar = np.empty(n, dtype=np.float64)
    B = 512
    for i in range(0, n, B):
        z_ctx = model.encoder.kodla(ctx_t[i:i + B])
        z_hat = model.predictor(z_ctx)
        z_tgt = model.encoder.kodla(tgt_t[i:i + B])
        skorlar[i:i + B] = (z_hat - z_tgt).square().mean(dim=1).cpu().numpy()

    cikis = pd.DataFrame({
        "tarih": tarih[ctx_uzunluk - 1: ctx_uzunluk - 1 + n],
        "kapanis": kapanis[ctx_uzunluk - 1: ctx_uzunluk - 1 + n],
        "surprise": skorlar,
    })

    # Robut sigma: kayan medyan/MAD (geleceğe sızma yok — shift(1)).
    # Sürpriz kare-mesafedir -> sağa çarpık; LOG alarak simetrikleştiririz,
    # böylece 3σ eşiği Gauss kalibrasyonuna (~%0.27) oturur.
    s_log = np.log(cikis["surprise"] + 1e-12)
    kayan_med = s_log.rolling(sigma_pencere, min_periods=60).median().shift(1)
    kayan_mad = s_log.rolling(sigma_pencere, min_periods=60).apply(
        lambda a: np.median(np.abs(a - np.median(a))), raw=True).shift(1)
    cikis["sigma_skoru"] = (s_log - kayan_med) / (KAYIT_MAD * kayan_mad + 1e-12)
    return cikis


# --------------------------------------------------------------------------
# 2) EMBEDDING ÜRETİMİ
# --------------------------------------------------------------------------
@torch.no_grad()
def gommeleri_cikar(model: LeWMDunyaModeli, vp, periyot: str,
                    hisseler: Optional[List[str]] = None,
                    ctx_uzunluk: int = 64,
                    hisse_basina_pencere: Optional[int] = None,
                    aygit: str = "cpu") -> pd.DataFrame:
    """Hisselerin latent gömmelerini üretir.

    hisse_basina_pencere=None  -> yalnızca SON pencere (tarama tablosu)
    hisse_basina_pencere=k     -> hisse başına son k pencere (harita/k-NN için)
    """
    if hisseler is None:
        dizin = (vp.csv_gunluk_dizini if periyot.lower() in ("gunluk", "haftalik", "aylik")
                 else vp.csv_60dk_dizini)
        hisseler = hisseleri_bul(dizin)

    satirlar = []
    for hisse in hisseler:
        try:
            df = vp.get(hisse, periyot.lower())
        except Exception:
            continue
        if df is None:
            continue
        oz = ozellik_hesapla(df)
        arr = oz.to_numpy(dtype=np.float32)
        tarih = oz.index
        L = len(arr)
        if L < ctx_uzunluk:
            continue
        sonlar = ([L - ctx_uzunluk] if hisse_basina_pencere is None
                  else list(range(max(0, L - ctx_uzunluk - hisse_basina_pencere + 1),
                                  L - ctx_uzunluk + 1)))
        pencere = np.stack([arr[s:s + ctx_uzunluk] for s in sonlar])
        pencere_n = pencere_normalle(pencere, pencere)

        z_listesi = []
        B = 512
        p_t = torch.from_numpy(pencere_n.astype(np.float32)).to(aygit)
        for i in range(0, len(sonlar), B):
            z_listesi.append(model.encoder.kodla(p_t[i:i + B]).cpu().numpy())
        z = np.concatenate(z_listesi)

        D = z.shape[1]
        for j, s in enumerate(sonlar):
            satir = {"hisse": hisse, "tarih": tarih[s + ctx_uzunluk - 1]}
            satir.update({f"z{k:03d}": float(z[j, k]) for k in range(D)})
            satirlar.append(satir)

    return pd.DataFrame(satirlar)


# --------------------------------------------------------------------------
# 3) GELECEK TAHMİNİ — latent k-NN senaryo analizi
# --------------------------------------------------------------------------
@torch.no_grad()
def _referans_havuzu(model: LeWMDunyaModeli, vp, periyot: str,
                     ctx_uzunluk: int, tgt_uzunluk: int,
                     referans_hisseler: Optional[List[str]] = None,
                     referans_pencere: int = 500, aygit: str = "cpu",
                     bildir=None) -> tuple:
    """k-NN referans havuzu: tüm (veya seçili) hisselerin tarihsel hedef
    gömmeleri + gerçekleşen getirileri.

    Havuz kurmak pahalıdır (veri kümesinin büyük bölümü gömülür). gelecek_tahmini
    her çağrıda havuzu yeniden kurar; senaryo_tarama ise havuzu BİR KEZ kurup
    tüm hedeflerde paylaştığından çok-hisseli tarama ~N katı değil ~1 katı
    maliyetle döner.
    Döner: (z_ref [M,D] float32, getiri_ref [M] float64, meta_ref [M] list).
    """
    veri = LeWMVeriSeti(vp, periyot, referans_hisseler, ctx_uzunluk, tgt_uzunluk,
                        bolum="egitim", sustur=True)
    # son `referans_pencere` örneği al (hisse dağılımını korumak için sondan)
    if referans_pencere and len(veri) > referans_pencere * max(1, len(veri.hisse_adlari)):
        adim = len(veri) // (referans_pencere * max(1, len(veri.hisse_adlari)))
        ornekler = list(range(0, len(veri), adim))[-referans_pencere * max(1, len(veri.hisse_adlari)):]
    else:
        ornekler = list(range(len(veri)))

    z_ref_listesi, getiri_ref, meta_ref = [], [], []
    B = 512
    for i0 in range(0, len(ornekler), B):
        parti = ornekler[i0:i0 + B]
        ctx_b = torch.stack([veri[i][0] for i in parti]).to(aygit)
        tgt_b = torch.stack([veri[i][1] for i in parti]).to(aygit)
        z_tgt = model.encoder.kodla(tgt_b).cpu().numpy()
        z_ref_listesi.append(z_tgt.astype(np.float16))
        getiri_ref.extend(veri.stok_getirisi(veri.ornek_indeksi[i][0], veri.ornek_indeksi[i][1])
                          for i in parti)
        meta_ref.extend(veri.meta(i) for i in parti)
        if bildir:
            bildir(min(i0 + len(parti), len(ornekler)), len(ornekler))
    z_ref = np.concatenate(z_ref_listesi).astype(np.float32)     # [M, D]
    getiri_ref = np.asarray(getiri_ref, dtype=np.float64)
    return z_ref, getiri_ref, meta_ref


@torch.no_grad()
def _senaryo_cekirdek(model: LeWMDunyaModeli, vp, hisse: str, periyot: str,
                      ctx_uzunluk: int, tgt_uzunluk: int,
                      z_ref: np.ndarray, getiri_ref: np.ndarray,
                      meta_ref: list, komsu: int = 25,
                      aygit: str = "cpu") -> dict:
    """Havuz verildiğinde tek hissenin k-NN senaryo özeti (gelecek_tahmini'nin
    b-c adımları). Havuz paylaşıldığından sonuçlar gelecek_tahmini ile birebir
    aynı çıkar."""
    # b) hedef hissenin son penceresi -> tahmin
    df = vp.get(hisse, periyot.lower())
    oz = ozellik_hesapla(df)
    arr = oz.to_numpy(dtype=np.float32)
    tarih = oz.index
    if len(arr) < ctx_uzunluk:
        raise ValueError(f"{hisse}: yetersiz bar.")
    ctx = arr[-ctx_uzunluk:]
    ctx_n = pencere_normalle(ctx, ctx)
    z_ctx = model.encoder.kodla(torch.from_numpy(ctx_n[None].astype(np.float32)).to(aygit))
    z_hat = model.predictor(z_ctx).cpu().numpy()[0]

    # c) k-NN (kosinüs benzerliği)
    zr = z_ref / (np.linalg.norm(z_ref, axis=1, keepdims=True) + 1e-8)
    zh = z_hat / (np.linalg.norm(z_hat) + 1e-8)
    benzerlik = zr @ zh
    sirali = np.argsort(-benzerlik)[:komsu]

    secilen = [meta_ref[i] for i in sirali]
    secilen_getiri = getiri_ref[sirali]
    secilen_benzer = benzerlik[sirali]

    tablo = pd.DataFrame({
        "hisse": [m["hisse"] for m in secilen],
        "hedef_basi": [m["hedef_basi"] for m in secilen],
        "hedef_sonu": [m["hedef_sonu"] for m in secilen],
        "benzerlik": np.round(secilen_benzer, 4),
        "gerceklesen_getiri_%": np.round(secilen_getiri * 100, 2),
    }).sort_values("benzerlik", ascending=False).reset_index(drop=True)

    return {
        "hisse": hisse.upper(),
        "baglam_sonu": str(tarih[-1].date()),
        "tahmin_edilen_gomme": z_hat,
        "medyan_senaryo_getiri_%": float(np.median(secilen_getiri) * 100),
        "ortalama_senaryo_getiri_%": float(secilen_getiri.mean() * 100),
        "yukselen_orani_%": float((secilen_getiri > 0).mean() * 100),
        "ceyrekler_%": [float(q * 100) for q in np.percentile(secilen_getiri, [10, 25, 50, 75, 90])],
        "senaryolar": tablo,
    }


@torch.no_grad()
def gelecek_tahmini(model: LeWMDunyaModeli, vp, hisse: str, periyot: str,
                    ctx_uzunluk: int, tgt_uzunluk: int,
                    referans_hisseler: Optional[List[str]] = None,
                    referans_pencere: int = 500, komsu: int = 25,
                    aygit: str = "cpu") -> dict:
    """Son bağlam penceresinden gelecek gömmeyi tahmin eder ve latent uzayda
    en yakın tarihsel senaryoları toplar."""
    z_ref, getiri_ref, meta_ref = _referans_havuzu(
        model, vp, periyot, ctx_uzunluk, tgt_uzunluk,
        referans_hisseler=referans_hisseler, referans_pencere=referans_pencere,
        aygit=aygit)
    return _senaryo_cekirdek(model, vp, hisse, periyot, ctx_uzunluk, tgt_uzunluk,
                             z_ref, getiri_ref, meta_ref, komsu=komsu, aygit=aygit)


@torch.no_grad()
def senaryo_tarama(model: LeWMDunyaModeli, vp, periyot: str,
                   ctx_uzunluk: int, tgt_uzunluk: int, hedefler: List[str],
                   referans_hisseler: Optional[List[str]] = None,
                   referans_pencere: int = 500, komsu: int = 25,
                   aygit: str = "cpu") -> tuple:
    """TÜM hedef hisseler için TEK referans havuzuyla k-NN senaryo taraması.

    gelecek_tahmini'yi hisse başına çağırmak havuzu N kez kurardı (N×maliyet);
    burada havuz bir kez kurulur, hisse başına yalnız son pencere gömülür.
    Döner: (tablo DataFrame — medyan senaryo getirisine göre azalan sıralı,
    atlanan [(hisse, hata)] listesi)."""
    z_ref, getiri_ref, meta_ref = _referans_havuzu(
        model, vp, periyot, ctx_uzunluk, tgt_uzunluk,
        referans_hisseler=referans_hisseler, referans_pencere=referans_pencere,
        aygit=aygit)
    kolonlar = ["hisse", "baglam_sonu", "medyan_senaryo_getiri_%",
                "ortalama_senaryo_getiri_%", "p10_%", "p25_%", "p75_%",
                "p90_%", "yukselen_orani_%", "k"]
    satirlar, atlanan = [], []
    for i, h in enumerate(hedefler, 1):
        try:
            oz = _senaryo_cekirdek(model, vp, h, periyot, ctx_uzunluk,
                                   tgt_uzunluk, z_ref, getiri_ref, meta_ref,
                                   komsu=komsu, aygit=aygit)
        except Exception as e:
            atlanan.append((h, str(e)))
            continue
        c = oz["ceyrekler_%"]
        satirlar.append({
            "hisse": oz["hisse"], "baglam_sonu": oz["baglam_sonu"],
            "medyan_senaryo_getiri_%": round(oz["medyan_senaryo_getiri_%"], 2),
            "ortalama_senaryo_getiri_%": round(oz["ortalama_senaryo_getiri_%"], 2),
            "p10_%": round(c[0], 2), "p25_%": round(c[1], 2),
            "p75_%": round(c[3], 2), "p90_%": round(c[4], 2),
            "yukselen_orani_%": round(oz["yukselen_orani_%"], 1),
            "k": komsu,
        })
        if i % 25 == 0 or i == len(hedefler):
            print(f"   senaryo taraması: {i}/{len(hedefler)}")
    tablo = (pd.DataFrame(satirlar)
             .sort_values("medyan_senaryo_getiri_%", ascending=False)
             .reset_index(drop=True)) if satirlar else pd.DataFrame(columns=kolonlar)
    return tablo, atlanan


# --------------------------------------------------------------------------
# 4) YÖN PROBE'U — embedding kalite kontrolü ( Linear probe + ham baseline )
# --------------------------------------------------------------------------
def _lojistik_egit(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray,
                   y_va: np.ndarray, iterasyon: int = 300) -> float:
    """Basit lojistik regresyon (tam gradyan, Adam, sabit seed) — doğrulama
    doğruluğu döner. Hem gömme probe'u hem ham baseline İKİNSİ bu eğiticiyi
    kullanır; böylece karşılaştırma adil olur (aynı model, aynı seed)."""
    torch.manual_seed(0)
    W = torch.zeros(X_tr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    opt = torch.optim.Adam([W, b], lr=0.05)
    for _ in range(iterasyon):
        logit = Xt @ W + b
        kayip = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt)
        opt.zero_grad(); kayip.backward(); opt.step()

    with torch.no_grad():
        p_va = torch.sigmoid(torch.tensor(X_va, dtype=torch.float32) @ W + b).numpy()
    return float(((p_va > 0.5) == (y_va > 0.5)).mean())


def _ham_oznitelikler(parti: torch.Tensor) -> np.ndarray:
    """Pencere tensöründen [B, ctx, 6] ham (gömme DIŞI) öznitelikler.
    Sütunlar OZELLIK_ADLARI ile uyumlu: 0=log_getiri, 5=log_hacim.
    Yedi öznitelik — hepsi pencere İÇİNDEN, gelecek sızıntısı yok:
      1-3) çok ufuklu momentum : son 8 / 16 / 32 bar log-getiri toplamı
      4-5) kısa vadeli volatilite : son 8 / 16 bar getiri std'sı
      6)   hacim rejimi : son 8 bar ortalama log-hacim - önceki barların ortalaması
      7)   yol konumu : kümülatif fiyat yolunun 64-bar min-max aralığındaki yeri
    Bu, "model sadece trend tekrarını mı öğrendi?" sorusunun kontrol deneyidir:
    basit istatistikler aynı probe'la eğitilip gömmeyle kıyaslanır."""
    x = parti.detach().cpu().numpy()
    r = x[:, :, 0]
    v = x[:, :, 5]
    n_ctx = r.shape[1]
    k8, k16, k32 = min(8, n_ctx), min(16, n_ctx), min(32, n_ctx)
    yol = np.cumsum(r, axis=1)
    ymin = yol.min(axis=1)
    ymax = yol.max(axis=1)

    def _std(a: np.ndarray) -> np.ndarray:
        return a.std(axis=1) if a.shape[1] > 1 else np.zeros(a.shape[0])

    return np.column_stack([
        r[:, -k8:].sum(1),                              # son 8 bar getiri
        r[:, -k16:].sum(1),                             # son 16 bar getiri
        r[:, -k32:].sum(1),                             # son 32 bar getiri
        _std(r[:, -k8:]),                               # son 8 bar volatilite
        _std(r[:, -k16:]),                              # son 16 bar volatilite
        v[:, -k8:].mean(1) - v[:, :-k8].mean(1),        # hacim rejimi değişimi
        (yol[:, -1] - ymin) / (ymax - ymin + 1e-8),     # aralık içi konum
    ]).astype(np.float64)


def yon_probe(model: LeWMDunyaModeli, vp, cfg_dict: dict,
              pencere_siniri: int = 400, aygit: str = "cpu") -> dict:
    """Dondurulmuş gömmelerden 8-bar yön sınıflandırması (lojistik regresyon)
    + HAM ÖZNİTELİK KARŞILAŞTIRMASI (kontrol deneyi).

    Aynı pencerelerden iki girdi seti çıkar:
      A) embedding        : encoder gömmesi (modelin öğrendiği temsil)
      B) ham öznitelikler : getiri+volatilite+hacim basit istatistikleri
    İkisi de AYNI lojistik probe'la eğitilip AYNI doğrulama setinde ölçülür.
    A, B'yi geçemiyorsa model basit trend tekrarından öteye bilgi eklemiyor
    demektir — bu, embedding kalitesinin dürüst bir göstergesidir."""
    ctx, tgt = cfg_dict["ctx_uzunluk"], cfg_dict["tgt_uzunluk"]
    egitim = LeWMVeriSeti(vp, cfg_dict["periyot"], cfg_dict["hisseler"], ctx, tgt,
                          bolum="egitim", sustur=True)
    dogrulama = LeWMVeriSeti(vp, cfg_dict["periyot"], cfg_dict["hisseler"], ctx, tgt,
                             bolum="dogrulama", sustur=True)

    def _hazirla(veri: LeWMVeriSeti) -> tuple:
        n = min(len(veri), pencere_siniri)
        ornekler = np.linspace(0, len(veri) - 1, n).astype(int)
        z, h, y = [], [], []
        with torch.no_grad():      # embedding üretimi gradyansız, probe eğitimi değil
            for i0 in range(0, n, 512):
                parti = ornekler[i0:i0 + 512]
                xb = torch.stack([veri[i][0] for i in parti]).to(aygit)
                z.append(model.encoder.kodla(xb).cpu().numpy())
                h.append(_ham_oznitelikler(xb))
                y.extend([1.0 if veri.stok_getirisi(veri.ornek_indeksi[i][0],
                                                    veri.ornek_indeksi[i][1]) > 0 else 0.0
                          for i in parti])
        return (np.concatenate(z), np.concatenate(h), np.asarray(y))

    Z_tr, H_tr, y_tr = _hazirla(egitim)
    Z_va, H_va, y_va = _hazirla(dogrulama)

    def _standartla(X_tr_, X_va_):
        mu, sd = X_tr_.mean(0), np.maximum(X_tr_.std(0), 1e-6)
        return (X_tr_ - mu) / sd, (X_va_ - mu) / sd

    Z_tr, Z_va = _standartla(Z_tr, Z_va)
    H_tr, H_va = _standartla(H_tr, H_va)

    emb_dogruluk = _lojistik_egit(Z_tr, y_tr, Z_va, y_va)
    ham_dogruluk = _lojistik_egit(H_tr, y_tr, H_va, y_va)
    taban = float(max(y_va.mean(), 1 - y_va.mean()))
    return {"dogruluk": emb_dogruluk, "ham_dogruluk": ham_dogruluk,
            "taban_dogruluk": taban,
            "n_egitim": len(y_tr), "n_dogrulama": len(y_va)}
