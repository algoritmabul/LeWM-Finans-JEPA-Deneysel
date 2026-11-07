# -*- coding: utf-8 -*-
"""
lewm_harman.py — kural/model harmanı laboratuvarı
==================================================
"AL listesi kurallarının (K1-K6) hangi harmanı, 250-bar doğrulamada en iyi
isabet/fark veriyor?" sorusunu sızıntısız ölçen tarama aracı.

Mantık:
 1) Model sigma matrisi (tahmin-tarihli) + kapanış/hacim matrisleri TEK
    geçişte üretilir ve <cikti>/harman_onbellek.pkl.gz dosyasına yazılır.
    (--onbellek-kullan ile model hiç açılmadan diskten yüklenir; sonraki
     taramalar GPU'suz, saniyeler içinde döner.)
 2) gz kaydırması uygulanır: gz(D) = sigma(D - tgt) — D gününde yalnız
    o gün bilinen sigma okunur (sızıntı yok; lewm_al_listesi ile birebir).
 3) 10 harman varyantı üretilir; HER BİRİ aynı doğrulama koduyla
    (lewm_al_listesi.dogrulama_tablosu) son --dogrula-bar bar üzerinde
    ölçülür: medyan fark%, pozitif gün%, isabet%...
 4) Sonuçlar fark% sıralı tablo + harman_karsilastirma.csv.

Varyant aileleri:
  TABAN_*   mevcut kuralın eşik taraması (skor 4/5/6)
  K1_*      model filtresinin (gz sigma) sertleştirilmesi
  K7_*      7. koşul olarak "model sakin" bandı eklenmesi
  HACIM/UZAMA  fiyat-hacim teyitlerinin sertleştirilmesi
  REL_TOP15 günlük seçim daraltma (göreli güç ilk 15)
  REJIM_*   piyasa rejimi filtresi (ALARM günlerinde liste alma)

Kullanım:
  python lewm_harman.py                       # sigma geçişi + tüm varyantlar
  python lewm_harman.py --onbellek-kullan     # hızlı tarama (model/GPU'suz)
  python lewm_harman.py --dogrula-bar 500     # daha uzun doğrulama penceresi
  python lewm_harman.py --dogrula-ufuk 10     # 10-bar ileri ufuk
  python lewm_harman.py --cikti lewm_cikti_buyuk   # büyük modeli sınama

Dürüstlük kaydı: tüm varyantlar aynı sigma matrisini kullanır — varyantlar
arası kıyas adildir; ancak checkpoint bugünkü ağırlıklar olduğundan TÜM
varyantlar aynı ağırlık sızıntısını taşır (lewm_al_listesi notu ile aynı).
Kazanan varyant şansla öne çıkmış olabilir (çoklu-test riski): en iyi 2-3
varyantı farklı --dogrula-bar ve --dogrula-ufuk değerlerinde yeniden sınayın;
pencere/ufuk değişince sınıfta kalanı benimsemeyin.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lewm_al_listesi import (ESIK_GZ_MUTLAK, ESIK_GZ_SIGMA, ESIK_HACIM,
                             ESIK_R5_UST, _rejim, dogrulama_tablosu,
                             gz_kaydir, veri_matrisleri)
from lewm_veri import hisseleri_bul

ONBELEK_ADI = "harman_onbellek.pkl.gz"
RAPOR_ADI = "harman_karsilastirma.csv"


# --------------------------------------------------------------------------
def kosullar_flex(sigma_gz: pd.DataFrame, kapanis: pd.DataFrame,
                  hacim: pd.DataFrame, min_fiyat: float = 1.0,
                  gz_ust: float | None = None, gz_mutlak: float | None = None,
                  r5_ust: float | None = None,
                  hacim_min: float | None = None) -> tuple:
    """lewm_al_listesi.kosul_matrisleri'nin eşikleri parametrik sürümü.

    Varsayılan eşikler orijinalle BİREBİR (K1..K6 aynı); yalnızca belirtilen
    eşik değişir. Dönüş: (kosullar, skor, rel20, hacim_oran, r5, r20).
    """
    gz_ust = ESIK_GZ_SIGMA if gz_ust is None else gz_ust
    gz_mutlak = ESIK_GZ_MUTLAK if gz_mutlak is None else gz_mutlak
    r5_ust = ESIK_R5_UST if r5_ust is None else r5_ust
    hacim_min = ESIK_HACIM if hacim_min is None else hacim_min

    r5 = kapanis / kapanis.shift(5) - 1
    r20 = kapanis / kapanis.shift(20) - 1
    ma20 = kapanis.rolling(20, min_periods=20).mean()
    ma50 = kapanis.rolling(50, min_periods=50).mean()
    hacim_oran = (hacim.rolling(5, min_periods=5).mean()
                  / (hacim.rolling(60, min_periods=30).mean() + 1e-12))
    med_r20 = r20.median(axis=1)                      # kesitsel piyasa medyanı
    rel20 = r20.sub(med_r20, axis=0)                  # göreli güç

    kosullar = {
        "K1_model": sigma_gz.notna() & sigma_gz.le(gz_ust) \
                    & sigma_gz.abs().le(gz_mutlak),
        "K2_ma20": kapanis > ma20,
        "K3_trend": ma20 > ma50,
        "K4_goreli": rel20 > 0,
        "K5_teyit": (r5 > 0) & (r5 <= r5_ust),
        "K6_hacim": hacim_oran >= hacim_min,
    }
    kosullar = {k: v.fillna(False).astype(bool) for k, v in kosullar.items()}
    skor = sum(v.astype(int) for v in kosullar.values())
    ehliyet = sigma_gz.notna() & (kapanis > min_fiyat)
    skor = skor.where(ehliyet, other=0)
    return kosullar, skor, rel20, hacim_oran, r5, r20


def harman_varyantlari(sigma_gz: pd.DataFrame, kapanis: pd.DataFrame,
                       hacim: pd.DataFrame, min_fiyat: float,
                       gz_rejim: pd.Series) -> list:
    """(ad, açıklama, liste_mask) üçlüleri — hepsi yalnız D gününde bilinen
    bilgilerle üretilir; maskeler aynı tarih/hisse ekseninde bool DataFrame."""
    V = []

    # --- A) TABAN ailesi: mevcut kural, eşik taraması ---------------------
    _, skor, rel20, _, _, _ = kosullar_flex(sigma_gz, kapanis, hacim, min_fiyat)
    V.append(("TABAN_skor5", "mevcut kural: K1-K6, skor>=5 (al_listesi ile aynı)",
              skor >= 5))
    V.append(("TABAN_skor4", "geniş: aynı kurallar, skor>=4 (İZLE dahil)",
              skor >= 4))
    V.append(("TABAN_skor6", "seçici: aynı kurallar, skor>=6 (yalnız GÜÇLÜ AL)",
              skor >= 6))

    # --- B) K1 varyantları: model filtresi sertleştirme --------------------
    _, skor_sifir, _, _, _, _ = kosullar_flex(sigma_gz, kapanis, hacim,
                                              min_fiyat, gz_ust=0.0)
    V.append(("K1_sifir",
              "K1 sert: gz sigma<=0 (pozitif sigma'lılar elenir), skor>=5",
              skor_sifir >= 5))
    _, skor_sakin, _, _, _, _ = kosullar_flex(sigma_gz, kapanis, hacim,
                                              min_fiyat, gz_mutlak=1.0)
    V.append(("K1_sakin",
              "K1 sakin: |gz sigma|<=1 (model hiç şaşırmadı), skor>=5",
              skor_sakin >= 5))

    # --- C) K7: 7. koşul olarak "model sakin" bandı ------------------------
    _, skor_taban, _, _, _, _ = kosullar_flex(sigma_gz, kapanis, hacim,
                                              min_fiyat)
    k7 = sigma_gz.notna() & sigma_gz.abs().le(1.0)
    skor7 = skor_taban + k7.astype(int)
    skor7 = skor7.where(sigma_gz.notna() & (kapanis > min_fiyat), other=0)
    V.append(("K7_sakin_ek",
              "7 koşul: K1-K6 + K7(|gz sigma|<=1), giris skoru>=6",
              skor7 >= 6))

    # --- D) fiyat/hacim teyit varyantları ----------------------------------
    _, skor_h15, _, _, _, _ = kosullar_flex(sigma_gz, kapanis, hacim,
                                            min_fiyat, hacim_min=1.5)
    V.append(("HACIM_15", "K6 sert: hacim teyidi >= 1.5x, skor>=5",
              skor_h15 >= 5))
    _, skor_u10, _, _, _, _ = kosullar_flex(sigma_gz, kapanis, hacim,
                                            min_fiyat, r5_ust=0.10)
    V.append(("UZAMA_10", "K5 sert: 5-bar getiri <= %10, skor>=5",
              skor_u10 >= 5))

    # --- E) seçim daraltma: günlük göreli güç ilk 15 ------------------------
    mask_top = (skor >= 5) & (rel20.rank(axis=1, ascending=False) <= 15)
    V.append(("REL_TOP15", "skor>=5 içinden günlük rel20 ilk 15", mask_top))

    # --- F) rejim filtresi: ALARM günlerinde liste alma ---------------------
    izin = gz_rejim.reindex(skor.index).ne("ALARM").to_numpy()
    izin_df = pd.DataFrame(
        np.broadcast_to(izin[:, None], skor.shape).copy(),
        index=skor.index, columns=skor.columns)
    V.append(("REJIM_ALARMSIZ", "skor>=5 ama yalnız ALARM olmayan günlerde",
              (skor >= 5) & izin_df))

    return V


def degerlendir(varyantlar: list, kapanis: pd.DataFrame, ufuk: int,
                gz_rejim: pd.Series, bar: int) -> pd.DataFrame:
    """Her varyantı aynı doğrulama koduyla ölçer; özet tablo döner."""
    satirlar = []
    for ad, aciklama, mask in varyantlar:
        t = dogrulama_tablosu(mask, kapanis, ufuk, gz_rejim).tail(bar)
        g = t.dropna(subset=["fark%"])
        if len(g) == 0:
            satirlar.append({"varyant": ad, "aciklama": aciklama, "n_gun": 0})
            continue
        satirlar.append({
            "varyant": ad, "aciklama": aciklama,
            "n_gun": len(g),
            "boyut_ort": round(float(g["liste_boyutu"].mean()), 1),
            "bos_gun_%": round(float((g["liste_boyutu"] == 0).mean() * 100), 0),
            "medyan_fark_%": round(float(g["fark%"].median()), 2),
            "pozitif_gun_%": round(float((g["fark%"] > 0).mean() * 100), 0),
            "medyan_isabet_%": round(float(g["isabet%"].median()), 0),
            "ort_isabet_%": round(float(g["isabet%"].mean()), 0),
            "hisse_gun": int(g["liste_boyutu"].sum()),
        })
    return pd.DataFrame(satirlar)


def _tablo_yaz(tablo: pd.DataFrame):
    """fark% sıralı karşılaştırma tablosu (TABAN_skor5 başta, işaretli)."""
    sirali = tablo.sort_values("medyan_fark_%", ascending=False,
                               na_position="last").reset_index(drop=True)
    taban_i = sirali.index[sirali["varyant"] == "TABAN_skor5"]
    taban = sirali.loc[taban_i] if len(taban_i) else sirali.iloc[0:0]
    diger = sirali[sirali["varyant"] != "TABAN_skor5"]
    yazilacak = pd.concat([taban, diger], ignore_index=True)

    baslik = (f"{'varyant':<16}{'n gün':>6}{'boyut':>7}{'boş%':>5}"
              f"{'fark%':>8}{'poz.gün%':>9}{'isabet%':>8}{'ort.is':>7}"
              f"{'hisse-gün':>10}")
    print("   " + baslik)
    print("   " + "-" * len(baslik))
    for _, r in yazilacak.iterrows():
        isaret = " <=TABAN" if r["varyant"] == "TABAN_skor5" else ""
        if r.get("n_gun", 0) == 0:
            print(f"   {r['varyant']:<16}  ölçülebilir gün yok{isaret}")
            continue
        print(f"   {r['varyant']:<16}{r['n_gun']:>6.0f}{r['boyut_ort']:>7.1f}"
              f"{r['bos_gun_%']:>5.0f}{r['medyan_fark_%']:>+8.2f}"
              f"{r['pozitif_gun_%']:>9.0f}{r['medyan_isabet_%']:>8.0f}"
              f"{r['ort_isabet_%']:>7.0f}{r['hisse_gun']:>10,}{isaret}")
    return yazilacak


# --------------------------------------------------------------------------
def ana():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    p = argparse.ArgumentParser(
        description="LeWM kural/model harmanı — 10 varyantın sızıntısız "
                    "250-bar kıyası")
    p.add_argument("--gunluk", default=r"D:\new\yeniden\metastock-gun-csv",
                   help="günlük CSV dizini (yalnız model koşusunda kullanılır)")
    p.add_argument("--cikti", default="lewm_cikti", help="çıktı dizini "
                   "(model + harman_onbellek.pkl.gz + harman_karsilastirma.csv)")
    p.add_argument("--model", default=None,
                   help="checkpoint (varsayılan: <cikti>/lewm_model_eniyi.pt)")
    p.add_argument("--min-fiyat", type=float, default=1.0,
                   help="asgari kapanış (TL) — kuruşluk kâğıt filtresi")
    p.add_argument("--dogrula-bar", type=int, default=250,
                   help="doğrulamada geriye kaç bar")
    p.add_argument("--dogrula-ufuk", type=int, default=5,
                   help="doğrulama ileri ufku (bar)")
    p.add_argument("--onbellek-kullan", action="store_true",
                   help="modeli AÇMA; sigma/kapanış/hacim matrislerini "
                        "<cikti>/harman_onbellek.pkl.gz dosyasından oku "
                        "(hızlı tarama, GPU gerekmez)")
    a = p.parse_args()

    if a.dogrula_ufuk < 1 or a.dogrula_bar < 10:
        sys.exit("[HATA] --dogrula-ufuk >= 1 ve --dogrula-bar >= 10 olmalı.")
    cikti = Path(a.cikti)
    cikti.mkdir(parents=True, exist_ok=True)
    onbellek_yolu = cikti / ONBELEK_ADI

    print("=" * 66)
    print("LeWM-Finans KURAL/MODEL HARMANI  (10 varyant, sızıntısız kıyas)")
    print("=" * 66)

    # 1) veri: model koşusu YA DA önbellek ---------------------------------
    if a.onbellek_kullan:
        if not onbellek_yolu.exists():
            sys.exit(f"[HATA] önbellek yok: {onbellek_yolu}\n"
                     "       Önce çalıştırın: python lewm_harman.py "
                     "(--onbellek-kullan'sız)")
        paket = pd.read_pickle(onbellek_yolu)
        sigma, kapanis, hacim = paket["sigma"], paket["kapanis"], paket["hacim"]
        tgt = int(paket["tgt"])
        print(f"[1/2] ÖNBELLEK yüklendi: {sigma.shape[1]} hisse, "
              f"{len(kapanis)} bar ({kapanis.index[0].date()} .. "
              f"{kapanis.index[-1].date()}), tgt={tgt} "
              f"(kaydın kaynağı: {paket.get('model', '?')}, "
              f"{paket.get('kayit', '?')[:19]})")
        print("      model açılmadı — bu koşu GPU'suz.")
    else:
        import torch
        model_yolu = Path(a.model) if a.model else cikti / "lewm_model_eniyi.pt"
        if not model_yolu.exists():
            sys.exit(f"[HATA] model bulunamadı: {model_yolu}\n"
                     "       Önce çalıştırın: python lewm_cli.py egit "
                     "(YA DA --onbellek-kullan ile mevcut önbelleği okuyun)")
        aygit = "cuda" if torch.cuda.is_available() else "cpu"
        from lewm_uygula import _modeli_yukle
        from veri_periyot import VeriPeriyot
        model, cfg_d = _modeli_yukle(model_yolu, aygit)
        ctx, tgt = int(cfg_d["ctx_uzunluk"]), int(cfg_d["tgt_uzunluk"])
        periyot = cfg_d["periyot"]
        if periyot not in ("gunluk", "haftalik", "aylik"):
            sys.exit(f"[HATA] yalnızca günlük kaynak (periyot: {periyot})")
        kaynak = Path(a.gunluk)
        if not kaynak.exists():
            sys.exit(f"[HATA] günlük CSV dizini yok: {kaynak}")
        vp = VeriPeriyot(csv_gunluk_dizini=str(kaynak),
                         csv_60dk_dizini=str(kaynak))
        hedefler = hisseleri_bul(kaynak)
        print(f"[1/2] MODEL koşusu: {len(hedefler)} hissede surprise geçişi "
              f"(model: {model_yolu.name}, aygıt: {aygit}) — tek seferlik, "
              f"sonuç önbelleğe yazılacak...")
        sigma, kapanis, hacim, n_ok = veri_matrisleri(
            model, vp, hedefler, periyot, ctx, tgt, aygit)
        pd.to_pickle({"sigma": sigma, "kapanis": kapanis, "hacim": hacim,
                      "tgt": tgt, "model": model_yolu.name,
                      "kayit": pd.Timestamp.now().isoformat(),
                      "n_hisse": int(n_ok)}, onbellek_yolu)
        print(f"      {n_ok} hisse skorlandı; önbellek: {onbellek_yolu.name}")

    # 2) gz görünümü + rejim serisi ----------------------------------------
    birlesik = kapanis.index
    sigma_gz = gz_kaydir(sigma, birlesik, tgt)
    alarm_oran_gz = (sigma_gz.abs() > 3).sum(axis=1) / \
        sigma_gz.notna().sum(axis=1).replace(0, np.nan) * 100
    medyan5 = (kapanis / kapanis.shift(5) - 1).median(axis=1) * 100
    gz_rejim = pd.Series([_rejim(o, m)[0] for o, m in
                          zip(alarm_oran_gz, medyan5)], index=birlesik)

    # 3) varyantlar + değerlendirme -----------------------------------------
    varyantlar = harman_varyantlari(sigma_gz, kapanis, hacim,
                                    a.min_fiyat, gz_rejim)
    print(f"[2/2] {len(varyantlar)} varyant, son {a.dogrula_bar} bar, "
          f"ufuk {a.dogrula_ufuk} bar — doğrulama (yalnız o gün bilinenlerle):")
    print()
    tablo = degerlendir(varyantlar, kapanis, a.dogrula_ufuk, gz_rejim,
                        a.dogrula_bar)
    _tablo_yaz(tablo)

    # 4) okuma lejantı + kayıt ------------------------------------------------
    print("""
   okuma: fark% = liste medyan ileri getirisi − piyasa medyanı (pozitif = iyi);
          poz.gün% = listenin piyasa üstü olduğu günlerin oranı;
          isabet% = listedeki hisselerin pozitif kapanan oranı;
          boyut = ortalama günlük liste uzunluğu (0-2'se sonuç gürültülüdür);
          hisse-gün = toplam gözlem sayısı (küçükse güven düşük).
   UYARI: en yüksek fark% her zaman 'daha iyi kural' demek değildir —
          liste 1-2 hisseye düşen varyantlar şansla öne çıkabilir.
          boyut_ort >= 5 VE hisse_gun >= 200 alt sınırını şart koşun;
          kazananı farklı --dogrula-bar / --dogrula-ufuk ile tekrar sınayın.""")
    yol = cikti / RAPOR_ADI
    tablo.to_csv(yol, index=False, encoding="utf-8-sig")
    print(f"   notlar: (1) tüm varyantlar aynı sigma matrisini kullanır — "
          f"varyantlar arası kıyas adildir;")
    print("   (2) checkpoint bugünkü ağırlıklar — ağırlık sızıntısı TÜM "
          "varyantlarda ortak;")
    print("   (3) kapanışta hesap, ileri getiri kapanış→kapanış; sigma "
          f"gz kaydırılmış (+{tgt} bar);")
    print("   (4) bu bir sinyal araştırma aracıdır, yatırım tavsiyesi değildir.")
    print(f"[TAMAM] karşılaştırma: {yol.resolve()}")
    print(f"        önbellek : {onbellek_yolu.resolve()} "
          f"(sonraki taramalar: --onbellek-kullan)")


if __name__ == "__main__":
    ana()
