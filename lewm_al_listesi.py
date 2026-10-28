# -*- coding: utf-8 -*-
"""
lewm_al_listesi.py — "AL" durumdaki hisseler (sızıntısız, doğrulamalı)
======================================================================
"Gerçekten AL" tanımı kanıta dayalıdır ve ÜÇ katmandan oluşur:

  1) MODEL KATMANI (K1) — risk filtresi. Geriye dönük test (lewm_geriye_donuk)
     yüksek pozitif sigma kuyruğunun piyasa ALTINDA getiri verdiğini gösterdi
     (TOP-20 −13.96% vs piyasa −10.72%). Modelin bugünkü dürüst katkısı YÖN
     iddiası değil, ANOMALİ tespitidir: gz sigma ≤ +1 ve |gz sigma| ≤ 3
     olmayan hisse AL olamaz. (gz = gerçek-zamanlı: sigma satırı t, t+tgt
     kapanışında bilinir → D gününde bilinen son okuma sigma(D−tgt).)

  2) FİYAT/HACİM KATMANI (K2..K6) — klasik teyitler, hepsi D kapanışında
     bilinir: Close>MA20, MA20>MA50, 20-bar göreli güç > piyasa medyanı,
     0 < 5-bar getiri ≤ +15% (aşırı uzama değil), hacim teyidi (5-bar ort /
     60-bar ort ≥ 1).

  3) PİYASA KATMANI — piyasa radarıyla aynı eşiklerle GZ rejim okunur.
     ALARM rejiminde AL listesi risklidir; --piyasa-onay ile sert filtre.

SKOR = geçilen koşul sayısı (0-6):
  6 → GÜÇLÜ AL   5 → AL   4 → İZLE   (eşik --esik-skor, varsayılan 5)

DOĞRULAMA: aynı kural geçmiş her günde yalnız o gün bilinenlerle
uygulanır; listenin ileri-ufuk getirisi piyasa medyanıyla kıyaslanır
(al_dogrulama.csv). Kural tarihsel olarak piyasa altıysa dürüstçe söyler.

Kullanım:
  python lewm_al_listesi.py
  python lewm_al_listesi.py --esik-skor 6 --top 20
  python lewm_al_listesi.py --senaryo          # + k-NN senaryo kolonu (yavaş)
  python lewm_al_listesi.py --dogrulassiz      # doğrulamayı atla
  python lewm_al_listesi.py --son-tarih 2026-10-16 --kontrol-tarih 2026-10-23
  #   → veriyi FİZİKSEL olarak son tarihe kadar kırpar (sonrasını GÖRMEZ);
  #     listeyi o günü bilerek üretir, listedekileri ORİJİNAL CSV'lerin
  #     kontrol-tarihi kapanışlarıyla ölçer (lewm_geriye_donuk deseni)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lewm_veri import hisseleri_bul
from lewm_uygula import _modeli_yukle, surprise_skorlari
from veri_periyot import VeriPeriyot

# --- AL koşul eşikleri ------------------------------------------------------
ESIK_GZ_SIGMA = 1.0     # K1: gz sigma üst sınırı (pozitif kuyruk riskliydi)
ESIK_GZ_MUTLAK = 3.0    # K1: |gz sigma| üst sınırı (±3σ = olağan dışı)
ESIK_R5_UST = 0.15      # K5: 5 barda +%15 üstü = aşırı uzama (kovalama)
ESIK_HACIM = 1.0        # K6: hacim teyidi (5-bar ort / 60-bar ort)
# --- piyasa rejim eşikleri (lewm_piyasa_radari ile aynı) --------------------
ESIK_ALARM_ORAN = 4.0
ESIK_TEDBIR_ORAN = 1.0
ESIK_DUSUS_ALARM = -7.0
ESIK_DUSUS_TEDBIR = -3.0

KOSUL_ADLARI = ["K1_model", "K2_ma20", "K3_trend", "K4_goreli",
                "K5_teyit", "K6_hacim"]


def gz_kaydir(sigma: pd.DataFrame, birlesik_indeks: pd.DatetimeIndex,
              tgt: int) -> pd.DataFrame:
    """Tahmin-tarihli sigma matrisini gerçek-zamanlı görüne kaydırır.

    Satır t'nin değeri t+tgt kapanışında bilinir → pozisyon kaydırması:
    gz(D) = sigma(D − tgt). Sızıntı yok; gelecekteki hiçbir satır kullanılmaz.
    """
    return sigma.reindex(birlesik_indeks).shift(tgt)


def kosul_matrisleri(sigma_gz: pd.DataFrame, kapanis: pd.DataFrame,
                     hacim: pd.DataFrame, min_fiyat: float = 1.0) -> tuple:
    """AL koşullarının bool matrislerini üretir (NaN → False; sızıntısız).

    Tüm girdiler D kapanışında bilinen serilerdir (gz sigma kaydırılmış,
    MA/rolling geriye bakar). Döner: (kosullar dict, skor DataFrame int).
    """
    r5 = kapanis / kapanis.shift(5) - 1
    r20 = kapanis / kapanis.shift(20) - 1
    ma20 = kapanis.rolling(20, min_periods=20).mean()
    ma50 = kapanis.rolling(50, min_periods=50).mean()
    hacim_oran = (hacim.rolling(5, min_periods=5).mean()
                  / (hacim.rolling(60, min_periods=30).mean() + 1e-12))
    med_r20 = r20.median(axis=1)                      # kesitsel piyasa medyanı
    rel20 = r20.sub(med_r20, axis=0)                  # göreli güç

    kosullar = {
        # K1 — model: anomali yok (ne pozitif kuyruk ne ±3σ dışı)
        "K1_model": sigma_gz.notna() & sigma_gz.le(ESIK_GZ_SIGMA) \
                    & sigma_gz.abs().le(ESIK_GZ_MUTLAK),
        # K2 — kısa trend yukarı
        "K2_ma20": kapanis > ma20,
        # K3 — orta trend yukarı
        "K3_trend": ma20 > ma50,
        # K4 — 20-bar göreli güç piyasa medyanının üstünde
        "K4_goreli": rel20 > 0,
        # K5 — teyit var ama aşırı uzama yok
        "K5_teyit": (r5 > 0) & (r5 <= ESIK_R5_UST),
        # K6 — hacim teyidi
        "K6_hacim": hacim_oran >= ESIK_HACIM,
    }
    kosullar = {k: v.fillna(False).astype(bool) for k, v in kosullar.items()}
    skor = sum(v.astype(int) for v in kosullar.values())
    # adaylık: sigma okuması var + en düşük likidite filtresi
    ehliyet = sigma_gz.notna() & (kapanis > min_fiyat)
    skor = skor.where(ehliyet, other=0)
    return kosullar, skor, rel20, hacim_oran, r5, r20


def kategori(skor: int) -> str:
    if skor >= 6:
        return "GÜÇLÜ AL"
    if skor == 5:
        return "AL"
    if skor == 4:
        return "İZLE"
    return ""


def _rejim(alarm_oran: float, medyan5: float) -> tuple:
    """Piyasa rejimi (lewm_piyasa_radari eşikleriyle birebir)."""
    neden, seviye = [], 0
    if np.isfinite(alarm_oran):
        if alarm_oran > ESIK_ALARM_ORAN:
            seviye = 2; neden.append(f"sigma %{alarm_oran:.1f}")
        elif alarm_oran > ESIK_TEDBIR_ORAN:
            seviye = max(seviye, 1); neden.append(f"sigma %{alarm_oran:.1f}")
    if np.isfinite(medyan5):
        if medyan5 < ESIK_DUSUS_ALARM:
            seviye = 2; neden.append(f"fiyat {medyan5:.1f}%")
        elif medyan5 < ESIK_DUSUS_TEDBIR:
            seviye = max(seviye, 1); neden.append(f"fiyat {medyan5:.1f}%")
    return ("ALARM" if seviye == 2 else "TEDBİR" if seviye == 1 else "NORMAL",
            " + ".join(neden))


def dogrulama_tablosu(liste_mask: pd.DataFrame, kapanis: pd.DataFrame,
                      ufuk: int, gz_rejim: pd.Series) -> pd.DataFrame:
    """Kuralın tarihsel defter tutanlığı (yalnız o gün bilinenlerle).

    Her D: liste = skor≥eşik (D'de bilinir) → üyelerin ileri-ufuk getirisi
    (kapanış→kapanış) medyanı vs piyasa medyanı; isabet = listede pozitif oranı.
    """
    ileri = kapanis.shift(-ufuk) / kapanis - 1
    liste_ge = ileri.where(liste_mask)
    boyut = liste_mask.sum(axis=1)
    liste_med = liste_ge.median(axis=1) * 100
    piyasa_med = ileri.median(axis=1) * 100
    isabet = ((liste_ge > 0).sum(axis=1) / boyut.replace(0, np.nan)) * 100
    tablo = pd.DataFrame({
        "liste_boyutu": boyut,
        "liste_ileri%": liste_med,
        "piyasa_ileri%": piyasa_med,
        "fark%": liste_med - piyasa_med,
        "isabet%": isabet,
        "gz_rejim": gz_rejim.reindex(liste_mask.index).fillna(""),
    })
    # veri sonunda ufuk tamamlanmamış günler ölçülemez → temizle
    son_gecerli = len(tablo) - ufuk
    tablo.iloc[max(0, son_gecerli):,
               tablo.columns.get_indexer(["liste_ileri%", "piyasa_ileri%",
                                          "fark%", "isabet%"])] = np.nan
    return tablo


def veri_matrisleri(model, vp, hedefler, periyot, ctx, tgt, aygit):
    """Tek geçiş: sigma (tahmin-tarihli) + TAM kapanış/hacim matrisleri."""
    sigma_seriler, kapanis_tam, hacim_tam = {}, {}, {}
    n_ok = 0
    for i, h in enumerate(hedefler, 1):
        try:
            sk = surprise_skorlari(model, vp, h, periyot, ctx, tgt, aygit=aygit)
            t = pd.to_datetime(sk["tarih"])
            sigma_seriler[h] = pd.Series(sk["sigma_skoru"].to_numpy(), index=t)
            df = vp.get(h, periyot)
            kapanis_tam[h] = df["Close"]
            hacim_tam[h] = df["Volume"]
            n_ok += 1
        except Exception:
            continue
        if i % 100 == 0:
            print(f"      taranan: {i}/{len(hedefler)}")
    if not sigma_seriler:
        sys.exit("[HATA] hiç hisse skorlanamadı.")
    return (pd.DataFrame(sigma_seriler), pd.DataFrame(kapanis_tam),
            pd.DataFrame(hacim_tam), n_ok)


def _csv_kirp(kaynak: Path, hedef: Path, son_tarih: pd.Timestamp) -> int:
    """Tüm günlük CSV'leri son_tarih'e kadar kırpıp hedef dizine yazar.
    Sonrası modelin erişiminin DIŞINDA kalır (girdi sızıntısı imkânsız)."""
    hedef.mkdir(parents=True, exist_ok=True)
    say = 0
    for dosya in sorted(kaynak.glob("*.csv")):
        try:
            df = pd.read_csv(dosya)
        except Exception:
            continue
        if "Date" not in df.columns:
            continue
        dt = pd.to_datetime(df["Date"], errors="coerce")
        df = df[dt.notna() & (dt <= son_tarih)]
        if df.empty:
            continue
        df.to_csv(hedef / dosya.name, index=False)
        say += 1
    return say


def _kapanis_al(dosya: Path, ref_t: pd.Timestamp, knt_t: pd.Timestamp,
                tolerans_gun: int = 5):
    """ORİJİNAL CSV'den (ref_kapanis, ref_tarih, kontrol_kapanis, kontrol_tarih).
    Kontrol barı kontrol tarihinin `tolerans_gun` gün öncesinden itibaren
    aranır (hisse o hafta işlem görmediyse None döner -> 'veri yok')."""
    try:
        df = pd.read_csv(dosya)
    except Exception:
        return None, None, None, None
    if "Date" not in df.columns:
        return None, None, None, None
    dt = pd.to_datetime(df["Date"], errors="coerce")
    df = df.assign(_dt=dt).dropna(subset=["_dt"]).sort_values("_dt")
    ref = df[df["_dt"] <= ref_t].tail(1)
    knt = df[(df["_dt"] <= knt_t)
             & (df["_dt"] >= knt_t - pd.Timedelta(days=tolerans_gun))].tail(1)
    rc = float(ref["Close"].iloc[0]) if not ref.empty else None
    rd = str(ref["_dt"].iloc[0].date()) if not ref.empty else None
    kc = float(knt["Close"].iloc[0]) if not knt.empty else None
    kd = str(knt["_dt"].iloc[0].date()) if not knt.empty else None
    return rc, rd, kc, kd


def _hindsight_degerlendir(liste: pd.DataFrame, kaynak_orijinal: Path,
                           son_t: pd.Timestamp, knt_t: pd.Timestamp,
                           cikti: Path):
    """Listedekileri ORİJİNAL CSV'lerin kontrol-tarihi kapanışlarıyla ölçer;
    piyasa (tüm hisseler) benchmark'ıyla kıyaslar."""
    print(f"\n--- DEĞERLENDİRME (orijinal kapanışlarla: "
          f"{son_t.date()} → {knt_t.date()}) ---")
    hisse_tumu = hisseleri_bul(kaynak_orijinal)
    ref_map, knt_map = {}, {}
    for i, h in enumerate(hisse_tumu, 1):
        rc, rd, kc, kd = _kapanis_al(kaynak_orijinal / f"{h}.csv", son_t, knt_t)
        ref_map[h], knt_map[h] = (rc, rd), (kc, kd)
        if i % 200 == 0:
            print(f"      okunan: {i}/{len(hisse_tumu)}")

    def _getiri(h):
        rc = ref_map.get(h, (None, None))[0]
        kc = knt_map.get(h, (None, None))[0]
        return (kc / rc - 1) * 100 if (rc and kc and rc != 0) else np.nan

    piyasa = pd.Series({h: _getiri(h) for h in hisse_tumu}).dropna()

    satirlar = []
    for _, r in liste.iterrows():
        h = r["hisse"]
        rc, rd = ref_map.get(h, (None, None))
        kc, kd = knt_map.get(h, (None, None))
        satir = dict(r)
        satir.update({"ref_kapanis": rc, "ref_tarih": rd,
                      "kontrol_kapanis": kc, "kontrol_tarih": kd,
                      "getiri_%": _getiri(h)})
        satirlar.append(satir)
    dfr = pd.DataFrame(satirlar)
    if len(dfr) == 0:
        print("   (liste boş — kural o gün AL dememiş; "
              "bu da bir karardır, piyasa özeti aşağıda)")
    else:
        print(f"   {'hisse':<12}{'kategori':>10}{'skor':>5}{'gzσ':>8}"
              f"{'ref':>10}{'kontrol':>10}{'getiri':>9}")
        for _, r in dfr.iterrows():
            if pd.isna(r["getiri_%"]):
                print(f"   {r['hisse']:<12}{str(r['kategori']):>10}{r['skor']:>5}"
                      f"{_h(r['gz_sigma'], '+.2f'):>8}   veri yok")
            else:
                print(f"   {r['hisse']:<12}{str(r['kategori']):>10}{r['skor']:>5}"
                      f"{_h(r['gz_sigma'], '+.2f'):>8}{r['ref_kapanis']:>10.2f}"
                      f"{r['kontrol_kapanis']:>10.2f}{r['getiri_%']:>+8.2f}%")

    def _ozet_satiri(g, ad):
        if len(g) == 0:
            return f"   {ad:<22} n=0"
        return (f"   {ad:<22} n={len(g):<4} medyan={g.median():+6.2f}%  "
                f"ort={g.mean():+6.2f}%  yükselen=%{(g > 0).mean() * 100:3.0f}  "
                f"[{g.min():+.1f} .. {g.max():+.1f}]")

    print(f"\n   ÖZET ({son_t.date()} kapanışı → {knt_t.date()} kapanışı):")
    print(_ozet_satiri(piyasa, "PİYASA (tümü)"))
    if len(dfr) > 0:
        g_tum = dfr["getiri_%"].dropna()
        print(_ozet_satiri(g_tum, "AL listesi (toplam)"))
        for kat in ("GÜÇLÜ AL", "AL", "İZLE"):
            g_k = dfr[dfr["kategori"] == kat]["getiri_%"].dropna()
            if len(g_k):
                print(_ozet_satiri(g_k, kat))
        if len(g_tum) and len(piyasa):
            fark = g_tum.median() - piyasa.median()
            ustun = (g_tum.values[:, None] > piyasa.values[None, :]).mean() * 100
            print(f"   liste − piyasa medyan fark: {fark:+.2f} pt | "
                  f"listedekilerin piyasayı geçme oranı: %{ustun:.0f}")
    dfr.to_csv(cikti / "al_degerlendirme.csv", index=False,
               encoding="utf-8-sig")
    return dfr


def ana():
    p = argparse.ArgumentParser(
        description="LeWM AL listesi — sızıntısız koşullar + tarihsel doğrulama")
    p.add_argument("--gunluk", default=r"D:\new\yeniden\metastock-gun-csv",
                   help="orijinal günlük CSV dizini")
    p.add_argument("--cikti", default="lewm_cikti", help="çıktı dizini")
    p.add_argument("--model", default=None,
                   help="checkpoint (varsayılan: <cikti>/lewm_model_eniyi.pt)")
    p.add_argument("--esik-skor", type=int, default=5, choices=(4, 5, 6),
                   help="listeye girmek için gereken skor (4=İZLE 5=AL 6=GÜÇLÜ AL)")
    p.add_argument("--top", type=int, default=30, help="tablo satır sınırı")
    p.add_argument("--min-fiyat", type=float, default=1.0,
                   help="asgari kapanış (TL) — kuruşluk kâğıt filtresi")
    p.add_argument("--senaryo", action="store_true",
                   help="listedekilere k-NN senaryo kolonu ekle (YAVAŞ)")
    p.add_argument("--piyasa-onay", action="store_true",
                   help="piyasa ALARM/TEDBİR ise liste ÜRETME")
    p.add_argument("--dogrula-bar", type=int, default=250,
                   help="doğrulamada geriye kaç bar")
    p.add_argument("--dogrula-ufuk", type=int, default=5,
                   help="doğrulama ileri ufku (bar)")
    p.add_argument("--dogrulassiz", action="store_true",
                   help="tarihsel doğrulamayı atla")
    p.add_argument("--son-tarih", default=None,
                   help="hindsight: veriyi BU tarihe kadar kırp (sonrasını görmez)")
    p.add_argument("--kontrol-tarih", default=None,
                   help="hindsight: listedekileri BU günün orijinal kapanışıyla ölç")
    p.add_argument("--sakla", action="store_true",
                   help="hindsight modunda kırpılmış geçici dizini silme")
    a = p.parse_args()

    kaynak_orijinal = Path(a.gunluk)
    if not kaynak_orijinal.exists():
        sys.exit(f"[HATA] günlük CSV dizini yok: {kaynak_orijinal}")
    cikti = Path(a.cikti)
    cikti.mkdir(parents=True, exist_ok=True)
    model_yolu = Path(a.model) if a.model else cikti / "lewm_model_eniyi.pt"
    if not model_yolu.exists():
        sys.exit(f"[HATA] model bulunamadı: {model_yolu}\n"
                 f"       Önce çalıştırın: python lewm_cli.py egit")

    kaynak, son_t, knt_t, gecici = kaynak_orijinal, None, None, None
    if a.son_tarih:
        son_t = pd.Timestamp(a.son_tarih)
        if a.kontrol_tarih:
            knt_t = pd.Timestamp(a.kontrol_tarih)
            if knt_t <= son_t:
                sys.exit("[HATA] kontrol tarihi, son tarihten sonra olmalı.")
        gecici = cikti / "_al_kirp"
        if gecici.exists():
            shutil.rmtree(gecici)
        n_kopya = _csv_kirp(kaynak_orijinal, gecici, son_t)
        if n_kopya == 0:
            sys.exit("[HATA] kırpılan veri boş — tarih formatını kontrol edin.")
        kaynak = gecici
    aygit = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 66)
    print("LeWM-Finans AL LİSTESİ  (sızıntısız koşullar + tarihsel doğrulama)")
    print(f"koşullar: K1 gzσ≤{ESIK_GZ_SIGMA:+.0f} & |gzσ|≤{ESIK_GZ_MUTLAK:.0f} | "
          f"K2 C>MA20 | K3 MA20>MA50 | K4 r20>piyasa")
    print(f"          K5 0<r5≤%{ESIK_R5_UST * 100:.0f} | K6 hacim≥{ESIK_HACIM:.1f}×"
          f"   | eşik: skor≥{a.esik_skor}")
    if son_t is not None:
        print(f"GERİYE DÖNÜK MOD: veri {son_t.date()}'e kadar FİZİKSEL kırpıldı"
              + (f" | değerlendirme: {knt_t.date()} orijinal kapanışları"
                 if knt_t is not None else ""))
    print("=" * 66)

    # 1) veri ------------------------------------------------------------------
    model, cfg_d = _modeli_yukle(model_yolu, aygit)
    ctx, tgt = int(cfg_d["ctx_uzunluk"]), int(cfg_d["tgt_uzunluk"])
    periyot = cfg_d["periyot"]
    if periyot not in ("gunluk", "haftalik", "aylik"):
        sys.exit(f"[HATA] yalnızca günlük kaynak (checkpoint periyodu: {periyot})")
    vp = VeriPeriyot(csv_gunluk_dizini=str(kaynak), csv_60dk_dizini=str(kaynak))
    hedefler = hisseleri_bul(kaynak)
    print(f"[1/4] {len(hedefler)} hissenin surprise + fiyat/hacim serisi...")
    print(f"      sigma bildirim gecikmesi: +{tgt} bar (GZ kaydırma uygulanacak)")

    sigma, kapanis, hacim, n_ok = veri_matrisleri(
        model, vp, hedefler, periyot, ctx, tgt, aygit)
    birlesik = kapanis.index
    sigma_gz = gz_kaydir(sigma, birlesik, tgt)
    print(f"[2/4] {n_ok} hisse; fiyat {len(birlesik)} bar, "
          f"sigma {sigma.shape[0]} bar → GZ görünüm hazır")

    # 2) koşullar + piyasa rejimi ----------------------------------------------
    kosullar, skor, rel20, hacim_oran, r5, r20 = kosul_matrisleri(
        sigma_gz, kapanis, hacim, a.min_fiyat)
    liste_mask = skor >= a.esik_skor

    alarm_oran_gz = (sigma_gz.abs() > 3).sum(axis=1) / \
        sigma_gz.notna().sum(axis=1).replace(0, np.nan) * 100
    medyan5 = (kapanis / kapanis.shift(5) - 1).median(axis=1) * 100
    rejim, neden = _rejim(float(alarm_oran_gz.iloc[-1]), float(medyan5.iloc[-1]))

    print(f"[3/4] piyasa rejimi (GZ, {birlesik[-1].date()}): {rejim}"
          + (f" — {neden}" if neden else ""))
    if rejim != "NORMAL":
        print("      uyarı: AL listesi NORMAL piyasada tasarlandı; alarm piyasasında"
              " seçimler düşüşe yakalanabilir (radar: lewm_piyasa_radari.py)")
        if a.piyasa_onay:
            print("      --piyasa-onay: liste ÜRETİLMEDİ (koşul: rejim NORMAL)")
            pd.DataFrame(columns=["hisse", "kategori", "skor"]).to_csv(
                cikti / "al_listesi.csv", index=False, encoding="utf-8-sig")
            sys.exit(0)

    # 3) SON DURUM listesi -------------------------------------------------------
    son_t = birlesik[-1]
    print(f"[4/4] SON DURUM ({son_t.date()} kapanışı — yalnız o gün bilinenlerle):")
    secim = liste_mask.loc[son_t]
    satirlar = []
    for h in secim[secim].index:
        s = int(skor.at[son_t, h])
        satir = {
            "hisse": h, "kategori": kategori(s), "skor": s,
            "kapanis": round(float(kapanis.at[son_t, h]), 2),
            "gz_sigma": round(float(sigma_gz.at[son_t, h]), 2),
            "r5_%": round(float(r5.at[son_t, h]) * 100, 1),
            "r20_%": round(float(r20.at[son_t, h]) * 100, 1),
            "rel20_%": round(float(rel20.at[son_t, h]) * 100, 1),
            "hacim_x": round(float(hacim_oran.at[son_t, h]), 2),
        }
        for k in KOSUL_ADLARI:
            satir[k] = bool(kosullar[k].at[son_t, h])
        satirlar.append(satir)
    kolonlar = ["hisse", "kategori", "skor", "kapanis", "gz_sigma",
                "r5_%", "r20_%", "rel20_%", "hacim_x"] + KOSUL_ADLARI
    if a.senaryo:
        kolonlar += ["senaryo_med_%", "senaryo_yukselen_%"]
    if satirlar:
        liste = pd.DataFrame(satirlar).sort_values(
            ["skor", "rel20_%"], ascending=[False, False]).head(a.top)
    else:
        liste = pd.DataFrame(columns=kolonlar)
    if a.senaryo and len(liste) > 0:
        print("      k-NN senaryo (yavaş; listedeki her hisse için):")
        sen_med, sen_yuk = [], []
        for h in liste["hisse"]:
            try:
                oz = gelecek_tahmini_guvenli(model, vp, h, periyot, ctx, tgt,
                                             hedefler, aygit)
                sen_med.append(round(oz["medyan_senaryo_getiri_%"], 1))
                sen_yuk.append(round(oz["yukselen_orani_%"], 0))
            except Exception:
                sen_med.append(np.nan); sen_yuk.append(np.nan)
        liste["senaryo_med_%"] = sen_med
        liste["senaryo_yukselen_%"] = sen_yuk
    if len(liste) == 0:
        print("   (koşulları geçen hisse yok — piyasa geneli zayıf/alarm olabilir)")
    else:
        print(f"   {len(liste)} hisse (skor≥{a.esik_skor}; "
              f"{(liste['kategori'] == 'GÜÇLÜ AL').sum()} GÜÇLÜ AL):")
        print(f"   {'hisse':<12}{'kategori':>10}{'skor':>5}{'kapanış':>9}"
              f"{'gzσ':>7}{'r5%':>7}{'r20%':>7}{'rel%':>7}{'hacim':>7}")
        for _, s in liste.iterrows():
            sen = (f"  senaryo {s.get('senaryo_med_%', ''):+.0f}%"
                   f" yüks%{s.get('senaryo_yukselen_%', ''):.0f}"
                   if "senaryo_med_%" in liste.columns and np.isfinite(
                       s.get("senaryo_med_%", np.nan)) else "")
            print(f"   {s['hisse']:<12}{s['kategori']:>10}{s['skor']:>5}"
                  f"{s['kapanis']:>9.2f}{s['gz_sigma']:>+7.2f}{s['r5_%']:>+7.1f}"
                  f"{s['r20_%']:>+7.1f}{s['rel20_%']:>+7.1f}{s['hacim_x']:>7.2f}{sen}")
    liste.to_csv(cikti / "al_listesi.csv", index=False, encoding="utf-8-sig")

    # 3b) HINDSIGHT değerlendirme (--son-tarih + --kontrol-tarih) -----------------
    if knt_t is not None:
        _hindsight_degerlendir(liste, kaynak_orijinal, son_t, knt_t, cikti)

    # 4) DOĞRULAMA ----------------------------------------------------------------
    if not a.dogrulassiz:
        gz_rejim_serisi = pd.Series(
            [_rejim(o, m)[0] for o, m in
             zip(alarm_oran_gz, medyan5)], index=birlesik)
        dtablo = dogrulama_tablosu(liste_mask, kapanis, a.dogrula_ufuk,
                                   gz_rejim_serisi).tail(a.dogrula_bar)
        gecerli = dtablo.dropna(subset=["fark%"])
        print(f"\n--- DOĞRULAMA (son {len(dtablo)} bar, ufuk {a.dogrula_ufuk} bar;"
              f" kural yalnız o gün bilinenlerle uygulandı) ---")
        print(f"{'tarih':<12}{'boyut':>6}{'liste%':>9}{'piyasa%':>9}"
              f"{'fark%':>8}{'isabet%':>9}  rejim")
        for t, s in dtablo.tail(20).iterrows():
            print(f"{str(t.date()):<12}{s['liste_boyutu']:>6.0f}"
                  f"{_h(s['liste_ileri%'], '+.2f'):>9}{_h(s['piyasa_ileri%'], '+.2f'):>9}"
                  f"{_h(s['fark%'], '+.2f'):>8}{_h(s['isabet%'], '.0f'):>9}"
                  f"  {s['gz_rejim']}")
        if len(gecerli) > 0:
            fark_pos = (gecerli["fark%"] > 0).mean() * 100
            alarm_gun = gecerli[gecerli["gz_rejim"] == "ALARM"]
            print(f"   ÖZET ({len(gecerli)} ölçülebilir gün):"
                  f" liste medyan ileri {gecerli['liste_ileri%'].median():+.2f}%"
                  f" vs piyasa {gecerli['piyasa_ileri%'].median():+.2f}%"
                  f" → medyan fark {gecerli['fark%'].median():+.2f} pt")
            print(f"   liste>piyasa gün oranı: %{fark_pos:.0f}"
                  f" | ort. liste boyutu {gecerli['liste_boyutu'].mean():.1f}"
                  f" | GZ-ALARM günlerinde medyan fark "
                  f"{alarm_gun['fark%'].median():+.2f} pt ({len(alarm_gun)} gün)"
                  if len(alarm_gun) else
                  f"   liste>piyasa gün oranı: %{fark_pos:.0f}"
                  f" | ort. liste boyutu {gecerli['liste_boyutu'].mean():.1f}"
                  f" | ALARM günü yok")
        dtablo.to_csv(cikti / "al_dogrulama.csv", encoding="utf-8-sig")

    # dürüstlük notları ------------------------------------------------------------
    print("\n   notlar: (1) sigma GZ kaydırılmış — satır t'nin değeri t+"
          f"{tgt} kapanışında bilinir;")
    print("   (2) kapanışta hesap; gerçek işlem ertesi açılışta — ileri getiri"
          " kapanış→kapanış;")
    print("   (3) K1 kanıtı geriye dönük testten: pozitif σ kuyruğu piyasa-altı"
          " getirdi; modelin yön iddiası YOK;")
    print("   (4) checkpoint bugünkü ağırlıklar — ağırlık sızıntısı içerir"
          + (f"; girdi sızıntısı yok ({son_t.date()}'e kadar fiziksel kırpıldı)."
             if son_t is not None else "."))
    print("   (5) bu bir sinyal listesidir, yatırım tavsiyesi değildir.")
    print(f"[TAMAM] liste: {(cikti / 'al_listesi.csv').resolve()}")
    if not a.dogrulassiz:
        print(f"[TAMAM] doğrulama: {(cikti / 'al_dogrulama.csv').resolve()}")
    if knt_t is not None:
        print(f"[TAMAM] değerlendirme: {(cikti / 'al_degerlendirme.csv').resolve()}")
    if gecici is not None and not a.sakla:
        shutil.rmtree(gecici, ignore_errors=True)
        print("[TEMİZLİK] geçici kırpılmış veri silindi (--sakla ile tutulur)")


def _h(x, bicim):
    return "—" if (x is None or not np.isfinite(x)) else format(x, bicim)


def gelecek_tahmini_guvenli(model, vp, hisse, periyot, ctx, tgt, hedefler,
                            aygit):
    from lewm_uygula import gelecek_tahmini
    return gelecek_tahmini(model, vp, hisse, periyot, ctx, tgt,
                           referans_hisseler=hedefler, aygit=aygit)


if __name__ == "__main__":
    ana()
