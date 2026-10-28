# -*- coding: utf-8 -*-
"""
lewm_geriye_donuk.py — "Ne yapsaydık, ne olurdu?" geriye dönük testi
=====================================================================
Belirli bir SON TARİH'e kadar olan veriyle (sonrasını GÖRMEDEN) surprise
taraması yapar, listeyi üretir; sonra listelenen hisselerin KONTROL
TARİHİ'ndeki kapanışlarıyla gerçek getirisini ölçer.

Akış:
 1) Günlük CSV'ler okunur, SON TARİH'e kadar olan satırlar geçici dizine
    yazılır  -> model sonrasını FİZİKSEL olarak göremez (girdi sızıntısı yok)
 2) Checkpoint yüklenir; tüm hisselerde surprise taraması (uygula --tum ile
    aynı kod yolu: surprise_skorlari)
 3) TOP-N listesi |sigma| ile üretilir
 4) ORİJİNAL CSV'lerden referans (son tarih) ve kontrol (kontrol tarihi)
    kapanışları okunur; 16.10 -> 23.10 tarzı getiri hesaplanır
 5) Piyasa benchmark'ı (taranan TÜM hisseler) ile kıyas + özet istatistik
 6) Rapor: geriye_donuk_rapor.csv (tüm hisseler, listede_mi bayrağıyla)

Dürüstlük kaydı: mevcut checkpoint muhtemelen son tarihten SONRAKİ verilerle
de eğitildi (ağırlık sızıntısı). Bu betik GİRDİ sızıntısını engeller; tam
temiz bir walk-forward için modelin yalnızca son tarihe kadar eğitilmesi gerekir.

Kullanım:
  python lewm_geriye_donuk.py
  python lewm_geriye_donuk.py --son-tarih 2026-10-16 --kontrol-tarih 2026-10-23 --top 20
  python lewm_geriye_donuk.py --gunluk "D:\\new\\yeniden\\metastock-gun-csv" --sakla
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


# --------------------------------------------------------------------------
def _csv_kirp(kaynak: Path, hedef: Path, son_tarih: pd.Timestamp) -> int:
    """Kaynak dizindeki tüm CSV'leri okur, yalnızca son_tarih'e kadar olan
    satırları hedef dizine yazar. Kopyalanan dosya sayısını döndürür."""
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
    """ORİJİNAL CSV'den (ref_tarih, ref_kapanis, kontrol_tarih, kontrol_kapanis).
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
    if ref.empty or knt.empty:
        return (None if ref.empty else float(ref["Close"].iloc[0]),
                None if ref.empty else str(ref["_dt"].iloc[0].date()),
                None if knt.empty else float(knt["Close"].iloc[0]),
                None if knt.empty else str(knt["_dt"].iloc[0].date()))
    return (float(ref["Close"].iloc[0]), str(ref["_dt"].iloc[0].date()),
            float(knt["Close"].iloc[0]), str(knt["_dt"].iloc[0].date()))


def _istat(df: pd.DataFrame, ad: str) -> dict:
    g = df["getiri_%"].dropna()
    if len(g) == 0:
        return {"grup": ad, "n": 0}
    return {"grup": ad, "n": len(g),
            "medyan_%": float(g.median()), "ortalama_%": float(g.mean()),
            "yukselen_%": float((g > 0).mean() * 100),
            "medyan_|getiri|_%": float(g.abs().median()),
            "en_iyi_%": float(g.max()), "en_kotu_%": float(g.min())}


def _satir_yaz(s: dict) -> str:
    if s["n"] == 0:
        return f"   {s['grup']:<24} veri yok"
    return (f"   {s['grup']:<24} n={s['n']:<4} medyan={s['medyan_%']:+6.2f}%  "
            f"ort={s['ortalama_%']:+6.2f}%  yükselen=%{s['yukselen_%']:4.0f}  "
            f"medyan|g|={s['medyan_|getiri|_%']:5.2f}%  "
            f"[{s['en_kotu_%']:+.1f} .. {s['en_iyi_%']:+.1f}]")


def _korelasyon(x: pd.Series, y: pd.Series) -> float:
    maske = x.notna() & y.notna()
    if maske.sum() < 10 or x[maske].std() == 0 or y[maske].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[maske], y[maske])[0, 1])


# --------------------------------------------------------------------------
def ana():
    p = argparse.ArgumentParser(
        description="LeWM geriye dönük test: son tarihe kadar tarama, "
                    "kontrol tarihinde gerçek sonuç")
    p.add_argument("--son-tarih", default="2026-10-16",
                   help="taramanın göreceği son gün (varsayılan: 2026-10-16)")
    p.add_argument("--kontrol-tarih", default="2026-10-23",
                   help="sonuçların ölçüleceği gün (varsayılan: 2026-10-23)")
    p.add_argument("--top", type=int, default=20,
                   help="|sigma| ile seçilecek liste büyüklüğü (varsayılan 20)")
    p.add_argument("--gunluk", default=r"D:\new\yeniden\metastock-gun-csv",
                   help="orijinal günlük CSV dizini")
    p.add_argument("--cikti", default="lewm_cikti", help="çıktı dizini")
    p.add_argument("--model", default=None,
                   help="checkpoint yolu (varsayılan: <cikti>/lewm_model_eniyi.pt)")
    p.add_argument("--sakla", action="store_true",
                   help="kırpılmış geçici veri dizinini silme")
    a = p.parse_args()

    son_t = pd.Timestamp(a.son_tarih)
    knt_t = pd.Timestamp(a.kontrol_tarih)
    if knt_t <= son_t:
        sys.exit("[HATA] kontrol tarihi, son tarihten sonra olmalı.")
    kaynak = Path(a.gunluk)
    if not kaynak.exists():
        sys.exit(f"[HATA] günlük CSV dizini yok: {kaynak}")
    cikti = Path(a.cikti)
    cikti.mkdir(parents=True, exist_ok=True)
    model_yolu = Path(a.model) if a.model else cikti / "lewm_model_eniyi.pt"
    if not model_yolu.exists():
        sys.exit(f"[HATA] model bulunamadı: {model_yolu}\n"
                 f"       Önce çalıştırın: python lewm_cli.py egit")
    aygit = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 66)
    print("LeWM-Finans GERİYE DÖNÜK TEST  ('ne yapsaydık, ne olurdu?')")
    print(f"son tarih: {son_t.date()}  ->  kontrol: {knt_t.date()}  "
          f"(TOP {a.top}, |sigma| sıralı)")
    print("=" * 66)

    # 1) Veriyi FİZİKSEL olarak kırp --------------------------------------
    gecici = cikti / "_geriye_donuk_veri"
    if gecici.exists():
        shutil.rmtree(gecici)
    n_kopya = _csv_kirp(kaynak, gecici, son_t)
    if n_kopya == 0:
        sys.exit("[HATA] kırpılan veri boş — tarih formatını kontrol edin.")
    print(f"[1/4] {n_kopya} CSV '{son_t.date()}' tarihine kadar kırpıldı "
          f"(sonrası modelin erişiminin DIŞINDA)")

    # 2) Model + tarama (uygula --tum ile aynı kod yolu) -------------------
    model, cfg_d = _modeli_yukle(model_yolu, aygit)
    ctx, tgt = cfg_d["ctx_uzunluk"], cfg_d["tgt_uzunluk"]
    periyot = cfg_d["periyot"]
    if periyot not in ("gunluk", "haftalik", "aylik"):
        sys.exit(f"[HATA] bu betik yalnızca günlük kaynakla çalışır "
                 f"(checkpoint periyodu: {periyot}).")
    print(f"[2/4] model yüklendi ({model_yolu.name}, ctx={ctx}, tgt={tgt}) "
          f"— tarama başlıyor...")

    vp = VeriPeriyot(csv_gunluk_dizini=str(gecici), csv_60dk_dizini=str(gecici))
    hedefler = hisseleri_bul(gecici)
    satirlar, son_bar = [], None
    for i, h in enumerate(hedefler, 1):
        try:
            tablo = surprise_skorlari(model, vp, h, periyot, ctx, tgt,
                                      aygit=aygit)
        except Exception:
            continue
        son = tablo.dropna(subset=["sigma_skoru"]).tail(1)
        if len(son) == 0:
            continue
        if son_bar is None:
            son_bar = str(pd.Timestamp(son["tarih"].iloc[0]).date())
        satirlar.append({"hisse": h,
                         "sigma": float(son["sigma_skoru"].iloc[0]),
                         "surprise": float(son["surprise"].iloc[0]),
                         "tarama_tarihi": son_bar})
        if i % 100 == 0:
            print(f"      taranan: {i}/{len(hedefler)}")
    tarama = pd.DataFrame(satirlar)
    if tarama.empty:
        sys.exit("[HATA] hiç hisse skorlanamadı.")
    n_alarm = int((tarama["sigma"].abs() > 3).sum())
    print(f"[3/4] tarama bitti: {len(tarama)} hisse skorlandı "
          f"(son görülen bar: {son_bar}), {n_alarm} tanesi ±3σ üstünde")

    # 3) ORİJİNAL CSV'lerden gerçek sonuçlar -------------------------------
    ref_list, knt_list = [], []
    for _, r in tarama.iterrows():
        ref_c, ref_d, knt_c, knt_d = _kapanis_al(
            kaynak / f"{r['hisse']}.csv", son_t, knt_t)
        ref_list.append((ref_c, ref_d))
        knt_list.append((knt_c, knt_d))
    tarama["ref_kapanis"] = [x[0] for x in ref_list]
    tarama["ref_tarih"] = [x[1] for x in ref_list]
    tarama["kontrol_kapanis"] = [x[0] for x in knt_list]
    tarama["kontrol_tarih"] = [x[1] for x in knt_list]
    tarama["getiri_%"] = np.where(
        tarama["ref_kapanis"].notna() & tarama["kontrol_kapanis"].notna()
        & (tarama["ref_kapanis"] != 0),
        (tarama["kontrol_kapanis"] / tarama["ref_kapanis"] - 1) * 100,
        np.nan)
    tarama["_abs"] = tarama["sigma"].abs()
    tarama = tarama.sort_values("_abs", ascending=False)
    tarama["listede_mi"] = False
    tarama.iloc[:a.top, tarama.columns.get_loc("listede_mi")] = True

    # 4) Rapor --------------------------------------------------------------
    liste = tarama[tarama["listede_mi"]]
    alarm = tarama[tarama["_abs"] > 3]
    print(f"\n--- LİSTE (TOP {min(a.top, len(liste))}) ve "
          f"{son_t.date()} → {knt_t.date()} GETİRİLERİ ---")
    for _, r in liste.iterrows():
        isaret = "!" if r["_abs"] > 3 else " "
        if pd.isna(r["getiri_%"]):
            sonuc = "veri yok"
        else:
            sonuc = f"{r['ref_kapanis']:.2f} → {r['kontrol_kapanis']:.2f}  " \
                    f"{r['getiri_%']:+.2f}%"
        print(f"   {isaret} {r['hisse']:<12} σ={r['sigma']:+6.2f}  "
              f"skor={r['surprise']:.4f} | {sonuc}")

    print(f"\n--- ÖZET ('ne yaparsak ne olur') ---")
    ozet = [_istat(liste, f"TOP-{a.top} liste"),
            _istat(alarm, "±3σ alarm (tümü)"),
            _istat(tarama, "PİYASA (taranan tümü)")]
    for s in ozet:
        print(_satir_yaz(s))

    k1 = _korelasyon(tarama["sigma"], tarama["getiri_%"])
    k2 = _korelasyon(tarama["_abs"], tarama["getiri_%"].abs())
    print(f"\n   σ  ↔ getiri  korelasyonu : {k1:+.2f}  "
          f"(sürpriz yön öngörüyor mu? ~0 ise hayır)")
    print(f"   |σ| ↔ |getiri| korelasyonu: {k2:+.2f}  "
          f"(sürpriz sonraki hareketi BÜYÜTÜYOR mu?)")

    rapor = (tarama.sort_values("_abs", ascending=False, na_position="last")
             .drop(columns="_abs"))
    yol = cikti / "geriye_donuk_rapor.csv"
    rapor.to_csv(yol, index=False, encoding="utf-8-sig")
    print(f"\n   not: model ağırlıkları muhtemelen {son_t.date()} sonrasını "
          f"da gördü; bu test girdi sızıntısını engeller,")
    print("        ağırlık sızıntısını içerir (tam walk-forward için "
          "modeli yalnızca son tarihe kadar eğitin).")
    print(f"[TAMAM] rapor: {yol.resolve()}")
    if not a.sakla:
        shutil.rmtree(gecici, ignore_errors=True)
        print("[TEMİZLİK] geçici kırpılmış veri silindi (--sakla ile tutulur)")


if __name__ == "__main__":
    ana()
