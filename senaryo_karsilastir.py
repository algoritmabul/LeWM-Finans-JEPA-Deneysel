# -*- coding: utf-8 -*-
"""
senaryo_karsilastir.py — İki senaryo_tarama.csv'yi karşılaştırır (model stabilite testi).

Kullanım:
    python senaryo_karsilastir.py  eski\senaryo_tarama.csv  yeni\senaryo_tarama.csv
    python senaryo_karsilastir.py  eski.csv yeni.csv --top 20

Ölçütler (sertlik sırasıyla):
  1) top-N kesişim            — sıralama kararlılığı (en sert ölçüt)
  2) pozitif-medyan kümesi    — yön kararlılığı + Jaccard (daha yumuşak)
  3) Spearman sıra korelasyonu — ortak hisselerde medyan getiri sırası
  4) yön değişenler           — pozitif→negatif ve negatif→pozitif dönenler
  5) en büyük sıra kaymaları  — ortak hisselerde rütbe atlayışları
Ayrıca baglam_sonu tarihlerini yazdırır (koşu günleri farklıysa kesişimin
düşüklüğü "model epoch'u" değil "piyasa günü" kaynaklı olabilir — karışık
değişken uyarısı).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def yukle(yol: str) -> pd.DataFrame:
    p = Path(yol)
    if not p.exists():
        raise SystemExit(f"[HATA] dosya bulunamadı: {p}")
    df = pd.read_csv(p)
    gerekli = {"hisse", "medyan_senaryo_getiri_%"}
    eksik = gerekli - set(df.columns)
    if eksik:
        raise SystemExit(f"[HATA] {p.name}: kolon(lar) eksik: {sorted(eksik)}")
    df["hisse"] = df["hisse"].astype(str).str.strip()
    df = df.drop_duplicates(subset="hisse").reset_index(drop=True)
    return df.sort_values("medyan_senaryo_getiri_%", ascending=False).reset_index(drop=True)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy(dtype=float)
    rb = pd.Series(b).rank().to_numpy(dtype=float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    p = argparse.ArgumentParser(description="İki senaryo taraması karşılaştırması")
    p.add_argument("eski", help="ilk senaryo_tarama.csv (ör. eski model)")
    p.add_argument("yeni", help="ikinci senaryo_tarama.csv (ör. yeni model)")
    p.add_argument("--top", type=int, default=20, help="top-N kesişim için N (vars. 20)")
    a = p.parse_args()

    eski = yukle(a.eski)
    yeni = yukle(a.yeni)

    print("=" * 66)
    print("SENARYO KARŞILAŞTIRMA")
    print("=" * 66)

    # --- koşu günleri (karışık değişken teşhisi) -----------------------
    for ad, df in (("eski", eski), ("yeni", yeni)):
        if "baglam_sonu" in df.columns:
            print(f"[{ad.upper()}] {len(df)} hisse | bağlam sonu: "
                  f"{df['baglam_sonu'].min()} .. {df['baglam_sonu'].max()}")
        else:
            print(f"[{ad.upper()}] {len(df)} hisse | baglam_sonu kolonu yok")

    pos_e = eski[eski["medyan_senaryo_getiri_%"] > 0]["hisse"].tolist()
    pos_y = yeni[yeni["medyan_senaryo_getiri_%"] > 0]["hisse"].tolist()

    # --- 1) top-N kesişim ----------------------------------------------
    top_e = eski.head(a.top)["hisse"].tolist()
    top_y = yeni.head(a.top)["hisse"].tolist()
    kesisim = sorted(set(top_e) & set(top_y))
    print(f"\n[1] top-{a.top} kesişim: {len(kesisim)}/{a.top}")
    if kesisim:
        print("    ortak:", ", ".join(kesisim))
    print(f"    top-{a.top} eşiği — eski: {eski['medyan_senaryo_getiri_%'].iloc[a.top-1]:+.2f}% | "
          f"yeni: {yeni['medyan_senaryo_getiri_%'].iloc[a.top-1]:+.2f}% "
          f"(eşikler yakınsa sıralama, değil küme değişimi okunmalı)")

    # --- 2) pozitif küme kesişimi + Jaccard -----------------------------
    se, sy = set(pos_e), set(pos_y)
    ortak_kume = se & sy
    jacc = len(ortak_kume) / len(se | sy) if (se | sy) else float("nan")
    print(f"\n[2] pozitif-medyan kümesi: eski {len(se)} | yeni {len(sy)} | "
          f"ortak {len(ortak_kume)} | Jaccard {jacc:.2f}")
    print("    (Jaccard ~0.5 üstü = 'model hangi hisseleri beğeniyor' fikri "
          "epoch'lar arası korunmuş demektir)")

    # --- 3) Spearman sıra korelasyonu -----------------------------------
    m = eski.merge(yeni, on="hisse", suffixes=("_e", "_y"))
    rho = spearman(m["medyan_senaryo_getiri_%_e"].to_numpy(),
                   m["medyan_senaryo_getiri_%_y"].to_numpy())
    poz = m[m["medyan_senaryo_getiri_%_e"] > 0]
    rho_poz = spearman(poz["medyan_senaryo_getiri_%_e"].to_numpy(),
                       poz["medyan_senaryo_getiri_%_y"].to_numpy())
    print(f"\n[3] Spearman sıra korelasyonu (ortak {len(m)} hisse): {rho:.3f}")
    print(f"    yalnız eski-pozitiflerde ({len(poz)}): {rho_poz:.3f}")
    print("    (0.4-0.6 orta; 0.6 üstü güçlü kararlılık; düşükse pozitif "
          "kümesinin İÇ sırası da oynuyor)")

    # --- 4) yön değişenler -----------------------------------------------
    p2n = sorted(se - sy)
    n2p = sorted(sy - se)
    print(f"\n[4] yön değişenler: pozitif→negatif {len(p2n)} | negatif→pozitif {len(n2p)}")
    for ad, liste in (("p→n", p2n), ("n→p", n2p)):
        if liste:
            print(f"    {ad}: {', '.join(liste[:8])}{' ...' if len(liste) > 8 else ''}")

    # --- 5) en büyük sıra kaymaları -------------------------------------
    m["sira_e"] = m["medyan_senaryo_getiri_%_e"].rank(ascending=False)
    m["sira_y"] = m["medyan_senaryo_getiri_%_y"].rank(ascending=False)
    m["kayma"] = (m["sira_y"] - m["sira_e"]).abs()
    buyuk = m[m["hisse"].isin(ortak_kume)].sort_values("kayma", ascending=False).head(6)
    if len(buyuk):
        print("\n[5] en büyük sıra kayması (ortak pozitifler):")
        for _, r in buyuk.iterrows():
            print(f"    {r['hisse']:<12} {int(r['sira_e'])} → {int(r['sira_y'])} "
                  f"(kayma {int(r['kayma'])}) | medyan {r['medyan_senaryo_getiri_%_e']:+.2f}% "
                  f"→ {r['medyan_senaryo_getiri_%_y']:+.2f}%")

    print("\n[OKUMA] top-N kesişimi düşük ama eşikler (top-N sınır değeri) ve "
          "Jaccard/Spearman yüksekse: model kararlı, yalnız sıralama hassas → "
          "kararı top-N kesitine değil pozitif kümesine bağla.")
    print("        baglam_sonu günleri farklıysa: kesişim farkında piyasa hareketi "
          "etkilidir — model epoch etkisini yalıtmak için iki modeli AYNI gün "
          "kesitinde koşmak gerekir.")


if __name__ == "__main__":
    main()
