# -*- coding: utf-8 -*-
"""
lewm_rapor.py
=============
Eğitim grafikleri, sürpriz zaman serisi, embedding haritası (PCA) ve
tarama raporları (CSV).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")            # ekransız ortamda güvenli
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BANT_RENK = "#d62728"

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def egitim_grafigi(gecmis: dict, cikti_dizin: str | Path) -> Path:
    """Toplam / tahmin / SIGReg kayıp eğrileri + doğrulama."""
    dosya = Path(cikti_dizin) / "egitim_eğrileri.png"
    ep = gecmis["epoch"]
    fig, aks = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    aks[0].plot(ep, gecmis["kayip"], label="Toplam kayıp", lw=1.8)
    aks[0].plot(ep, gecmis["kayip_tahmin"], label="Tahmin (MSE)", lw=1.4)
    aks[0].plot(ep, gecmis["kayip_sigreg"], label="SIGReg", lw=1.4)
    aks[0].set_yscale("log")
    aks[0].set_xlabel("Epoch"); aks[0].set_ylabel("Kayıp")
    aks[0].set_title("Eğitim kayıpları")
    aks[0].legend(frameon=False)

    aks[1].plot(ep, gecmis["dogrulama_kayip"], color="#2ca02c", label="Doğrulama MSE", lw=1.8)
    aks[1].set_xlabel("Epoch"); aks[1].set_ylabel("Doğrulama tahmin MSE")
    aks[1].set_title("Doğrulama (zaman bölmesi)")
    aks[1].legend(frameon=False)

    fig.savefig(dosya, dpi=150)
    plt.close(fig)
    return dosya


def surprise_grafigi(tablo: pd.DataFrame, hisse: str,
                     cikti_dizin: str | Path, esik: float = 3.0) -> Path:
    """Üst: kapanış fiyatı; alt: sigma skoru (±3σ bantlı)."""
    dosya = Path(cikti_dizin) / f"surprise_{hisse.upper()}.png"
    d = tablo.dropna(subset=["sigma_skoru"])
    fig, aks = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True,
                            constrained_layout=True)

    aks[0].plot(d["tarih"], d["kapanis"], color="#1f77b4", lw=1.2)
    aks[0].set_ylabel("Kapanış")
    aks[0].set_title(f"{hisse.upper()} — LeWM sürpriz analizi")

    aks[1].plot(d["tarih"], d["sigma_skoru"], color="#444444", lw=0.9)
    aks[1].axhline(esik, color=BANT_RENK, ls="--", lw=1, label=f"+{esik}σ")
    aks[1].axhline(-esik, color=BANT_RENK, ls="--", lw=1, label=f"−{esik}σ")
    asiri = d[d["sigma_skoru"].abs() > esik]
    aks[1].scatter(asiri["tarih"], asiri["sigma_skoru"], color=BANT_RENK,
                   s=14, zorder=3, label="anomali")
    aks[1].set_ylabel("Sürpriz (σ)")
    aks[1].legend(frameon=False, loc="upper left", ncol=3)

    fig.savefig(dosya, dpi=150)
    plt.close(fig)
    return dosya


def embedding_haritasi(gomme_tablo: pd.DataFrame, cikti_dizin: str | Path,
                       etiket: Optional[pd.Series] = None) -> Path:
    """Gömmelerin PCA (2B) haritası — benzer davranışlı hisseler yakın durur."""
    dosya = Path(cikti_dizin) / "embedding_haritasi.png"
    kolonlar = [c for c in gomme_tablo.columns if c.startswith("z")]
    if len(gomme_tablo) < 3:
        return dosya
    Z = gomme_tablo[kolonlar].to_numpy(dtype=np.float64)
    Z = Z - Z.mean(0)
    # PCA: SVD ile ilk 2 bileşen
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    P = Z @ Vt[:2].T
    var_oran = S[:2] ** 2 / np.sum(S ** 2)

    hisse = gomme_tablo["hisse"].to_numpy()
    benzersiz = sorted(pd.unique(hisse))
    renkler = plt.cm.tab20(np.linspace(0, 1, max(2, len(benzersiz))))
    fig, aks = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    for i, h in enumerate(benzersiz):
        m = hisse == h
        aks.scatter(P[m, 0], P[m, 1], s=18, alpha=0.75,
                    color=renkler[i % len(renkler)], label=h)
    aks.set_xlabel(f"PC1 (%{var_oran[0]*100:.0f} varyans)")
    aks.set_ylabel(f"PC2 (%{var_oran[1]*100:.0f} varyans)")
    aks.set_title("LeWM embedding haritası (PCA)")
    if len(benzersiz) <= 20:
        aks.legend(frameon=False, fontsize=8, markerscale=1.4,
                   loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(dosya, dpi=150)
    plt.close(fig)

    # PCA koordinatlarını da CSV'ye yaz
    pd.DataFrame({
        "hisse": hisse,
        "tarih": gomme_tablo["tarih"].to_numpy(),
        "PC1": P[:, 0], "PC2": P[:, 1],
    }).to_csv(Path(cikti_dizin) / "embedding_pca.csv", index=False, encoding="utf-8-sig")
    return dosya


def tarama_raporu(gomme_tablo: pd.DataFrame, model, vp, cfg_dict: dict,
                  cikti_dizin: str | Path, aygit: str = "cpu") -> Path:
    """Son pencere gömmeleri + benzerlik tabanlı 'en yakın 3 komşu' listesi ile
    günlük tarama tablosu üretir (kendi tarama sisteminize bağlanabilir)."""
    from lewm_uygula import gommeleri_cikar
    dosya = Path(cikti_dizin) / "tarama_son_gommeler.csv"
    if len(gomme_tablo) == 0:      # hiç hisse bulunamadıysa boş tabloyla çık
        return dosya
    kolonlar = [c for c in gomme_tablo.columns if c.startswith("z")]
    Z = gomme_tablo[kolonlar].to_numpy(np.float32)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    benzerlik = Zn @ Zn.T
    np.fill_diagonal(benzerlik, -1)
    en_yakin = benzerlik.argmax(axis=1)
    rapor = pd.DataFrame({
        "hisse": gomme_tablo["hisse"].to_numpy(),
        "tarih": gomme_tablo["tarih"].astype(str).to_numpy(),
        "en_benzer_hisse": gomme_tablo["hisse"].to_numpy()[en_yakin],
        "en_benzer_tarih": gomme_tablo["tarih"].astype(str).to_numpy()[en_yakin],
        "benzerlik": np.round(benzerlik[np.arange(len(benzerlik)), en_yakin], 4),
    }).sort_values("hisse")
    rapor.to_csv(dosya, index=False, encoding="utf-8-sig")
    return dosya
