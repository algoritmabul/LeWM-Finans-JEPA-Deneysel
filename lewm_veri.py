# -*- coding: utf-8 -*-
"""
lewm_veri.py
============
Mevcut `veri_periyot.py` dosyanızdaki VeriPeriyot sınıfını kullanarak
LeWM eğitimi için pencere (window) veri seti üretir.

Özellik tasarımı (her bar için 6 türetilmiş özellik — istasyonerlik için
ham fiyat yerine log-oranlar kullanılır):
  f1  log getiri            log(C_t / C_{t-1})
  f2  gövde                 log(C_t / O_t)
  f3  aralık                log(H_t / L_t)
  f4  kapanış konumu        (C - L) / (H - L + eps)          [0, 1]
  f5  gap                   log(O_t / C_{t-1})
  f6  log hacim             log1p(V_t)

Normalleştirme: HER PENCERE, bağlam (context) penceresinin kendi ortalama/std
istatistikleriyle z-skorlanır ve hedef pencereye AYNI istatistikler uygulanır.
Böylece (a) veri sızıntısı olmaz, (b) hisseler fiyat seviyesinden bağımsız
olarak ortak latent uzayda temsil edilebilir.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    # veri_periyot.py ile aynı klasörde duruyorsa doğrudan bulunur
    from veri_periyot import VeriPeriyot
except ImportError:                                  # pragma: no cover
    raise ImportError(
        "veri_periyot.py bulunamadı. lewm_* dosyalarını veri_periyot.py ile "
        "AYNI klasöre kopyalayın (örn. D:\\new\\yeniden\\)."
    )

OZELLIK_ADLARI = ["log_getiri", "govde", "aralik", "kapanis_konumu", "gap", "log_hacim"]
OZELLIK_SAYISI = len(OZELLIK_ADLARI)
EPS = 1e-8
KIRPMA = 8.0        # aşırı değer koruması (z-skor kırpma)
KAYIT_MAD_MAD = 1.4826  # MAD -> sigma (robut std)


def pencere_normalle(pencere: np.ndarray, baglam: np.ndarray) -> np.ndarray:
    """`pencere`yi, `baglam`ın robust (medyan/MAD) istatistikleriyle normalize eder.
    Eğitim (lewm_veri) ve çıkarım (lewm_uygula) AYNI dönüşümü kullanmalı.

    baglam: [L, F] (tek pencere) veya [N, L, F] (pencere yığını) — istatistik
    HER ZAMAN zaman ekseni (axis=-2) üzerinden, pencere başına hesaplanır.
    """
    mu = np.median(baglam, axis=-2, keepdims=True)
    mad = np.median(np.abs(baglam - mu), axis=-2, keepdims=True)
    sd = np.maximum(KAYIT_MAD_MAD * mad, 1e-4)
    return np.clip((pencere - mu) / sd, -KIRPMA, KIRPMA)


def hisseleri_bul(dizin: str | Path) -> List[str]:
    """Dizindeki tüm *.csv dosyalarını hisse kodu olarak toplar."""
    dizin = Path(dizin)
    return sorted(p.stem.upper() for p in dizin.glob("*.csv"))


def ozellik_hesapla(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame -> [L, 6] özellik matrisi (ilk bar atılır)."""
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float).clip(lower=0)

    c_once = c.shift(1)
    f = pd.DataFrame(index=df.index)
    f["log_getiri"] = np.log(c / c_once)
    f["govde"] = np.log(c / o)
    f["aralik"] = np.log(h / l)
    f["kapanis_konumu"] = (c - l) / (h - l + EPS)
    f["gap"] = np.log(o / c_once)
    f["log_hacim"] = np.log1p(v)
    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    return f


class LeWMVeriSeti(Dataset):
    """Çok hisseli pencere veri seti.

    Bir örnek = (bağlam penceresi [ctx bar], hedef pencere [tgt bar]).
    Veri bölmesi ZAMANA göredir: her hissenin son `dogrulama_oran` kısmı
    doğrulamaya ayrılır (finansal pratiğe uygun; geleceğe bakma yok).
    """

    def __init__(self,
                 vp: VeriPeriyot,
                 periyot: str = "gunluk",
                 hisseler: Optional[Sequence[str]] = None,
                 ctx_uzunluk: int = 64,
                 tgt_uzunluk: int = 8,
                 bolum: str = "egitim",
                 dogrulama_oran: float = 0.10,
                 sustur: bool = False):
        self.vp = vp
        self.periyot = periyot.lower()
        self.ctx = ctx_uzunluk
        self.tgt = tgt_uzunluk
        self.bolum = bolum

        if hisseler is None:
            dizin = (vp.csv_gunluk_dizini if self.periyot in ("gunluk", "haftalik", "aylik")
                     else vp.csv_60dk_dizini)
            hisseler = hisseleri_bul(dizin)
        self.hisse_adlari: List[str] = list(hisseler)

        self.diziler: List[np.ndarray] = []       # her hisse için [L, 6] float32
        self.tarihler: List[pd.DatetimeIndex] = []
        self.ornek_indeksi: List[Tuple[int, int]] = []   # (hisse_idx, baslangic)

        n_atlanan = 0
        for hi, hisse in enumerate(self.hisse_adlari):
            try:
                df = vp.get(hisse, self.periyot)
            except Exception:
                df = None
            if df is None or len(df) < self.ctx + self.tgt + 2:
                n_atlanan += 1
                continue
            oz = ozellik_hesapla(df)
            if len(oz) < self.ctx + self.tgt:
                n_atlanan += 1
                continue
            tarih = oz.index
            arr = oz.to_numpy(dtype=np.float32)
            self.diziler.append(arr)
            self.tarihler.append(tarih)

            # geçerli başlangıçlar: [0, L - ctx - tgt]
            maks_bas = len(oz) - self.ctx - self.tgt
            kesme = int(maks_bas * (1.0 - dogrulama_oran))
            aralik = range(0, maks_bas) if bolum == "egitim" else range(kesme, maks_bas)
            for s in aralik:
                self.ornek_indeksi.append((len(self.diziler) - 1, s))

        if n_atlanan and not sustur:
            print(f"   [BILGI] {n_atlanan} hisse yetersiz veri nedeniyle atlandı.")

        if len(self.ornek_indeksi) == 0:
            raise ValueError(
                f"'{bolum}' bölmesi için uygun pencere bulunamadı. "
                "CSV'leri ve ctx/tgt uzunluklarını kontrol edin."
            )

    def __len__(self) -> int:
        return len(self.ornek_indeksi)

    def _z_skorla(self, baglam: np.ndarray, hedef: np.ndarray):
        """Bağlam robust (medyan/MAD) istatistikleriyle iki pencereyi de normalize et.

        Tek bir ani hareket (gap, %10'luk zıplama) bağlam std'sini şişirip
        anomaliyi seyreltemez — anomali barı normalize sonrası büyük kalır.
        """
        return (pencere_normalle(baglam, baglam),
                pencere_normalle(hedef, baglam))

    def __getitem__(self, i: int):
        hi, s = self.ornek_indeksi[i]
        arr = self.diziler[hi]
        baglam = arr[s: s + self.ctx]
        hedef = arr[s + self.ctx: s + self.ctx + self.tgt]
        ctx_n, tgt_n = self._z_skorla(baglam, hedef)
        return (torch.from_numpy(ctx_n.astype(np.float32)),
                torch.from_numpy(tgt_n.astype(np.float32)))

    def meta(self, i: int) -> dict:
        """İndeksin insan-okur bilgisi (uygulama/rapor adımları için)."""
        hi, s = self.ornek_indeksi[i]
        tarih = self.tarihler[hi]
        return {
            "hisse": self.hisse_adlari[hi],
            "baglam_sonu": tarih[s + self.ctx - 1],
            "hedef_basi": tarih[s + self.ctx],
            "hedef_sonu": tarih[min(s + self.ctx + self.tgt - 1, len(tarih) - 1)],
        }

    def stok_getirisi(self, hi: int, s: int) -> float:
        """(hi, s) penceresinin hedef bölgesindeki toplam getiri (k-NN senaryo analizi).

        Hedef bölge log-getirilerinin toplamı -> basit getiriye çevrilir.
        """
        arr = self.diziler[hi]
        toplam = float(arr[s + self.ctx: s + self.ctx + self.tgt, 0].sum())
        return float(np.expm1(toplam))


def veri_seti_olustur(vp: VeriPeriyot, cfg) -> Tuple["LeWMVeriSeti", "LeWMVeriSeti"]:
    """(egitim, dogrulama) veri seti çifti üretir."""
    egitim = LeWMVeriSeti(vp, cfg.periyot, cfg.hisseler, cfg.ctx_uzunluk,
                          cfg.tgt_uzunluk, bolum="egitim")
    dogrulama = LeWMVeriSeti(vp, cfg.periyot, cfg.hisseler, cfg.ctx_uzunluk,
                             cfg.tgt_uzunluk, bolum="dogrulama", sustur=True)
    return egitim, dogrulama
