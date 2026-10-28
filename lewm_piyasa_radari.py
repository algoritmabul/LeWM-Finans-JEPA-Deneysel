# -*- coding: utf-8 -*-
"""
lewm_piyasa_radari.py — Piyasa geneli çöküş/kriz radarı (v2: gerçek-zamanlı görünüm)
=====================================================================================
Bireysel hisse sürprizlerini KESİTSEL (cross-sectional) okuyup piyasa
rejimini verir: çöküş günlerinde onlarca hisse AYNI anda ±3σ üstüne fırlar —
bu toplu sapma, bireysel gürültüden ayırt edilebilir piyasa alarmıdır.

İki radar birden:
  A) SIGMA RADARI  : |σ|>3 hisse oranı (Gauss beklentisi ~%0.3;
                     >%1 TEDBİR, >%4 ALARM)
  B) FİYAT RADARI  : piyasa medyan son-5-bar getiri (anlık, gecikmesiz;
                     <-%3 TEDBİR, <-%7 ALARM)

ÖNEMLİ — BİLDİRİM GECİKMESİ (v2'de dürüstçe modellendi):
  Sigma satırı t, t anında kurulan bağlamla İZLEYEN tgt barın gerçekleşen
  yolu arasındaki sapmadır; değeri ancak t+tgt kapanışında hesaplanabilir.
  Yani tablodaki sigma kolonları TAHMİN-tarihlidir; o gün canlıda görülen
  okuma GZ (gerçek-zamanlı) kolonlarıdır:
      gz_sigma(D) = sigma_tarama(D − tgt)      (en son bilinen tarama)
      gz_fiyat(D) = fiyat(D)                   (anlık)
  Yakalama analizi (olaylar/bölümler) GZ görünüm üzerinden yapılır —
  aksi hâlde radar çöküşten ÖNCE uyarmış gibi görünür ama o okuma o gün
  henüz yoktu.

Kullanım:
  python lewm_piyasa_radari.py
  python lewm_piyasa_radari.py --son-bar 60 --cokus-esik -5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lewm_veri import hisseleri_bul
from lewm_uygula import _modeli_yukle, surprise_skorlari
from veri_periyot import VeriPeriyot

ESIK_ALARM_ORAN = 4.0    # |σ|>3 hisse yüzdesi üstü -> ALARM
ESIK_TEDBIR_ORAN = 1.0   # üstü -> TEDBİR
ESIK_DUSUS_ALARM = -7.0  # medyan 5-bar getiri altı -> ALARM
ESIK_DUSUS_TEDBIR = -3.0 # üstü -> TEDBİR

ONCEDEN_PENCERE = 5      # olay başlangıcından kaç bar öncesi radar için taranır
BOLUM_BOSLUK = 2         # olay/bölüm tespitinde birleştirilecek maksimum boş bar


def _rejim(alarm_oran: float, medyan5: float) -> tuple:
    neden, seviye = [], 0
    if np.isfinite(alarm_oran):
        if alarm_oran > ESIK_ALARM_ORAN:
            seviye = 2; neden.append(f"sigma %{alarm_oran:.1f}>{ESIK_ALARM_ORAN:.0f}")
        elif alarm_oran > ESIK_TEDBIR_ORAN:
            seviye = max(seviye, 1); neden.append(f"sigma %{alarm_oran:.1f}")
    if np.isfinite(medyan5):
        if medyan5 < ESIK_DUSUS_ALARM:
            seviye = 2; neden.append(f"fiyat {medyan5:.1f}%")
        elif medyan5 < ESIK_DUSUS_TEDBIR:
            seviye = max(seviye, 1); neden.append(f"fiyat {medyan5:.1f}%")
    return ("ALARM" if seviye == 2 else "TEDBİR" if seviye == 1 else "NORMAL",
            " + ".join(neden))


def _seri_bolumler(mask: pd.Series, izin: int = BOLUM_BOSLUK):
    """mask True olan maksimal aralıklar [(baş, bitiş), ...]; ≤izin boşluk birleştirilir."""
    idx = mask.index
    gunler = idx[mask.fillna(False).to_numpy()]
    if len(gunler) == 0:
        return []
    bolumler, bas, onceki = [], gunler[0], gunler[0]
    onceki_p = idx.get_loc(gunler[0])
    for g in gunler[1:]:
        p = idx.get_loc(g)
        if p - onceki_p > izin + 1:
            bolumler.append((bas, onceki))
            bas = g
        onceki, onceki_p = g, p
    bolumler.append((bas, onceki))
    return bolumler


def _hücre(x, bicim: str, bos: str = "—") -> str:
    return bos if (x is None or not np.isfinite(x)) else format(x, bicim)


def _grafik(tablo: pd.DataFrame, yol: Path, son_bar: int, bolumler):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = tablo.tail(son_bar)
    fig, aks1 = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    # gerçek-zamanlı ALARM bölümlerini gölgele
    for b in bolumler:
        aks1.axvspan(b["baslangic"], b["bitis"], color="#c62828", alpha=0.08, lw=0)
    aks1.plot(v.index, v["alarm_oran_%"], color="#ef9a9a", lw=1.4,
              label="±3σ oranı % (tahmin-tarihli, değeri +N barda bilinir)")
    gz_renk = "#c62828"
    aks1.plot(v.index, v["gz_alarm_oran_%"], color=gz_renk, lw=1.8, ls="--",
              label="±3σ oranı % (GZ: o gün bilinen)")
    aks1.axhline(ESIK_ALARM_ORAN, color=gz_renk, ls=":", lw=0.8, alpha=0.6)
    aks1.axhline(ESIK_TEDBIR_ORAN, color="#ef6c00", ls=":", lw=0.8, alpha=0.6)
    aks1.set_ylabel("±3σ alarm oranı (%)", color=gz_renk)
    aks2 = aks1.twinx()
    renk = np.where(v["piyasa_5bar_%"] < 0, "#37474f", "#66bb6a")
    aks2.bar(v.index, v["piyasa_5bar_%"], color=renk, alpha=0.45, width=1.0,
             label="piyasa medyan 5-bar getiri %")
    aks2.set_ylabel("piyasa medyan 5-bar getiri (%)")
    aks1.set_title("Piyasa radarı — kesitsel sigma alarm oranı vs piyasa getiri\n"
                   "(kırmızı gölgeler: gerçek-zamanlı ALARM bölümleri)")
    h1, l1 = aks1.get_legend_handles_labels()
    h2, l2 = aks2.get_legend_handles_labels()
    aks1.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(0.0, -0.12),
                ncol=3, fontsize=8, frameon=False)
    fig.autofmt_xdate()
    fig.savefig(yol, dpi=140)
    plt.close(fig)


def _yakalama_analizi(tablo: pd.DataFrame, tgt: int, cokus_esik: float):
    """GZ (gerçek-zamanlı) görünüm üzerinden çöküş olayları + ALARM bölümleri."""
    idx = tablo.index
    n = len(idx)

    # --- çöküş olayları: ileri5 ≤ eşik olan günlerin (≤2 boşluk birleşik) koşuları
    cokus_mask = tablo["ileri_5bar_%"] <= cokus_esik      # NaN -> False
    olaylar = []
    for bas, bitis in _seri_bolumler(cokus_mask):
        p_bas, p_bit = idx.get_loc(bas), idx.get_loc(bitis)
        min_ileri = tablo.loc[bas:bitis, "ileri_5bar_%"].min()
        p0 = max(0, p_bas - ONCEDEN_PENCERE)
        p1 = min(n - 1, p_bit + tgt)
        bulundu = None
        for p in range(p0, p1 + 1):
            if tablo["gz_rejim"].iloc[p] == "ALARM":
                bulundu = p
                break
        if bulundu is None:
            sonuc = "KAÇIRDI"
        else:
            k = bulundu - p_bas
            # fiyat radarı sondan-5-bar'lıdır: 5 barlık düşüş birikmeden çalamaz;
            # sigma radarı da +tgt gecikmeli. Olay bitişine kadar çalan alarm = SÜRERKEN.
            efektif_bit = p_bit + tgt if p_bit >= n - 6 else p_bit
            if k < 0:
                sonuc = f"ÖNCEDEN ({-k} bar)"
            elif k <= 2:
                sonuc = f"BAŞLANGICINDA (+{k} bar)"
            elif k <= efektif_bit - p_bas:
                sonuc = f"SÜRERKEN (+{k} bar)"
            else:
                sonuc = f"GEÇ (+{k} bar)"
        olaylar.append({
            "tip": "OLAY",
            "baslangic": bas, "bitis": bitis,
            "zirve_alarm_%": np.nan,
            "en_kotu_ileri5_%": min_ileri,
            "radar_alarm": idx[bulundu] if bulundu is not None else pd.NaT,
            "onset_gunu_gz": tablo["gz_rejim"].iloc[p_bas],
            "sonuc": sonuc,
            "veri_sonu": p_bit >= n - 6,
        })

    # --- ALARM bölümleri: GZ rejimi ALARM olan gün koşuları
    bolumler = []
    gz_mask = tablo["gz_rejim"] == "ALARM"
    for bas, bitis in _seri_bolumler(gz_mask):
        p_bas, p_bit = idx.get_loc(bas), idx.get_loc(bitis)
        zirve = tablo.loc[bas:bitis, "gz_alarm_oran_%"].max()
        p1 = min(n - 1, p_bit + tgt)
        ileri = tablo["ileri_5bar_%"].iloc[p_bas:p1 + 1]
        if ileri.notna().any():
            en_kotu = ileri.min()
            durum = "GERÇEK" if en_kotu <= cokus_esik else "YANLIŞ"
        else:
            en_kotu, durum = np.nan, "AÇIK (veri sonu)"
        bolumler.append({
            "tip": "BÖLÜM",
            "baslangic": bas, "bitis": bitis,
            "zirve_alarm_%": zirve,
            "en_kotu_ileri5_%": en_kotu,
            "radar_alarm": bas,
            "onset_gunu_gz": "",
            "sonuc": durum,
            "veri_sonu": p_bit >= n - 6,
        })
    return olaylar, bolumler


def ana():
    p = argparse.ArgumentParser(
        description="LeWM piyasa radarı — kesitsel sigma + fiyat rejimi (gerçek-zamanlı)")
    p.add_argument("--gunluk", default=r"D:\new\yeniden\metastock-gun-csv",
                   help="orijinal günlük CSV dizini")
    p.add_argument("--cikti", default="lewm_cikti", help="çıktı dizini")
    p.add_argument("--model", default=None,
                   help="checkpoint (varsayılan: <cikti>/lewm_model_eniyi.pt)")
    p.add_argument("--son-bar", type=int, default=60,
                   help="konsol tablosunda gösterilecek bar sayısı")
    p.add_argument("--cokus-esik", type=float, default=-5.0,
                   help="çöküş olayı eşiği: piyasa medyan ileri-5-bar %% (varsayılan -5)")
    p.add_argument("--grafsiz", action="store_true", help="PNG üretme")
    a = p.parse_args()

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
    print("LeWM-Finans PİYASA RADARI v2  (sigma + fiyat, gerçek-zamanlı görünüm)")
    print(f"eşikler: ALARM > %{ESIK_ALARM_ORAN:.0f} sigma | < %{ESIK_DUSUS_ALARM:.0f} fiyat;"
          f"  TEDBİR > %{ESIK_TEDBIR_ORAN:.0f} sigma | < %{ESIK_DUSUS_TEDBIR:.0f} fiyat")
    print("=" * 66)

    # 1) Tek geçiş: her hissenin TAM sigma + TAM kapanış serisi -------------
    model, cfg_d = _modeli_yukle(model_yolu, aygit)
    ctx, tgt = int(cfg_d["ctx_uzunluk"]), int(cfg_d["tgt_uzunluk"])
    periyot = cfg_d["periyot"]
    if periyot not in ("gunluk", "haftalik", "aylik"):
        sys.exit(f"[HATA] yalnızca günlük kaynak (checkpoint periyodu: {periyot})")
    vp = VeriPeriyot(csv_gunluk_dizini=str(kaynak), csv_60dk_dizini=str(kaynak))
    hedefler = hisseleri_bul(kaynak)
    print(f"[1/3] {len(hedefler)} hissenin surprise serisi (tek geçiş)...")
    print(f"      sigma bildirim gecikmesi: +{tgt} bar (satır t'nin değeri t+{tgt}'de bilinir)")

    sigma_seriler, kapanis_tam, n_ok = {}, {}, 0
    for i, h in enumerate(hedefler, 1):
        try:
            skor_tablo = surprise_skorlari(model, vp, h, periyot, ctx, tgt,
                                           aygit=aygit)
        except Exception:
            continue
        t = pd.to_datetime(skor_tablo["tarih"])
        sigma_seriler[h] = pd.Series(skor_tablo["sigma_skoru"].to_numpy(), index=t)
        # FİYAT radarı için TAM kapanış serisi (sigma kırpılmaz; önbellekten bedava)
        try:
            kapanis_tam[h] = vp.get(h, periyot)["Close"]
        except Exception:
            kapanis_tam[h] = pd.Series(skor_tablo["kapanis"].to_numpy(), index=t)
        if i % 100 == 0:
            print(f"      taranan: {i}/{len(hedefler)}")
    if not sigma_seriler:
        sys.exit("[HATA] hiç hisse skorlanamadı.")

    sigma = pd.DataFrame(sigma_seriler)        # [tarih, hisse] (tahmin-tarihli)
    kapanis = pd.DataFrame(kapanis_tam)        # [tarih, hisse] TAM takvim
    birlesik = kapanis.index                   # veri sonuna kadar
    print(f"[2/3] {sigma.shape[1]} hisse; sigma {sigma.shape[0]} bar, "
          f"fiyat {len(birlesik)} bar hizalandı")

    # 2) Kesitsel istatistikler + GZ görünüm --------------------------------
    alarm_n = (sigma.abs() > 3).sum(axis=1)
    gecerli_n = sigma.notna().sum(axis=1)
    oran = pd.Series(
        np.where(gecerli_n > 0, alarm_n / gecerli_n * 100, np.nan),
        index=sigma.index)
    # GZ: D gününde bilinen son tarama = D−tgt (pozisyon kaydırması)
    gz_oran = oran.reindex(birlesik).shift(tgt)
    gz_sayi = alarm_n.reindex(birlesik).shift(tgt)
    gz_gecerli = gecerli_n.reindex(birlesik).shift(tgt)

    tablo = pd.DataFrame({
        "alarm_sayisi": alarm_n.reindex(birlesik),
        "gecerli_hisse": gecerli_n.reindex(birlesik),
        "alarm_oran_%": oran.reindex(birlesik),
        "gz_alarm_sayisi": gz_sayi,
        "gz_gecerli_hisse": gz_gecerli,
        "gz_alarm_oran_%": gz_oran,
        "medyan_|sigma|": sigma.abs().median(axis=1).reindex(birlesik),
        "piyasa_1bar_%": (kapanis / kapanis.shift(1) - 1).median(axis=1) * 100,
        "piyasa_5bar_%": (kapanis / kapanis.shift(5) - 1).median(axis=1) * 100,
        "yukselen_5bar_%": ((kapanis / kapanis.shift(5) - 1) > 0).mean(axis=1) * 100,
        # YALNIZCA ANALİZ — radar o gün geleceği göremezdi:
        "ileri_5bar_%": (kapanis.shift(-5) / kapanis - 1).median(axis=1) * 100,
    })
    r = [_rejim(o, m) for o, m in
         zip(tablo["alarm_oran_%"], tablo["piyasa_5bar_%"])]
    tablo["rejim"] = [x if np.isfinite(o) else ""
                      for x, o in zip([q[0] for q in r], tablo["alarm_oran_%"])]
    tablo["neden"] = [x if np.isfinite(o) else ""
                      for x, o in zip([q[1] for q in r], tablo["alarm_oran_%"])]
    gz = [_rejim(o, m) for o, m in
          zip(tablo["gz_alarm_oran_%"], tablo["piyasa_5bar_%"])]
    tablo["gz_rejim"] = [q[0] for q in gz]
    tablo["gz_neden"] = [q[1] for q in gz]

    # 3) Yakalama analizi (GZ görünüm) ---------------------------------------
    olaylar, bolumler = _yakalama_analizi(tablo, tgt, a.cokus_esik)

    # 4) Rapor ---------------------------------------------------------------
    son = tablo.iloc[-1]
    son_t = tablo.index[-1]
    sigma_son_t = oran.dropna().index[-1] if oran.notna().any() else None
    print(f"[3/3] fiyat serisi: {birlesik[0].date()} → {son_t.date()} | "
          f"son sigma taraması: {sigma_son_t.date() if sigma_son_t is not None else '—'}")
    print(f"\n--- SON DURUM (gerçek-zamanlı: {son_t.date()} kapanışı itibarıyla BİLİNENLER) ---")
    if np.isfinite(son["gz_alarm_oran_%"]):
        kaynak_p = tablo.index.get_loc(son_t) - tgt
        print(f"   gz ±3σ alarm oranı     : %{son['gz_alarm_oran_%']:.1f} "
              f"({int(son['gz_alarm_sayisi'])}/{int(son['gz_gecerli_hisse'])} hisse)"
              f"   [tarama: {tablo.index[kaynak_p].date()} — +{tgt} bar gecikme]"
              f"   [Gauss beklentisi ~%0.3]")
    else:
        print(f"   gz ±3σ alarm oranı     : — (henüz sigma taraması bilinmiyor)")
    print(f"   piyasa medyan 5-bar    : {son['piyasa_5bar_%']:+.2f}% "
          f"(yükselen %{son['yukselen_5bar_%']:.0f})")
    print(f"   GZ REJİM: {son['gz_rejim']}"
          + (f" — {son['gz_neden']}" if son['gz_neden'] else ""))
    if sigma_son_t is not None:
        bilindi = tablo.index[min(len(tablo) - 1, tablo.index.get_loc(sigma_son_t) + tgt)]
        print(f"   not: son sigma satırı {sigma_son_t.date()} — değeri "
              f"{bilindi.date()} günü belli oldu")

    print(f"\n--- SON {min(a.son_bar, len(tablo))} BAR  "
          f"(σ kolonları TAHMİN-tarihli: değeri +{tgt} barda bilinir;"
          f" GZ = o gün gerçekten bilinen okuma) ---")
    baslik = (f"{'tarih':<12}{'alarm%':>8}{'gz-alarm%':>10}{'med|σ|':>8}"
              f"{'p-1bar%':>9}{'p-5bar%':>9}{'yüks5%':>7}{'ileri5%':>9}"
              f"  {'REJİM':<7}{'GZ-REJİM'}")
    print(baslik)
    for t, satir in tablo.tail(a.son_bar).iterrows():
        ileri = _hücre(satir["ileri_5bar_%"], "+.1f")
        print(f"{str(t.date()):<12}{_hücre(satir['alarm_oran_%'], '.1f'):>8}"
              f"{_hücre(satir['gz_alarm_oran_%'], '.1f'):>10}"
              f"{_hücre(satir['medyan_|sigma|'], '.2f'):>8}"
              f"{satir['piyasa_1bar_%']:>+9.2f}{satir['piyasa_5bar_%']:>+9.2f}"
              f"{satir['yukselen_5bar_%']:>6.0f}{ileri:>9}"
              f"  {satir['rejim']:<7}{satir['gz_rejim']}")

    # --- yakalama tabloları -------------------------------------------------
    print(f"\n--- YAKALAMA ANALİZİ (GZ görünüm; çöküş = ileri5 ≤ {a.cokus_esik:.1f}%;"
          f" radar = GZ ALARM) ---")
    if olaylar:
        print("ÇÖKÜŞ OLAYLARI:")
        print(f"{'başlangıç':<12}{'bitiş':<12}{'en kötü ileri5':>15}"
              f"{'onset günü GZ':>15}{'radar ALARM':>14}   sonuç")
        for o in olaylar:
            yildiz = "*" if o["veri_sonu"] else ""
            radar_t = (o["radar_alarm"].date() if pd.notna(o["radar_alarm"]) else "—")
            ek5 = _hücre(o["en_kotu_ileri5_%"], "+.1f") + "%"
            print(f"{str(o['baslangic'].date()):<12}{str(o['bitis'].date()) + yildiz:<12}"
                  f"{ek5:>15}"
                  f"{o['onset_gunu_gz']:>15}{str(radar_t):>14}   {o['sonuc']}")
        if any(o["veri_sonu"] for o in olaylar):
            print("   * olay veri sonunda — çöküş hâlâ sürüyor olabilir"
                  " (ileri5 yalnızca +5 bara kadar ölçülebilir)")
    else:
        print(f"ÇÖKÜŞ OLAYLARI: yok (seri boyunca ileri5 hiç ≤ {a.cokus_esik:.1f}% olmadı)")
    if bolumler:
        print("ALARM BÖLÜMLERİ (GZ):")
        print(f"{'başlangıç':<12}{'bitiş':<12}{'zirve alarm%':>13}"
              f"{'işlem+ufukta en kötü ileri5':>29}   değer")
        for b in bolumler:
            yildiz = "*" if b["veri_sonu"] else ""
            ek5 = _hücre(b["en_kotu_ileri5_%"], "+.1f") + "%"
            print(f"{str(b['baslangic'].date()):<12}{str(b['bitis'].date()) + yildiz:<12}"
                  f"{_hücre(b['zirve_alarm_%'], '.1f'):>13}"
                  f"{ek5:>29}   {b['sonuc']}")
    else:
        print("ALARM BÖLÜMLERİ: yok")
    if olaylar:
        yak = sum(1 for o in olaylar if o["sonuc"] != "KAÇIRDI")
        onceden = sum(1 for o in olaylar if o["sonuc"].startswith("ÖNCEDEN"))
        surerken = sum(1 for o in olaylar
                       if o["sonuc"].startswith(("SÜRERKEN", "BAŞLANGICINDA")))
        kacti = len(olaylar) - yak
        gercek = sum(1 for b in bolumler if b["sonuc"] == "GERÇEK")
        yanlis = sum(1 for b in bolumler if b["sonuc"] == "YANLIŞ")
        acik = sum(1 for b in bolumler if b["sonuc"].startswith("AÇIK"))
        print(f"   ÖZET: {len(olaylar)} çöküş olayı → {yak} yakalandı "
              f"({onceden} önceden, {surerken} çöküş sürerken, {kacti} kaçtı)")
        print(f"   ALARM bölümleri: {len(bolumler)} ({gercek} gerçek, "
              f"{yanlis} yanlış, {acik} açık)")

    # 5) Çıktılar -------------------------------------------------------------
    yol = cikti / "piyasa_radari.csv"
    tablo.to_csv(yol, encoding="utf-8-sig")
    yak_yol = cikti / "piyasa_radari_yakalama.csv"
    pd.DataFrame(olaylar + bolumler).to_csv(yak_yol, index=False,
                                            encoding="utf-8-sig")
    print(f"\n   not: sigma satırları TAHMİN-tarihlidir — satır t'nin değeri "
          f"t+{tgt} kapanışında bilinir (girdi sızıntısı yok; kayan kalibrasyon")
    print(f"   shift(1) yalnız geçmiş sürprizleri kullanır). GZ kolonları canlıda "
          f"görülecek okumadır. Model ağırlıkları bugünkü checkpoint —")
    print(f"   ağırlık sızıntısı içerir.")
    if not a.grafsiz:
        png = cikti / "piyasa_radari.png"
        _grafik(tablo, png, a.son_bar, bolumler)
        print(f"   grafik: {png.name}")
    print(f"[TAMAM] radar serisi: {yol.resolve()}")
    print(f"[TAMAM] yakalama tablosu: {yak_yol.resolve()}")


if __name__ == "__main__":
    ana()
