# -*- coding: utf-8 -*-
"""
lewm_cli.py
===========
LeWM-Finans komut satırı arayüzü.

Hızlı başlangıç (veri_periyot.py ile AYNI klasörde çalıştırın):

  # 1) Eğitim (günlük, tüm hisseler)
  python lewm_cli.py egit --periyot gunluk --epoch 30

  # 2) Uygulama: sürpriz skoru + embedding + gelecek tahmini
  python lewm_cli.py uygula --hisse THYAO --harita --probe

  # Tek komutta ikisi birden:
  python lewm_cli.py hepsi --hisse THYAO

  # Yarım kalan eğitimi sürdür (parçalı/serin çalıştırma için):
  python lewm_cli.py egit --devam --epoch 12

  # Tüm hisselerde k-NN senaryo taraması (tek referans havuzu):
  python lewm_cli.py uygula --tum --senaryo-tarama

  Not: ortak seçenekler (--epoch, --batch, ...) komuttan ÖNCE de SONRA da
  yazılabilir; her iki sıra da geçerlidir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from lewm_egitim import EgitimYapilandirma, egit
from lewm_model import parametre_sayisi
from lewm_uygula import (_modeli_yukle, gelecek_tahmini, gommeleri_cikar,
                         senaryo_tarama, surprise_skorlari, yon_probe)
from lewm_veri import VeriPeriyot, hisseleri_bul
from lewm_rapor import egitim_grafigi, surprise_grafigi, embedding_haritasi, tarama_raporu


def _utf8():
    """Windows konsolunda Türkçe çıktı güvenliği."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _vp_yap(a) -> VeriPeriyot:
    return VeriPeriyot(csv_60dk_dizini=a.saatlik, csv_gunluk_dizini=a.gunluk)


def _cfg_yap(a) -> EgitimYapilandirma:
    return EgitimYapilandirma(
        gunluk_dizin=a.gunluk, saatlik_dizin=a.saatlik, periyot=a.periyot,
        hisseler=(a.hisseler.split(",") if a.hisseler else None),
        ctx_uzunluk=a.ctx, tgt_uzunluk=a.tgt,
        gomme_boyut=a.emb, derinlik=a.derinlik, kafa=a.kafa,
        patch_uzunluk=a.patch,
        epoch=a.epoch, batch=a.batch, lr=a.lr, lam=a.lam,
        cikti_dizin=a.cikti, seed=a.seed,
        devam=bool(getattr(a, "devam", False)),
    )


def komut_egit(a):
    cfg = _cfg_yap(a)
    print("=" * 64)
    print("LeWM-Finans EĞİTİM")
    print("=" * 64)
    model, gecmis, cikti = egit(cfg)
    png = egitim_grafigi(gecmis, cikti)
    print(f"[GRAFIK] {png.resolve()}")


def komut_uygula(a):
    cikti = Path(a.cikti)
    cikti.mkdir(parents=True, exist_ok=True)   # özel --cikti klasörü yoksa yarat
    model_dosyasi = Path(a.model) if a.model else cikti / "lewm_model_eniyi.pt"
    if not model_dosyasi.exists():
        sys.exit(f"[HATA] Model bulunamadı: {model_dosyasi}\n"
                 f"       Önce çalıştırın: python lewm_cli.py egit")
    aygit = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model, cfg_d = _modeli_yukle(model_dosyasi, aygit)
    vp = _vp_yap(a)
    ctx, tgt = cfg_d["ctx_uzunluk"], cfg_d["tgt_uzunluk"]
    periyot = cfg_d["periyot"]

    print("=" * 64)
    print(f"LeWM-Finans UYGULAMA  (model: {model_dosyasi.name}, aygıt: {aygit})")
    print(f"ctx={ctx} bar, tgt={tgt} bar, periyot={periyot}, "
          f"parametre={parametre_sayisi(model):,}")
    print("=" * 64)

    # --- 1) Sürpriz skoru ----------------------------------------------
    hedefler = (hisseleri_bul(vp.csv_gunluk_dizini if periyot in
                              ("gunluk", "haftalik", "aylik")
                              else vp.csv_60dk_dizini)
                if a.tum else [h.strip().upper() for h in a.hisse.split(",")])
    tarama_satirlari = []
    for h in hedefler:
        try:
            tablo = surprise_skorlari(model, vp, h, periyot, ctx, tgt, aygit=aygit)
        except Exception as e:
            print(f"   [ATLANDI] {h}: {e}")
            continue
        csv_yolu = cikti / f"surprise_{h}.csv"
        tablo.to_csv(csv_yolu, index=False, encoding="utf-8-sig")
        png = surprise_grafigi(tablo, h, cikti)
        son = tablo.dropna(subset=["sigma_skoru"]).tail(1)
        if len(son):
            s = float(son["sigma_skoru"].iloc[0])
            durum = "OLAĞANDIŞI" if abs(s) > 3 else "normal"
            if not a.tum:
                print(f"[SURPRISE] {h}: son sigma={s:+.2f} ({durum}) "
                      f"| skor={float(son['surprise'].iloc[0]):.5f}")
                print(f"           CSV: {csv_yolu.name} | PNG: {png.name}")
            tarama_satirlari.append({"hisse": h, "sigma_skoru": s,
                                     "surprise": float(son["surprise"].iloc[0])})

    if a.tum and tarama_satirlari:
        sirali = (pd.DataFrame(tarama_satirlari)
                  .assign(_abs=lambda d: d["sigma_skoru"].abs())
                  .sort_values("_abs", ascending=False)
                  .drop(columns="_abs"))
        yol = cikti / "surprise_tarama.csv"
        sirali.to_csv(yol, index=False, encoding="utf-8-sig")
        n_alarm = int((sirali["sigma_skoru"].abs() > 3).sum())
        print(f"\n[SURPRISE-TARAMA] {len(sirali)} hisse tarandı, "
              f"{n_alarm} tanesi ±3σ eşiğini aştı.")
        print("En olağandışı 20 (|sigma| sıralı):")
        for _, r in sirali.head(20).iterrows():
            isaret = "!" if abs(r["sigma_skoru"]) > 3 else " "
            print(f"   {isaret} {r['hisse']:<12} sigma={r['sigma_skoru']:+6.2f}  "
                  f"skor={r['surprise']:.4f}")
        print(f"   tam liste: {yol.name}")

    # --- 1.5) Senaryo taraması — k-NN, TEK referans havuzu --------------
    # (hisse başına gelecek_tahmini çağırmak havuzu N kez kurardı; burada
    #  havuz bir kez kurulup tüm hedeflerde paylaşılır)
    if a.senaryo_tarama:
        print(f"\n[SENARYO-TARAMA] {len(hedefler)} hisse | referans havuzu "
              f"{a.senaryo_ref} örnek/hisse — havuz kuruluyor (tek seferlik)...")
        sen_tablo, atlanan = senaryo_tarama(
            model, vp, periyot, ctx, tgt, hedefler,
            referans_pencere=a.senaryo_ref, aygit=aygit)
        yol_sen = cikti / "senaryo_tarama.csv"
        sen_tablo.to_csv(yol_sen, index=False, encoding="utf-8-sig")
        if len(sen_tablo):
            pozitif = int((sen_tablo["medyan_senaryo_getiri_%"] > 0).sum())
            print(f"   {len(sen_tablo)} hisse ölçüldü | medyan senaryo getirisi > 0: "
                  f"{pozitif} | atlanan: {len(atlanan)}")
            print("En olumlu 20 (medyan senaryo getirisi sıralı):")
            for _, r in sen_tablo.head(20).iterrows():
                print(f"   {r['hisse']:<12} medyan={r['medyan_senaryo_getiri_%']:+7.2f}%  "
                      f"p10={r['p10_%']:+7.2f}%  p90={r['p90_%']:+7.2f}%  "
                      f"yükselen %{r['yukselen_orani_%']:>4.0f}")
        else:
            print("   (ölçülebilen hisse yok)")
        for h, hata in atlanan[:5]:
            print(f"   [ATLANDI] {h}: {hata}")
        print(f"   tam liste: {yol_sen.name}")

    # --- 2) Embedding + tarama raporu ----------------------------------
    gomme_son = gommeleri_cikar(model, vp, periyot, ctx_uzunluk=ctx, aygit=aygit)
    rapor = tarama_raporu(gomme_son, model, vp, cfg_d, cikti, aygit)
    print(f"[TARAMA]  {len(gomme_son)} hissenin son-pencere gömmesi -> {rapor.name}")

    if a.harita:
        gomme_map = gommeleri_cikar(model, vp, periyot, ctx_uzunluk=ctx,
                                    hisse_basina_pencere=400, aygit=aygit)
        gomme_map.to_csv(cikti / "embedding_tumu.csv", index=False,
                         encoding="utf-8-sig")
        png = embedding_haritasi(gomme_map, cikti)
        print(f"[HARITA]  {png.name} (PCA)")

    # --- 3) Gelecek tahmini (latent k-NN) ------------------------------
    # (--senaryo-tarama ile bu hisseler zaten tarandı; havuzu yeniden
    #  kurmamak için detaylı tahmin atlanır — sonuçlar aynı CSV'de)
    if a.hisse and not a.senaryo_tarama:
        for h in [x.strip().upper() for x in a.hisse.split(",")][:3]:
            try:
                ozet = gelecek_tahmini(model, vp, h, periyot, ctx, tgt,
                                       aygit=aygit)
            except Exception as e:
                print(f"   [ATLANDI] {h} tahmin: {e}")
                continue
            t = ozet["senaryolar"]
            print(f"\n[TAHMIN] {ozet['hisse']} — bağlam sonu {ozet['baglam_sonu']}, "
                  f"ileri {tgt} bar senaryo analizi (k={len(t)}):")
            print(f"   medyan senaryo getirisi : {ozet['medyan_senaryo_getiri_%']:+.2f}%")
            print(f"   %10-%90 aralık          : {ozet['ceyrekler_%'][0]:+.2f}% .. "
                  f"{ozet['ceyrekler_%'][4]:+.2f}%")
            print(f"   yükselen senaryo oranı  : %{ozet['yukselen_orani_%']:.0f}")
            yaz = cikti / f"tahmin_{h}.csv"
            t.to_csv(yaz, index=False, encoding="utf-8-sig")
            print(f"   senaryo tablosu         : {yaz.name}")

    # --- 4) Yön probe'u -------------------------------------------------
    if a.probe:
        sonuc = yon_probe(model, vp, cfg_d, aygit=aygit)
        fark = sonuc["dogruluk"] - sonuc["ham_dogruluk"]
        if fark >= 0.02:
            yargi = (f"VAR — embedding ham baseline'dan +{fark*100:.1f} puan iyi "
                     f"(model ek bilgi taşıyor)")
        elif fark <= -0.02:
            yargi = (f"YOK — ham baseline {(-fark)*100:.1f} puan daha iyi "
                     f"(model sadece trend tekrarı düzeyinde)")
        else:
            yargi = f"SINIRDA — fark {fark*100:+.1f} puan (~eşit)"
        print(f"\n[PROBE] 8-bar yön tahmini (lojistik probe) — kontrol deneyi:")
        print(f"   embedding gömmesi      : %{sonuc['dogruluk']*100:.1f}")
        print(f"   ham öznitelik baseline : %{sonuc['ham_dogruluk']*100:.1f} "
              f"(getiri+vol+hacim, aynı probe)")
        print(f"   şans tabanı            : %{sonuc['taban_dogruluk']*100:.1f}")
        print(f"   embedding ek bilgisi   : {yargi}")

    print(f"\n[TAMAM] tüm çıktılar: {cikti.resolve()}")


def komut_hepsi(a):
    komut_egit(a)
    print()
    komut_uygula(a)


def _secenek_tanimlari():
    """Ortak (veri/eğitim) seçenekleri — tek kaynak, alt komutlarla paylaşılır."""
    return [
        ("--gunluk", dict(default=r"D:\new\yeniden\metastock-gun-csv",
                          help="günlük CSV dizini")),
        ("--saatlik", dict(default=r"D:\new\yeniden\metastock-60dk-csv",
                           help="60dk CSV dizini")),
        ("--cikti", dict(default="lewm_cikti", help="çıktı klasörü")),
        ("--periyot", dict(default="gunluk",
                           choices=["gunluk", "60dk", "240dk", "haftalik", "aylik"])),
        ("--hisseler", dict(default=None,
                            help="virgüllü liste (boş = tümü); örn. THYAO,ASELS,GARAN")),
        ("--epoch", dict(type=int, default=30)),
        ("--batch", dict(type=int, default=256)),
        ("--lr", dict(type=float, default=3e-4)),
        ("--lam", dict(type=float, default=0.1, help="SIGReg ağırlığı")),
        ("--ctx", dict(type=int, default=64, help="bağlam penceresi (bar)")),
        ("--tgt", dict(type=int, default=8, help="ileri pencere (bar)")),
        ("--emb", dict(type=int, default=256, help="embedding boyutu")),
        ("--derinlik", dict(type=int, default=6, help="Transformer derinliği")),
        ("--kafa", dict(type=int, default=8, help="dikkat kafası")),
        ("--patch", dict(type=int, default=4,
                         help="patch uzunluğu (ctx/tgt'yi tam bölmeli, aksi halde otomatik küçültülür)")),
        ("--seed", dict(type=int, default=42)),
    ]


def _ortak_ekle(p, bastir: bool = False):
    """Ortak seçenekleri çözümleyiciye ekler.

    `bastir=True` (alt komut kopyası) durumunda default SUPPRESS yapılır:
    komuttan ÖNCE verilen değer korunur, yalnızca komuttan SONRA verilen
    değer üzerine yazar. Böylece her iki sıra da çalışır:
        python lewm_cli.py --epoch 12 egit
        python lewm_cli.py egit --epoch 12
    """
    for ad, kw in _secenek_tanimlari():
        if bastir:
            kw = dict(kw)
            kw["default"] = argparse.SUPPRESS
        p.add_argument(ad, **kw)


def cozumleyici_yap() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lewm_cli",
        description="LeWorldModel (arXiv:2603.19312) finansal OHLCV uygulaması",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _ortak_ekle(p)
    p.add_argument("--devam", action="store_true", default=False,
                   help="lewm_cikti'daki checkpoint'ten kaldığı yerden devam et "
                        "(komuttan önce de sonra da yazılabilir)")

    alt = p.add_subparsers(dest="komut", required=True)

    e = alt.add_parser("egit", help="modeli eğit",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _ortak_ekle(e, bastir=True)
    e.add_argument("--devam", action="store_true", default=argparse.SUPPRESS,
                   help="lewm_cikti'daki checkpoint'ten kaldığı yerden devam et "
                        "(her epoch sonunda durdurup serindestebilirsiniz)")
    e.set_defaults(f=komut_egit)

    u = alt.add_parser("uygula", help="sürpriz/embedding/tahmin üret",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _ortak_ekle(u, bastir=True)
    u.add_argument("--model", default=None, help="checkpoint yolu (varsayılan: en iyi)")
    u.add_argument("--hisse", default=None, help="virgüllü hisse listesi")
    u.add_argument("--tum", action="store_true", help="tüm hisseler için sürpriz")
    u.add_argument("--harita", action="store_true", help="PCA embedding haritası")
    u.add_argument("--probe", action="store_true", help="yön probe doğruluğu")
    u.add_argument("--senaryo-tarama", action="store_true",
                   help="tüm/seçili hisseler için k-NN senaryo taraması "
                        "(tek referans havuzu → senaryo_tarama.csv)")
    u.add_argument("--senaryo-ref", type=int, default=500,
                   help="senaryo taraması referans havuzu: hisse başına örnek "
                        "(küçültmek hızlandırır, komşu havuzu daraltır)")
    u.set_defaults(f=komut_uygula)

    h = alt.add_parser("hepsi", help="eğit + uygula",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _ortak_ekle(h, bastir=True)
    h.add_argument("--devam", action="store_true", default=argparse.SUPPRESS)
    h.add_argument("--model", default=None)
    h.add_argument("--hisse", default=None)
    h.add_argument("--tum", action="store_true")
    h.add_argument("--harita", action="store_true")
    h.add_argument("--probe", action="store_true")
    h.add_argument("--senaryo-tarama", action="store_true")
    h.add_argument("--senaryo-ref", type=int, default=500)
    h.set_defaults(f=komut_hepsi)
    return p


def ana():
    _utf8()
    p = cozumleyici_yap()
    a = p.parse_args()
    if a.komut in ("uygula", "hepsi") and not a.hisse and not a.tum:
        a.hisse = None
        a.tum = True   # varsayılan: tüm hisseler
    a.f(a)


if __name__ == "__main__":
    ana()
