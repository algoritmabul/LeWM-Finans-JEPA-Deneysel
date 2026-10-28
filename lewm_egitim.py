# -*- coding: utf-8 -*-
"""
lewm_egitim.py
==============
LeWM eğitimi — makaledeki iki terimli kayıp:
    L = MSE(predictor(z_ctx), z_tgt)  +  lam * SIGReg(z_ctx, z_tgt)
EMA yok, stop-gradient yok: encoder + predictor uçtan uca eğitilir.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lewm_model import LeWMDunyaModeli, parametre_sayisi
from lewm_sigreg import sigreg_kaybi
from lewm_veri import veri_seti_olustur


def _patch_duzelt(cfg: EgitimYapilandirma) -> int:
    """patch_uzunluk, ctx ve tgt pencelerini TAM bölmeli; aksi halde hedef
    pencere 0 patch üretir -> NaN. En güvenli: gcd ile otomatik küçült."""
    gcd = math.gcd(cfg.ctx_uzunluk, cfg.tgt_uzunluk)
    patch = min(cfg.patch_uzunluk, gcd)
    if cfg.tgt_uzunluk % patch != 0 or cfg.ctx_uzunluk % patch != 0:
        patch = gcd
    return max(1, patch)


@dataclass
class EgitimYapilandirma:
    # veri
    gunluk_dizin: str = r"D:\new\yeniden\metastock-gun-csv"
    saatlik_dizin: str = r"D:\new\yeniden\metastock-60dk-csv"
    periyot: str = "gunluk"                 # gunluk | 60dk | 240dk | haftalik | aylik
    hisseler: Optional[List[str]] = None    # None = klasördeki tümü
    ctx_uzunluk: int = 64                   # bağlam penceresi (bar)
    tgt_uzunluk: int = 8                    # tahmin edilecek ileri pencere (bar)
    dogrulama_oran: float = 0.10
    # model
    patch_uzunluk: int = 4
    gomme_boyut: int = 256
    derinlik: int = 6
    kafa: int = 8
    dropout: float = 0.0
    # eğitim
    epoch: int = 30
    batch: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.04
    warmup: int = 300                       # ısınma adımı (batch)
    lam: float = 0.1                        # SIGReg ağırlığı — makalenin TEK hiperparametresi
    sketch_dim: int = 64                    # SIGReg projeksiyon sayısı
    grad_kirpma: float = 1.0
    seed: int = 42
    num_workers: int = 0                    # Windows uyumluluğu için 0 önerilir
    devam: bool = False                     # checkpoint'ten kaldığı yerden devam
    # çıktı
    cikti_dizin: str = "lewm_cikti"


def _cizelge(adim: int, toplam_adim: int, warmup: int, lr: float):
    """Warmup + cosine çizelge (adım başına çarpan)."""
    if adim < warmup:
        return lr * (adim + 1) / max(1, warmup)
    ilerleme = (adim - warmup) / max(1, toplam_adim - warmup)
    return lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, ilerleme)))


def _mimari_kontrol(kayit_cfg: dict, cfg: EgitimYapilandirma):
    """--devam'da checkpoint mimarisi geçerli yapılandırmayla uyuşmalı."""
    anahtarlar = ("gomme_boyut", "derinlik", "kafa", "ctx_uzunluk",
                  "tgt_uzunluk", "patch_uzunluk")
    farkli = [k for k in anahtarlar
              if k in kayit_cfg and kayit_cfg[k] != getattr(cfg, k, None)]
    if farkli:
        detay = ", ".join(f"{k}: kayıt={kayit_cfg[k]} / şimdi={getattr(cfg, k)}"
                          for k in farkli)
        raise SystemExit(
            f"[HATA] --devam: checkpoint mimarisi uyuşmuyor ({detay}).\n"
            "       Aynı --emb/--derinlik/--kafa/--ctx/--tgt değerlerini kullanın.")


def egit(cfg: EgitimYapilandirma, sustur: bool = False):
    """Modeli eğitir; (model, gecmis, cikti_dizini) döner."""
    rastgele = torch.Generator().manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    aygit = "cuda" if torch.cuda.is_available() else "cpu"
    if not sustur:
        print(f"[AYGIT] {aygit.upper()}"
              + (f" — {torch.cuda.get_device_name(0)}" if aygit == "cuda" else ""))

    # VeriPeriyot her zaman kendi dizinlerinde okur
    from veri_periyot import VeriPeriyot
    vp = VeriPeriyot(csv_60dk_dizini=cfg.saatlik_dizin,
                     csv_gunluk_dizini=cfg.gunluk_dizin)

    egitim_seti, dogrulama_seti = veri_seti_olustur(vp, cfg)
    if not sustur:
        print(f"[VERI] eğitim penceresi: {len(egitim_seti):,} | "
              f"doğrulama penceresi: {len(dogrulama_seti):,} | "
              f"hisse: {len(egitim_seti.hisse_adlari)}")

    cfg.patch_uzunluk = _patch_duzelt(cfg)
    if not sustur:
        print(f"[MODEL] patch={cfg.patch_uzunluk} bar "
              f"(ctx {cfg.ctx_uzunluk}->{cfg.ctx_uzunluk // cfg.patch_uzunluk} patch, "
              f"tgt {cfg.tgt_uzunluk}->{cfg.tgt_uzunluk // cfg.patch_uzunluk} patch)")

    egitim_yukleyici = DataLoader(egitim_seti, batch_size=cfg.batch, shuffle=True,
                                  drop_last=True, num_workers=cfg.num_workers,
                                  generator=rastgele)
    dogrulama_yukleyici = DataLoader(dogrulama_seti, batch_size=cfg.batch,
                                     shuffle=False, num_workers=cfg.num_workers)

    model = LeWMDunyaModeli(
        ozellik_sayisi=6, ctx_uzunluk=cfg.ctx_uzunluk, tgt_uzunluk=cfg.tgt_uzunluk,
        patch_uzunluk=cfg.patch_uzunluk, gomme_boyut=cfg.gomme_boyut,
        derinlik=cfg.derinlik, kafa=cfg.kafa, dropout=cfg.dropout,
    ).to(aygit)
    if not sustur:
        print(f"[MODEL] {parametre_sayisi(model):,} eğitilebilir parametre | "
              f"z-boyutu: {cfg.gomme_boyut}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(aygit, enabled=(aygit == "cuda"))

    toplam_adim = cfg.epoch * max(1, len(egitim_yukleyici))
    adim = 0
    gecmis = {"epoch": [], "kayip": [], "kayip_tahmin": [], "kayip_sigreg": [],
              "dogrulama_kayip": [], "lr": []}
    en_iyi_dogrulama = float("inf")
    baslangic_ep = 0
    yuklendi = False
    cikti = Path(cfg.cikti_dizin)
    cikti.mkdir(parents=True, exist_ok=True)

    # --- kaldığı yerden devam (--devam) ---------------------------------
    if cfg.devam:
        son_yol = cikti / "lewm_model_son.pt"
        eniyi_yol = cikti / "lewm_model_eniyi.pt"
        yuklenecek = (son_yol if son_yol.exists()
                      else (eniyi_yol if eniyi_yol.exists() else None))
        if yuklenecek is None:
            if not sustur:
                print("[DEVAM] checkpoint bulunamadı; sıfırdan eğitim başlıyor.")
        else:
            kontrol = torch.load(yuklenecek, map_location=aygit, weights_only=False)
            _mimari_kontrol(kontrol.get("yapilandirma", {}), cfg)
            model.encoder.load_state_dict(kontrol["encoder_durum"])
            model.predictor.load_state_dict(kontrol["predictor_durum"])
            if "opt_durum" in kontrol:
                opt.load_state_dict(kontrol["opt_durum"])
            if "scaler_durum" in kontrol:
                scaler.load_state_dict(kontrol["scaler_durum"])
            adim = int(kontrol.get("adim", 0))
            # toplam_adim GERİ YÜKLENMEZ: cosine çizelgesi her zaman geçerli
            # --epoch planını izler; böylece daha büyük --epoch ile devam
            # edildiğinde LR sıfıra çakılmaz (düzeltilmiş: CLI-fix-2).
            en_iyi_dogrulama = float(kontrol.get("en_iyi_dogrulama", float("inf")))
            gecmis = dict(kontrol.get("gecmis") or gecmis)
            baslangic_ep = int(kontrol.get("epoch", 0))
            yuklendi = True

    if baslangic_ep >= cfg.epoch:
        if not sustur:
            print(f"[DEVAM] checkpoint'te {baslangic_ep} epoch kayıtlı; "
                  f"--epoch {cfg.epoch} hedefi zaten tamamlanmış. "
                  f"Daha uzun eğitim için --epoch değerini artırın.")
        return model, gecmis, cikti

    if not sustur and yuklendi:
        if baslangic_ep > 0:
            print(f"[DEVAM] checkpoint yüklendi — {baslangic_ep} epoch tamamlanmış, "
                  f"{baslangic_ep + 1}. epoch'tan devam.")
        else:
            print("[DEVAM] checkpoint'ten ağırlıklar yüklendi; "
                  "1. epoch'tan başlanıyor ( optimizer taze başlar ).")

    t0 = time.time()

    for ep in range(baslangic_ep + 1, cfg.epoch + 1):
        model.train()
        donem = {"kayip": 0.0, "tahmin": 0.0, "sigreg": 0.0, "n": 0}
        for ctx_x, tgt_x in egitim_yukleyici:
            ctx_x, tgt_x = ctx_x.to(aygit, non_blocking=True), tgt_x.to(aygit, non_blocking=True)
            lr_t = _cizelge(adim, toplam_adim, cfg.warmup, cfg.lr)
            for g in opt.param_groups:
                g["lr"] = lr_t

            with torch.autocast(aygit, enabled=(aygit == "cuda")):
                z_ctx, z_hat, z_tgt = model.tahmin_et(ctx_x, tgt_x)
                kayip_tahmin = F.mse_loss(z_hat, z_tgt)
                kayip_sigreg = 0.5 * (sigreg_kaybi(z_ctx.float(), cfg.sketch_dim)
                                      + sigreg_kaybi(z_tgt.float(), cfg.sketch_dim))
                kayip = kayip_tahmin + cfg.lam * kayip_sigreg

            opt.zero_grad(set_to_none=True)
            scaler.scale(kayip).backward()
            scaler.unscale_(opt)
            if cfg.grad_kirpma > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_kirpma)
            scaler.step(opt)
            scaler.update()
            adim += 1

            bs = ctx_x.shape[0]
            donem["kayip"] += float(kayip.detach()) * bs
            donem["tahmin"] += float(kayip_tahmin.detach()) * bs
            donem["sigreg"] += float(kayip_sigreg.detach()) * bs
            donem["n"] += bs

        # doğrulama
        model.eval()
        d_toplam, d_n = 0.0, 0
        with torch.no_grad():
            for ctx_x, tgt_x in dogrulama_yukleyici:
                ctx_x, tgt_x = ctx_x.to(aygit), tgt_x.to(aygit)
                z_ctx, z_hat, z_tgt = model.tahmin_et(ctx_x, tgt_x)
                d_toplam += float(F.mse_loss(z_hat, z_tgt)) * ctx_x.shape[0]
                d_n += ctx_x.shape[0]
        d_ort = d_toplam / max(1, d_n)

        gecmis["epoch"].append(ep)
        gecmis["kayip"].append(donem["kayip"] / max(1, donem["n"]))
        gecmis["kayip_tahmin"].append(donem["tahmin"] / max(1, donem["n"]))
        gecmis["kayip_sigreg"].append(donem["sigreg"] / max(1, donem["n"]))
        gecmis["dogrulama_kayip"].append(d_ort)
        gecmis["lr"].append(lr_t)

        if not sustur:
            gecen = time.time() - t0
            print(f"[EPOCH {ep:3d}/{cfg.epoch}] kayıp={gecmis['kayip'][-1]:.5f} "
                  f"(tahmin {gecmis['kayip_tahmin'][-1]:.5f} | sigreg {gecmis['kayip_sigreg'][-1]:.5f}) "
                  f"| doğrulama MSE={d_ort:.5f} | lr={lr_t:.2e} | {gecen:.0f}s")

        # en iyi modeli sakla (tam --devam durumu ile)
        if d_ort < en_iyi_dogrulama:
            en_iyi_dogrulama = d_ort
            _kaydet(model, cfg, cikti, en_iyi=True, opt=opt, scaler=scaler,
                    adim=adim, epoch=ep, en_iyi_dogrulama=en_iyi_dogrulama,
                    gecmis=gecmis, toplam_adim=toplam_adim)
    # son modeli de sakla
    _kaydet(model, cfg, cikti, en_iyi=False, opt=opt, scaler=scaler,
            adim=adim, epoch=gecmis["epoch"][-1] if gecmis["epoch"] else baslangic_ep,
            en_iyi_dogrulama=en_iyi_dogrulama, gecmis=gecmis,
            toplam_adim=toplam_adim)

    if not sustur:
        print(f"[TAMAM] en iyi doğrulama MSE: {en_iyi_dogrulama:.5f} | "
              f"çıktılar: {cikti.resolve()}")
    return model, gecmis, cikti


def _kaydet(model: LeWMDunyaModeli, cfg: EgitimYapilandirma,
            cikti: Path, en_iyi: bool, opt=None, scaler=None, adim: int = 0,
            epoch: int = 0, en_iyi_dogrulama: float = float("inf"),
            gecmis: Optional[dict] = None, toplam_adim: int = 0):
    """Modeli (+ --devam için eğitim durumu) diske yazar."""
    ad = "lewm_model_eniyi.pt" if en_iyi else "lewm_model_son.pt"
    paket = {
        "encoder_durum": model.encoder.state_dict(),
        "predictor_durum": model.predictor.state_dict(),
        "yapilandirma": asdict(cfg),
        "adim": adim,
        "toplam_adim": toplam_adim,
        "epoch": epoch,
        "en_iyi_dogrulama": en_iyi_dogrulama,
        "gecmis": gecmis if gecmis is not None else {},
    }
    if opt is not None:
        paket["opt_durum"] = opt.state_dict()
    if scaler is not None:
        paket["scaler_durum"] = scaler.state_dict()
    torch.save(paket, cikti / ad)


if __name__ == "__main__":  # pragma: no cover
    cfg = EgitimYapilandirma()
    egit(cfg)
