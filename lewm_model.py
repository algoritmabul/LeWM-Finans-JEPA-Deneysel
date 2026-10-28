# -*- coding: utf-8 -*-
"""
lewm_model.py
=============
LeWorldModel (arXiv:2603.19312) mimarisinin OHLCV finansal zaman serisine
uyarlanması.

Uyarlama haritası (makale -> burası):
  ham piksel karesi   -> bar penceresi [ctx x F]  (F = türetilmiş özellikler)
  ViT patch'i         -> `patch_uzunluk` bar'lık dilim (Lineer + pozisyon gömmesi)
  Encoder             -> pre-LN Transformer; tokenler havuzlanır -> z [D]
  Predictor           -> z_ctx'den gelecek pencerenin gömmesini tahmin eder
  Kayıp               -> MSE(z_hat, z_tgt) + lam * SIGReg(z_ctx, z_tgt)
  EMA / stop-gradient -> YOK (makalenin temel katkısı; uçtan uca eğitim)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LeWMEncoder", "LeWMPredictor", "LeWMDunyaModeli", "parametre_sayisi"]


class BarPatchGomme(nn.Module):
    """[B, L, F] bar dizisini `patch_uzunluk`'luk dilimlere bölüp lineer projeksiyon."""

    def __init__(self, ozellik_sayisi: int, patch_uzunluk: int,
                 gomme_boyut: int, maks_patch: int):
        super().__init__()
        self.patch_uzunluk = patch_uzunluk
        self.proj = nn.Linear(patch_uzunluk * ozellik_sayisi, gomme_boyut)
        self.norm = nn.LayerNorm(gomme_boyut)
        self.pos = nn.Parameter(torch.zeros(1, maks_patch, gomme_boyut))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, F_ = x.shape
        P = L // self.patch_uzunluk
        x = x[:, :P * self.patch_uzunluk, :]
        x = x.reshape(B, P, self.patch_uzunluk * F_)
        x = self.proj(x)
        x = self.norm(x + self.pos[:, :P])
        return x


class DikkatBlok(nn.Module):
    """Pre-LN Transformer bloğu."""

    def __init__(self, boyut: int, kafa: int, mlp_oran: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(boyut)
        self.attn = nn.MultiheadAttention(boyut, kafa, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(boyut)
        gizli = int(boyut * mlp_oran)
        self.mlp = nn.Sequential(
            nn.Linear(boyut, gizli), nn.GELU(), nn.Linear(gizli, boyut),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_cikti, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_cikti
        x = x + self.mlp(self.norm2(x))
        return x


class LeWMEncoder(nn.Module):
    """Bar penceresi -> tokenler -> havuzlanmış latent gömme z [B, D]."""

    def __init__(self, ozellik_sayisi: int = 6, patch_uzunluk: int = 4,
                 gomme_boyut: int = 256, derinlik: int = 6, kafa: int = 8,
                 maks_patch: int = 32, dropout: float = 0.0):
        super().__init__()
        self.patch = BarPatchGomme(ozellik_sayisi, patch_uzunluk, gomme_boyut, maks_patch)
        self.blok = nn.ModuleList([
            DikkatBlok(gomme_boyut, kafa, dropout=dropout) for _ in range(derinlik)
        ])
        self.norm = nn.LayerNorm(gomme_boyut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, F] -> tokenler [B, P, D]"""
        h = self.patch(x)
        for b in self.blok:
            h = b(h)
        return self.norm(h)

    def havuzla(self, tokenler: torch.Tensor) -> torch.Tensor:
        """Tokenler -> tek latent vektör (ortalama havuzlama)."""
        return tokenler.mean(dim=1)

    def kodla(self, x: torch.Tensor) -> torch.Tensor:
        return self.havuzla(self.forward(x))


class LeWMPredictor(nn.Module):
    """z_ctx -> ẑ_tgt. Küçük 2 katmanlı MLP (rezidüel + LayerNorm)."""

    def __init__(self, gomme_boyut: int = 256, gizli_oran: float = 4.0):
        super().__init__()
        gizli = int(gomme_boyut * gizli_oran)
        self.net = nn.Sequential(
            nn.LayerNorm(gomme_boyut),
            nn.Linear(gomme_boyut, gizli), nn.GELU(),
            nn.Linear(gizli, gomme_boyut),
        )

    def forward(self, z_ctx: torch.Tensor) -> torch.Tensor:
        return z_ctx + self.net(z_ctx)   # rezidüel: tahmin, gömme düzleminde küçük düzeltme


class LeWMDunyaModeli(nn.Module):
    """Encoder + Predictor sarmalayıcısı (eğitim ve uygulama için pratik API)."""

    def __init__(self, ozellik_sayisi: int = 6, ctx_uzunluk: int = 64,
                 tgt_uzunluk: int = 8, patch_uzunluk: int = 4,
                 gomme_boyut: int = 256, derinlik: int = 6, kafa: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        if ctx_uzunluk % patch_uzunluk != 0 or tgt_uzunluk % patch_uzunluk != 0:
            raise ValueError(
                f"patch_uzunluk ({patch_uzunluk}) hem ctx ({ctx_uzunluk}) hem "
                f"tgt ({tgt_uzunluk}) penceresini tam bölmeli; aksi halde hedef "
                f"pencere 0 patch üretir ve kayıp NaN olur.")
        maks_patch = max(ctx_uzunluk, tgt_uzunluk) // patch_uzunluk + 1
        self.encoder = LeWMEncoder(ozellik_sayisi, patch_uzunluk, gomme_boyut,
                                   derinlik, kafa, maks_patch, dropout)
        self.predictor = LeWMPredictor(gomme_boyut)

    def tahmin_et(self, ctx_x: torch.Tensor, tgt_x: torch.Tensor):
        """Eğitim çekirdeği: (kayıp bileşenleri için) z_ctx, z_hat, z_tgt döner."""
        z_ctx = self.encoder.kodla(ctx_x)
        z_tgt = self.encoder.kodla(tgt_x)
        z_hat = self.predictor(z_ctx)
        return z_ctx, z_hat, z_tgt


def parametre_sayisi(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
