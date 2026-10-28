# -*- coding: utf-8 -*-
"""
lewm_sigreg.py
==============
SIGReg — Sketched Isotropic Gaussian Regularization
(Maes vd., "LeWorldModel", arXiv:2603.19312 ve
 Bildirici vd. temelli "LeJEPA", arXiv:2511.08544, Algoritma 1)

Mantık:
  1) D boyutlu gömmeler, her çağrıda TAZE rastgele birim-normlu A matrisi ile
     `sketch_dim` adet 1B doğrultuya projelenir ("sketching").
  2) Her 1B doğrultu için Epps-Pulley istatistiği hesaplanır:
     ampirik karakteristik fonksiyon (ECF) ile standart normal CF'i arasındaki
     ağırlıklı L2 uzaklık.  T_2 = n * integral( |phi_n(t) - phi_0(t)|^2 w(t) dt )
  3) Sketch'lerin ortalaması SIGReg kaybı olarak döner.

Bu düzenleyici temsil çökmesini EMA / stop-gradient OLMADAN önler:
gömme bulutu N(0, I)'a çekildikçe bilgi taşımayan sabit çözümler cezalanır.
"""
from __future__ import annotations

import torch

__all__ = ["epps_pulley_istatistigi", "sigreg_kaybi", "zayif_sigreg_kaybi"]


def epps_pulley_istatistigi(x: torch.Tensor,
                            t_sayisi: int = 17,
                            t_maks: float = 5.0) -> torch.Tensor:
    """Tek değişkenli Epps-Pulley istatistiği (gradyan akar).

    Parametreler
    ------------
    x : [N, S]  — S adet sketch doğrultusuna projelenmiş değerler
    t_sayisi    — entegrasyon ızgara noktası sayısı (LeJEPA: 17)
    t_maks      — ızgara aralığı [-t_maks, t_maks] (LeJEPA: 5.0)

    Döner: [S] — her sketch için istatistik (n ile ölçekli, ~O(1) H0 altında)
    """
    N = x.shape[0]
    t = torch.linspace(-t_maks, t_maks, t_sayisi, device=x.device, dtype=x.dtype)

    # Ampirik CF: phi_n(t) = (1/n) * sum_n exp(i * t * x_n)
    args = x.unsqueeze(2) * t.view(1, 1, -1)          # [N, S, T]
    ecf = torch.exp(1j * args).mean(dim=0)            # [S, T] (complex)

    # Hedef: standart normal CF -> phi_0(t) = exp(-t^2 / 2); ağırlık w(t) = exp(-t^2/2)
    cf_hedef = torch.exp(-0.5 * t ** 2)               # [T]

    fark = (ecf - cf_hedef.view(1, -1)).abs().square()  # |phi_n - phi_0|^2  [S, T]
    agirlikli = fark * cf_hedef.view(1, -1)             # w(t) ile çarp

    # Trapz ile t üzerinde entegre et, örneklem boyu n ile ölçekle
    stat = torch.trapz(agirlikli, t, dim=1) * N         # [S]
    return stat


def sigreg_kaybi(z: torch.Tensor,
                 sketch_dim: int = 64,
                 t_sayisi: int = 17,
                 t_maks: float = 5.0) -> torch.Tensor:
    """Güçlü SIGReg kaybı (LeJEPA Algoritma 1 — ECF tabanlı).

    z : [N, D] gömme yığını (batch)
    Her çağrıda taze rastgele projeksiyon matrisi örneklenir -> düzenleyici
    tek bir doğrultuya aşırı uyum yapamaz.
    """
    N, D = z.shape
    S = min(sketch_dim, D)

    # Karma hassasiyet (AMP) güvenliği: complex-half bazı platformlarda
    # desteklenmez -> SIGReg her zaman float32'de hesaplanır.
    with torch.autocast(device_type=z.device.type, enabled=False):
        z = z.float()
        # Rastgele birim-normlu doğrultular: [D, S]
        A = torch.randn(D, S, device=z.device, dtype=torch.float32)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)

        proj = z @ A                                        # [N, S]
        stat = epps_pulley_istatistigi(proj, t_sayisi, t_maks)  # [S]
    return stat.mean()


def zayif_sigreg_kaybi(z: torch.Tensor,
                       sketch_dim: int = 64) -> torch.Tensor:
    """Zayıf SIGReg (uyarı: LeWM makalesindeki güçlü formun basitleştirmesi).

    Sadece 2. momenti eşler: Cov(z) ~ I (küresel bulut). CPU'da daha ucuzdur,
    ama tüm momentleri eşlemediği için çökme karşıtı garanti zayıflar.
    Varsayılan olarak `sigreg_kaybi` (güçlü) kullanın.
    """
    N, D = z.shape
    S = min(sketch_dim, D)
    if S < D:
        Sc = torch.randn(S, D, device=z.device, dtype=z.dtype) / (D ** 0.5)
        x = z @ Sc.T
    else:
        x = z
    x = x - x.mean(dim=0, keepdim=True)
    kov = (x.T @ x) / (N - 1 + 1e-6)
    return torch.norm(kov - torch.eye(S, device=z.device, dtype=z.dtype), p="fro")
