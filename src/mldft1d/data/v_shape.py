from typing import Dict, Any

import torch
import numpy as np

from qimpy.grid import FieldH, FieldR
from qimpy.math import abs_squared
import mldft1d
from .. import Grid1D


def get(grid1d: Grid1D, *, shape: str, scale: float = 1.0, **kwargs) -> FieldR:
    V = ~FieldH(grid1d.grid, data=_get_map[shape](grid1d, **kwargs))  # type: ignore
    return scale * V


def _get_gauss(grid1d: Grid1D, *, sigma: float) -> torch.Tensor:
    norm_fac = sigma * np.sqrt(2 * np.pi) / grid1d.L  # peak value at 1
    translation_phase = (2j * np.pi) * grid1d.iGz * 0.5  # shift to center of unit cell
    return (-0.5 * (grid1d.Gmag * sigma).square() + translation_phase).exp() * norm_fac


def _get_cosine(grid1d: Grid1D, *, n: int = 1) -> torch.Tensor:
    iGz = grid1d.iGz.to(torch.int)
    return torch.where(abs(iGz) == n, 0.5, 0.0).to(torch.complex128)


def _get_rectangular(
    grid1d: Grid1D, *, sigma: float = 0.0, duty: float = 0.5
) -> torch.Tensor:
    rect_tilde = torch.sinc(grid1d.iGz * duty) * duty
    gauss_tilde = (-0.5 * (grid1d.Gmag * sigma).square()).exp()
    return (rect_tilde * gauss_tilde).to(torch.complex128)


def _get_random(grid1d: Grid1D, *, sigma: float, seed: int) -> torch.Tensor:
    # Determine zero and Nyquist frequency weights / real constraints:
    iGz = grid1d.iGz
    Nz = grid1d.grid.shape[2]
    is_real = torch.logical_or(iGz == 0, 2 * iGz == Nz)
    Gweight = torch.where(is_real, 1.0, 2.0)
    # Create white noise with above constraints:
    Gmag = grid1d.Gmag
    torch.manual_seed(seed)
    Gnoise = torch.randn_like(Gmag, dtype=torch.complex128)
    Gnoise[is_real] = Gnoise[is_real].real.to(torch.complex128)
    Gnoise[iGz == 0] = 0.0  # set average to zero
    # Filter and normalize:
    Gnoise *= (-0.5 * (Gmag * sigma).square()).exp()
    Gnoise *= (1.0 / (abs_squared(Gnoise) * Gweight).sum()).sqrt()
    return Gnoise


def _get_coulomb1d(
    grid1d: Grid1D,
    *,
    a: float = 1.0,
    periodic: bool = True,
    fractional: bool = True,
    ionpos: list = None,
    Zs: list = None
) -> torch.Tensor:
    L = grid1d.L
    if ionpos is None or len(ionpos) == 0:
        ionpos = [0.5] if fractional else [0.5 * L]  # default one atom centered
    if Zs is None or len(Zs) == 0:
        Zs = [1.0]  # default atomic charge
    assert len(Zs) == 1 or len(Zs) == len(ionpos)
    if len(Zs) == 1:
        Zs = [Zs[0] for _ion in ionpos]
    soft_coulomb = mldft1d.hf.SoftCoulomb(a)
    V1 = (
        soft_coulomb.periodic_kernel(grid1d.Gmag)
        if periodic
        else soft_coulomb.truncated_kernel(grid1d.grid.shape[2], grid1d.dz, real=True)[
            None, None, :
        ]
    )
    V = torch.zeros(V1.shape, dtype=torch.complex128)
    for _ionx, _Z in zip(ionpos, Zs):
        trans = _ionx if fractional else _ionx / L
        translation_phase = ((2j * np.pi) * grid1d.iGz * -trans).exp()
        V += V1 * _Z * translation_phase / L
    return V


_get_map: Dict[str, Any] = {
    "gauss": _get_gauss,
    "cosine": _get_cosine,
    "rectangular": _get_rectangular,
    "random": _get_random,
    "coulomb1d": _get_coulomb1d,
}
