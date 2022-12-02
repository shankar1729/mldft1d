import torch
import numpy as np
import qimpy as qp
from .grid1d import Grid1D
from typing import Dict, Any


def get(grid1d: Grid1D, *, shape: str, **kwargs) -> qp.grid.FieldR:
    return ~qp.grid.FieldH(grid1d.grid, data=_get_map[shape](grid1d, **kwargs))  # type: ignore


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


_get_map: Dict[str, Any] = {
    "gauss": _get_gauss,
    "cosine": _get_cosine,
    "rectangular": _get_rectangular,
}
