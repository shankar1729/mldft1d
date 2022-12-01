import torch
import numpy as np
from typing import Dict, Any


def get(z: torch.Tensor, L: float, *, shape: str, **kwargs) -> torch.Tensor:
    return _get_map[shape](z, L, **kwargs)  # type: ignore


def _get_gauss(z: torch.Tensor, L: float, *, sigma: float) -> torch.Tensor:
    return (-0.5 * ((z - 0.5 * L) / sigma).square()).exp()


def _get_cosine(z: torch.Tensor, L: float, *, n: int = 1) -> torch.Tensor:
    return ((2 * n * np.pi / L) * z).cos()


_get_map: Dict[str, Any] = {"gauss": _get_gauss, "cosine": _get_cosine}
