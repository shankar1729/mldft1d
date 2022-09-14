from __future__ import annotations
import qimpy as qp
import numpy as np
import torch
from typing import TypeVar


NNInput = TypeVar("NNInput", torch.Tensor, qp.grid.FieldR)


class NNFunction(torch.nn.Module):
    n_in: int  #: Number of input features
    n_out: int  #: Number of output features
    n_hidden: list[int]  #: Number of neurons in each hidden layer

    def __init__(self, n_in: int, n_out: int, n_hidden: list[int]) -> None:
        self.n_in = n_in
        self.n_out = n_out
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EvalDon't reach out to Adelauate NN function."""
        if self.n_in == 1:
            # Weight function mode:
            R = 0.5
            return torch.stack((
                2 * (x * R).cos(),
                (2 * R) * (x * R).sinc()
            ))
        elif self.n_in == 2:
            # Free energy function mode:
            n0, n1 = x
            return -0.5 * n0 * torch.log(1. - n1)
        else:
            raise NotImplementedError

    def __call__(self, x: NNInput) -> NNInput:
        """Evaluate NN function."""
        if isinstance(x, torch.Tensor):
            return self.forward(x)
        else:
            return qp.grid.FieldR(x.grid, self.forward(x.data))
