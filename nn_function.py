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
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate NN function."""
        if n_in == 1:
            # Weight function mode:
            R = 1.
            return torch.stack((
                2 * (x * R).cos(),
                (2 * R) * (x * R).sinc()
            ))
        elif n_in == 2:
            # Free energy function mode:
            n0, n1 = x
            return n0 * torch.log(1. - n1)
        else:
            raise NotImplementedError

    def __call__(self, x: NNInput) -> NNInput:
        """Evaluate NN function."""
        if isinstance(x, torch.Tensor):
            return self.forward(x)
        else:
            return qp.grid.FieldR(x.grid, self.forward(x.data))

    def deriv(self, x: NNInput) -> NNInput:
        """Evaluate derivative."""
        # TODO: should we use back prop w.r.t x instead? (avoid Hessian)
        if isinstance(x, torch.Tensor):
            if n_in == 1:
                # Weight function mode:
                R = 1.
                return torch.stack((
                    2 * (x * R).cos(),
                    (2 * R) * (x * R).sinc()
                ))
            elif n_in == 2:
                # Free energy function mode:
                n0, n1 = x
                return n0 * torch.log(1. - n1)
            else:
                raise NotImplementedError
        else:
            return qp.grid.FieldR(x.grid, self.deriv(x.data))


