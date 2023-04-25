from __future__ import annotations
import qimpy as qp
import math
import torch
from torch.nn.parameter import Parameter


class Function(torch.nn.Module):
    """Neural network approximation to a function with specified inputs and outputs."""

    __constants__ = ["n_in", "n_out", "n_hidden"]
    n_in: int  #: Number of inputs
    n_out: int  #: Number of outputs (independent NN for each output)
    n_hidden: list[int]  #: Number of neurons in each hidden layer
    layers: torch.nn.ModuleList  #: Linear layers
    activation: torch.nn.Module  #: Activation layers

    def __init__(self, n_in: int, n_out: int, n_hidden: list[int]) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        n_nodes = [n_in] + n_hidden + [n_out]
        self.layers = torch.nn.ModuleList(
            [
                Linear(n1, n2, device=qp.rc.device)
                for n1, n2 in zip(n_nodes[:-1], n_nodes[1:])
            ]
        )
        self.activation = torch.nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate NN function."""
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class Linear(torch.nn.Module):
    """Linear layer, adapted from torch.nn.Linear to operate on first dimension."""

    __constants__ = ["n_in", "n_out", "n_batch"]
    n_in: int
    n_out: int
    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(self, n_in: int, n_out: int, **kwargs) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.weight = Parameter(torch.empty((n_out, n_in), **kwargs))
        self.bias = Parameter(torch.empty((n_out,), **kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.n_in)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.view((self.n_out,) + (1,) * (len(x.shape) - 1))
        return torch.tensordot(self.weight, x, dims=1) + bias

    def extra_repr(self) -> str:
        return f"n_in={self.n_in}, n_out={self.n_out}"
