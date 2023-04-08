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
    layers: torch.nn.ModuleList  #: Batched linear layers
    activation: torch.nn.Module  #: Activation layers

    def __init__(self, n_in: int, n_out: int, n_hidden: list[int]) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        n_nodes = [n_in] + n_hidden + [1]
        self.layers = torch.nn.ModuleList(
            [
                BatchedLinear(n1, n2, n_out, device=qp.rc.device)
                for n1, n2 in zip(n_nodes[:-1], n_nodes[1:])
            ]
        )
        self.activation = torch.nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate NN function."""
        x = x[None]  # Add dimension for batching over outputs
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x).squeeze(dim=1)  # Remove output-batching dimension


class BatchedLinear(torch.nn.Module):
    """Linear layer, batched with independent weights to avoid cross-talk.
    Adapted from torch.nn.Linear to add batching, and"""

    __constants__ = ["n_in", "n_out", "n_batch"]
    n_in: int
    n_out: int
    n_batch: int
    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(self, n_in: int, n_out: int, n_batch: int, **kwargs) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_batch = n_batch
        self.weight = Parameter(torch.empty((n_batch, n_out, n_in), **kwargs))
        self.bias = Parameter(torch.empty((n_batch, n_out), **kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.n_in)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.view(self.bias.shape + (1,) * (len(x.shape) - 2))
        return torch.einsum("boi, bi... -> bo...", self.weight, x) + bias

    def extra_repr(self) -> str:
        return f"n_in={self.n_in}, n_out={self.n_out}, n_batch={self.n_batch}"
