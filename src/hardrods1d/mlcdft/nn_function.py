from __future__ import annotations
import qimpy as qp
import torch
from typing import TypeVar


NNInput = TypeVar("NNInput", torch.Tensor, qp.grid.FieldR)


class NNFunction(torch.nn.Module):
    n_in: int  #: Number of input features
    n_out: int  #: Number of output features
    n_hidden: list[int]  #: Number of neurons in each hidden layer

    def __init__(self, n_in: int, n_out: int, n_hidden: list[int]) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        n_nodes = [n_in] + n_hidden + [n_out]
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(n1, n2) for n1, n2 in zip(n_nodes[:-1], n_nodes[1:])]
        )
        self.activation = torch.nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate NN function."""
        # Prepare inputs:
        batch_shape = x.shape[1:]
        if batch_shape:
            x = x.flatten(1).T  # combine and move batch dimensions to front
        # Apply fully connected feed-forward neural network:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        # Restore output dimensions:
        if batch_shape:
            x = x.T.unflatten(1, batch_shape)  # restore batch dimensions
        return x

    def __call__(self, x: NNInput) -> NNInput:
        """Evaluate NN function."""
        if isinstance(x, torch.Tensor):
            return self.forward(x)
        else:
            return qp.grid.FieldR(x.grid, data=self.forward(x.data))
