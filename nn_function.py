from __future__ import annotations
import qimpy as qp
import numpy as np
import torch


class NNFunction(torch.nn.Module):
    n_in: int  #: Number of input features
    n_out: int  #: Number of output features
    n_hidden: list[int]  #: Number of neurons in each hidden layer

    def __init__(self, n_in: int, n_out: int, n_hidden: list[int]) -> None:
        super().__init__()
        pass
