import torch
from .data import Data
from typing import Sequence


class Trainer(torch.nn.Module):  # type: ignore

    data_train: Sequence[Data]  #: Training data
    data_test: Sequence[Data]  #: Testing data

    def __init__(self, filenames: Sequence[str], train_fraction: float = 0.8) -> None:
        super().__init__()
        # Load Data from each filename and assign to one of data_train or data_test

    def forward(self, data: Data) -> torch.Tensor:
        """Compute loss function for one complete perturbation data-set"""
        # Set mu
        # Compute V for every data.n
        # Return ||V - data.V|| as loss

    def train_loop(self, optimizer) -> None:
        """Training loop."""
        # TODO: operate on batches of data
        for data in self.data_train:
            loss = self(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # loss, current = loss.item(), batch * len(X)
        # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
