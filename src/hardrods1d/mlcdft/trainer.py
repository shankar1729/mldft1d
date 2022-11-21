import sys
import torch
import qimpy as qp
import hardrods1d as hr
from typing import Sequence


class Trainer(torch.nn.Module):  # type: ignore

    data_train: Sequence[hr.mlcdft.Data]  #: Training data
    data_test: Sequence[hr.mlcdft.Data]  #: Testing data
    functional: hr.mlcdft.Functional

    def __init__(self, filenames: Sequence[str], train_fraction: float = 0.8) -> None:
        super().__init__()

        # Load and check data consistency:
        data = [hr.mlcdft.Data(filename) for filename in filenames]
        T = data[0].T
        R = data[0].R
        for data_i in data:
            assert T == data_i.T
            assert R == data_i.R

        # Split into train and test sets:
        train_count = int(len(data) * train_fraction)
        test_count = len(data) - train_count
        self.data_train, self.data_test = torch.utils.data.random_split(
            data, [train_count, test_count]
        )

        # Report and check consistency of train and test sets:
        qp.log.info("\nTraining set:")
        for data_i in self.data_train:
            qp.log.info(f"  {data_i}")

        qp.log.info("\nTest set:")
        for data_i in self.data_test:
            qp.log.info(f"  {data_i}")

        # Initialize functional:
        self.functional = hr.mlcdft.Functional(
            T=T, w=hr.mlcdft.NNFunction(1, 2, []), f_ex=hr.mlcdft.NNFunction(2, 2, [])
        )

    def forward(self, data: hr.mlcdft.Data) -> torch.Tensor:
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

    def test_loop(self) -> None:
        """Test loop."""


def main() -> None:
    qp.utils.log_config()  # default set up to log from MPI head alone
    qp.log.info("Using QimPy " + qp.__version__)
    qp.rc.init()

    # TODO: get all parameters from YAML input
    filenames = sys.argv[1:]
    assert len(filenames)
    trainer = Trainer(filenames)

    for name, param in trainer.functional.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    exit()
    optimizer = torch.optim.SGD(trainer.functional.w.parameters(), lr=0.1)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        trainer.train_loop(optimizer)
        trainer.test_loop()
    print("Done!")


if __name__ == "__main__":
    main()
