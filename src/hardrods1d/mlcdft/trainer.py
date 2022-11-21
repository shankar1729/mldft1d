import os.path
import sys
import torch
import qimpy as qp
import numpy as np
import hardrods1d as hr
from typing import Sequence


class Trainer(torch.nn.Module):  # type: ignore

    functional: hr.mlcdft.Functional
    data_train: Sequence[hr.mlcdft.Data]  #: Training data
    data_test: Sequence[hr.mlcdft.Data]  #: Testing data
    batch_sizes: Sequence[int]  #: Number of points in each batch during optimization

    def __init__(
        self,
        functional: hr.mlcdft.Functional,
        filenames: Sequence[str],
        train_fraction: float = 0.8,
        batch_size: int = 10,
    ) -> None:
        super().__init__()
        self.functional = functional

        # Load and check data consistency:
        data = [hr.mlcdft.Data(filename) for filename in filenames]
        T = functional.T
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

        # Initialize batch sizes used during training:
        n_train = len(self.data_train)
        n_batches = n_train // batch_size
        self.batch_sizes = np.full(n_batches, batch_size)
        n_remaining = n_train - n_batches * batch_size
        self.batch_sizes[:n_remaining] += 1
        assert self.batch_sizes.sum() == n_train

        # Report and check consistency of train and test sets:
        qp.log.info("\nTraining set:")
        for data_i in self.data_train:
            qp.log.info(f"  {data_i}")

        qp.log.info("\nTest set:")
        for data_i in self.data_test:
            qp.log.info(f"  {data_i}")

    def forward(self, data: hr.mlcdft.Data) -> torch.Tensor:
        """Compute loss function for one complete perturbation data-set"""
        # Set mu
        mu = self.functional.get_mu(data.n_bulk)

        # Compute energy and gradient (= error in V):
        data.n.data.requires_grad = True
        data.n.data.grad = None
        E = self.functional.get_energy(data.n, data.V - mu)
        Verr = torch.autograd.grad(
            E.sum_tensor().sum(), data.n.data, create_graph=True
        )[0]

        # Compute loss from error in V:
        return Verr.square().sum()

    def train_loop(self, optimizer) -> None:
        """Training loop."""
        # Split data into batches:
        data_batches = torch.utils.data.random_split(self.data_train, self.batch_sizes)

        # Step optimizer once per batch:
        loop_loss_total = 0.0
        for i_batch, data_batch in enumerate(data_batches):
            # Collect total loss and gradient over batch
            loss_total = 0.0
            optimizer.zero_grad()
            for data in data_batch:
                loss = self(data)
                loss.backward()
                loss_total += loss.item()
            optimizer.step()
            loop_loss_total += loss_total

            # Report mean loss in batch:
            loss_mean = loss_total / len(data_batch)
            qp.log.info(f"  Batch: {i_batch}  Mean loss: {loss_mean:>7f}")

        # Report mean loss in loop:
        loop_loss_mean = loop_loss_total / len(self.data_train)
        qp.log.info(f"  Train Mean loss: {loop_loss_mean:>7f}")

    def test_loop(self) -> None:
        """Test loop."""
        losses = np.array([self(data).item() for data in self.data_test])
        loss_mean = losses.mean()
        qp.log.info(f"  Test Mean loss: {loss_mean:>7f}")


def main() -> None:
    qp.utils.log_config()  # default set up to log from MPI head alone
    qp.log.info("Using QimPy " + qp.__version__)
    qp.rc.init()

    # TODO: get all parameters from YAML input

    # Initialize functional:
    functional = hr.mlcdft.Functional(
        T=1.0,
        w=hr.mlcdft.NNFunction(1, 2, [30, 30, 30]),
        f_ex=hr.mlcdft.NNFunction(2, 2, [30, 30, 30]),
    )
    params_filename = "mlcdft_params.dat"
    if os.path.isfile(params_filename):
        functional.load_state_dict(
            torch.load(params_filename, map_location=qp.rc.device)
        )

    # Initialize trainer:
    filenames = sys.argv[1:]
    assert len(filenames)
    torch.random.manual_seed(0)
    trainer = Trainer(functional, filenames, batch_size=8)
    optimizer = torch.optim.SGD(trainer.functional.parameters(), lr=3e-6)

    epochs = 100
    for t in range(epochs):
        qp.log.info(f"\n------------- Epoch {t + 1} -------------")
        trainer.train_loop(optimizer)
        trainer.test_loop()
    qp.log.info("Done!")
    torch.save(trainer.functional.state_dict(), params_filename)


if __name__ == "__main__":
    main()
