import os.path
import sys
import torch
import qimpy as qp
import hardrods1d as hr
from typing import Sequence


class Trainer(torch.nn.Module):  # type: ignore

    functional: hr.mlcdft.Functional
    data_train: Sequence[hr.mlcdft.Data]  #: Training data
    data_test: Sequence[hr.mlcdft.Data]  #: Testing data

    def __init__(
        self,
        functional: hr.mlcdft.Functional,
        filenames: Sequence[str],
        train_fraction: float = 0.8,
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

        # Fuse and report  train and test sets:
        qp.log.info("\nTraining set:")
        self.data_train = hr.mlcdft.fuse_data(self.data_train)

        qp.log.info("\nTest set:")
        self.data_test = hr.mlcdft.fuse_data(self.data_test)

    def forward(self, data: hr.mlcdft.Data) -> torch.Tensor:
        """Compute loss function for one complete perturbation data-set"""
        # Set mu
        mu = self.functional.get_mu(data.n_bulk, create_graph=True)
        V_minus_mu = qp.grid.FieldR(data.V.grid, data=(data.V.data - mu))

        # Compute energy and gradient (= error in V):
        data.n.data.requires_grad = True
        data.n.data.grad = None
        E = self.functional.get_energy(data.n, V_minus_mu)
        Verr = torch.autograd.grad(
            E.sum_tensor().sum(), data.n.data, create_graph=True
        )[0]

        # Compute loss from error in V:
        return Verr.square().sum()

    def train_loop(self, optimizer) -> float:
        """Run training loop and return mean loss (over epoch)."""
        loss_total = 0.0
        n_perturbations = 0
        for data in self.data_train:
            # Collect total loss and gradient over batch
            optimizer.zero_grad()
            loss = self(data)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            n_perturbations += data.n_perturbations
        return loss_total / n_perturbations

    def test_loop(self) -> float:
        """Run test loop and return mean loss."""
        loss_total = sum(self(data).item() for data in self.data_test)
        n_perturbations = sum(data.n_perturbations for data in self.data_test)
        return loss_total / n_perturbations


def main() -> None:
    qp.utils.log_config()  # default set up to log from MPI head alone
    qp.log.info("Using QimPy " + qp.__version__)
    qp.rc.init()
    torch.random.manual_seed(0)

    # TODO: get all parameters from YAML input

    # Initialize functional:
    functional = hr.mlcdft.Functional(
        T=1.0,
        w=hr.mlcdft.NNFunction(1, 2, [10, 10]),
        f_ex=hr.mlcdft.NNFunction(2, 2, [10, 10]),
    )
    params_filename = "mlcdft_params.dat"
    if os.path.isfile(params_filename):
        functional.load_state_dict(
            torch.load(params_filename, map_location=qp.rc.device)
        )

    # Initialize trainer:
    filenames = sys.argv[1:]
    assert len(filenames)
    trainer = Trainer(functional, filenames)
    optimizer = torch.optim.SGD(trainer.functional.parameters(), lr=1e-1)

    # Train:
    epochs = 300
    qp.log.info(f"\nTraining for {epochs} epochs")
    for t in range(epochs):
        loss_train = trainer.train_loop(optimizer)
        loss_test = trainer.test_loop()
        qp.log.info(
            f"Epoch: {t + 1:3d}  TrainLoss: {loss_train:>7f}"
            f"  TestLoss: {loss_test:>7f}  t[s]: {qp.rc.clock():.1f}"
        )
    qp.log.info("Done!")
    torch.save(trainer.functional.state_dict(), params_filename)


if __name__ == "__main__":
    main()
