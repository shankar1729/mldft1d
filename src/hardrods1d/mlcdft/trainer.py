from __future__ import annotations
import os.path
import sys
import glob
import torch
import qimpy as qp
import hardrods1d as hr
from typing import Sequence, Union


class Trainer(torch.nn.Module):  # type: ignore

    functional: hr.mlcdft.Functional
    data_train: Sequence[Sequence[hr.mlcdft.Data]]  #: Training data
    data_test: Sequence[hr.mlcdft.Data]  #: Testing data

    def __init__(
        self,
        functional: hr.mlcdft.Functional,
        filenames: Sequence[str],
        train_fraction: float,
        batch_size: int,
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
        data_train, data_test = hr.mlcdft.random_split(data, [train_count, test_count])

        # Split train set into batches, fuse and report:
        n_train = len(data_train)
        n_batches = qp.utils.ceildiv(n_train, batch_size)
        train_order = torch.randperm(n_train)
        self.data_train = []
        qp.log.info("\nTraining set:")
        for i_batch in range(n_batches):
            qp.log.info(f" Batch {i_batch + 1}:")
            batch_start = (i_batch * n_train) // n_batches
            batch_stop = ((i_batch + 1) * n_train) // n_batches
            data_batch = [data_train[i] for i in train_order[batch_start:batch_stop]]
            self.data_train.append(hr.mlcdft.fuse_data(data_batch))

        qp.log.info("\nTest set:")
        self.data_test = hr.mlcdft.fuse_data(data_test)

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
        for data_batch in self.data_train:
            # Step using total gradient over batch:
            optimizer.zero_grad()
            for data in data_batch:
                loss = self(data)
                loss.backward()
                loss_total += loss.item()
                n_perturbations += data.n_perturbations
            optimizer.step()
        return loss_total / n_perturbations

    def test_loop(self) -> float:
        """Run test loop and return mean loss."""
        loss_total = sum(self(data).item() for data in self.data_test)
        n_perturbations = sum(data.n_perturbations for data in self.data_test)
        return loss_total / n_perturbations


def initialize_functional(
    *,
    T: float,
    n_weights: int,
    w_hidden_sizes: list[int],
    f_ex_hidden_sizes: list[int],
    load_params: str = "",
) -> hr.mlcdft.Functional:
    functional = hr.mlcdft.Functional(
        T=T,
        w=hr.mlcdft.NNFunction(1, n_weights, w_hidden_sizes),
        f_ex=hr.mlcdft.NNFunction(n_weights, n_weights, f_ex_hidden_sizes),
    )
    if load_params and os.path.isfile(load_params):
        functional.load_state_dict(torch.load(load_params, map_location=qp.rc.device))
    return functional


def load_data(
    functional: hr.mlcdft.Functional,
    *,
    filenames: Union[str, Sequence[str]],
    train_fraction: float = 0.8,
    batch_size: int = 10,
) -> Trainer:
    # Expand list of filenames:
    filenames = [filenames] if isinstance(filenames, str) else filenames
    filenames_expanded: list[str] = sum(
        [glob.glob(filename) for filename in filenames], start=[]
    )
    assert len(filenames_expanded)

    # Create trainer with specified functional and data split:
    return Trainer(functional, filenames_expanded, train_fraction, batch_size)


def run_training_loop(
    trainer: Trainer,
    *,
    epochs: int,
    learning_rate: float,
    save_params: str,
) -> None:
    optimizer = torch.optim.SGD(trainer.functional.parameters(), lr=learning_rate)
    qp.log.info(f"\nTraining for {epochs} epochs")
    for t in range(epochs):
        loss_train = trainer.train_loop(optimizer)
        loss_test = trainer.test_loop()
        qp.log.info(
            f"Epoch: {t + 1:3d}  TrainLoss: {loss_train:>7f}"
            f"  TestLoss: {loss_test:>7f}  t[s]: {qp.rc.clock():.1f}"
        )
    qp.log.info("Done!")
    torch.save(trainer.functional.state_dict(), save_params)


def run(
    *,
    functional: dict,
    data: dict,
    train: dict,
) -> None:
    torch.random.manual_seed(0)
    trainer = load_data(
        initialize_functional(**qp.utils.dict.key_cleanup(functional)),
        **qp.utils.dict.key_cleanup(data),
    )
    run_training_loop(trainer, **qp.utils.dict.key_cleanup(train))


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m hardrods1d.mlcdft.trainer <input.yaml>")
        exit(1)
    in_file = sys.argv[1]

    qp.utils.log_config()  # default set up to log from MPI head alone
    qp.log.info("Using QimPy " + qp.__version__)
    qp.rc.init()

    input_dict = qp.utils.dict.key_cleanup(qp.utils.yaml.load(in_file))
    run(**input_dict)


if __name__ == "__main__":
    main()
