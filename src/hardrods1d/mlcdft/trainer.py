from __future__ import annotations
import sys
import glob
import torch
import qimpy as qp
import hardrods1d as hr
from typing import Sequence, Iterable, Union


class Trainer(torch.nn.Module):  # type: ignore

    functional: hr.mlcdft.Functional
    data_train: Sequence[hr.mlcdft.Data]  #: Training data
    data_test: Sequence[hr.mlcdft.Data]  #: Testing data

    def __init__(
        self,
        functional: hr.mlcdft.Functional,
        filenames: Sequence[str],
        train_fraction: float,
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

        # Report training set:
        qp.log.info("\nTraining set:")
        for data_train_i in data_train:
            qp.log.info(f"  {data_train_i}")
        self.data_train = data_train

        # Fuse and report test set:
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

    def train_loop(self, optimizer: torch.optim.Optimizer, batch_size: int) -> float:
        """Run training loop and return mean loss (over epoch)."""
        loss_total = 0.0
        n_perturbations = 0
        for data_batch in hr.mlcdft.random_batch_split(self.data_train, batch_size):
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


def load_data(
    functional: hr.mlcdft.Functional,
    *,
    filenames: Union[str, Sequence[str]],
    train_fraction: float = 0.8,
) -> Trainer:
    # Expand list of filenames:
    filenames = [filenames] if isinstance(filenames, str) else filenames
    filenames_expanded: list[str] = sum(
        [glob.glob(filename) for filename in filenames], start=[]
    )
    assert len(filenames_expanded)

    # Create trainer with specified functional and data split:
    return Trainer(functional, filenames_expanded, train_fraction)


def get_optimizer(
    params: Iterable[torch.Tensor], method: str, **method_kwargs
) -> torch.optim.Optimizer:
    qp.log.info(f"\nCreating {method} optimizer with {method_kwargs}")
    Optimizer = getattr(torch.optim, method)  # select optimizer class
    return Optimizer(params, **method_kwargs)


def run_training_loop(
    trainer: Trainer,
    *,
    save_file: str,
    save_interval: int,
    epochs: int,
    batch_size: int,
    method: str,
    **method_kwargs,
) -> None:
    optimizer = get_optimizer(trainer.functional.parameters(), method, **method_kwargs)
    qp.log.info(f"Training for {epochs} epochs")
    best_loss_test = trainer.test_loop()
    qp.log.info(f"Initial TestLoss: {best_loss_test:>7f}  t[s]: {qp.rc.clock():.1f}")
    for epoch in range(1, epochs + 1):
        loss_train = trainer.train_loop(optimizer, batch_size)
        loss_test = trainer.test_loop()
        qp.log.info(
            f"Epoch: {epoch:3d}  TrainLoss: {loss_train:>7f}"
            f"  TestLoss: {loss_test:>7f}  t[s]: {qp.rc.clock():.1f}"
        )
        if epoch % save_interval == 0:
            if loss_test < best_loss_test:
                best_loss_test = loss_test
                trainer.functional.save(save_file)
                qp.log.info(f"Saved parameters to '{save_file}'")
            else:
                qp.log.info(f"Skipped save because TestLoss >= {best_loss_test} (best)")
    qp.log.info("Done!")


def run(
    *,
    functional: dict,
    data: dict,
    train: dict,
) -> None:
    torch.random.manual_seed(0)
    trainer = load_data(
        hr.mlcdft.Functional.load(**qp.utils.dict.key_cleanup(functional)),
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
