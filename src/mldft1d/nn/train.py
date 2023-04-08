from __future__ import annotations
import sys
import glob
import torch
import numpy as np
import qimpy as qp
from qimpy.utils.dict import key_cleanup
from ..data import Data, random_split, random_batch_split, random_mpi_split, fuse_data
from ..nn import Functional
from mpi4py import MPI
from typing import Sequence, Iterable, Union


class Trainer(torch.nn.Module):  # type: ignore

    comm: MPI.Comm
    functional: Functional
    n_perturbations_train_tot: int  #: Total number of perturbations in training data
    n_perturbations_test_tot: int  #: Total number of perturbations in testing data
    data_train: Sequence[Data]  #: Training data (local to process)
    data_test: Sequence[Data]  #: Testing data (local to process)

    def __init__(
        self,
        comm: MPI.Comm,
        functional: Functional,
        filenames: Sequence[str],
        train_fraction: float,
    ) -> None:
        super().__init__()
        self.comm = comm
        self.functional = functional

        # Split filenames into train and test sets:
        n_train_tot = int(len(filenames) * train_fraction)
        n_test_tot = len(filenames) - n_train_tot
        filenames_train_all, filenames_test_all = random_split(
            filenames, [n_train_tot, n_test_tot], seed=0
        )

        # Split filenames within each set over MPI:
        filenames_train = random_mpi_split(filenames_train_all, comm)
        filenames_test = random_mpi_split(filenames_test_all, comm)

        # Load and fuse training set:
        self.data_train = fuse_data([Data(filename) for filename in filenames_train])

        # Load and fuse test set:
        self.data_test = fuse_data([Data(filename) for filename in filenames_test])

        # Report loaded data:
        def report(name: str, data_set: Sequence[Data]) -> int:
            n_perturbations_local = sum(data.n_perturbations for data in data_set)
            n_perturbations_tot = comm.allreduce(n_perturbations_local)
            qp.log.info(
                f"\n{name} set ({n_perturbations_local} local"
                f" of {n_perturbations_tot} perturbations):"
            )
            for data in data_set:
                qp.log.info(f"  {data}")
            return n_perturbations_tot

        self.n_perturbations_train_tot = report("Training", self.data_train)
        self.n_perturbations_test_tot = report("Testing", self.data_test)

    def forward(self, data: Data) -> torch.Tensor:
        """Compute loss function for one complete perturbation data-set"""
        # Compute energy and gradient (= error in V):
        data.n.data.requires_grad = True
        data.n.data.grad = None
        # TODO: systematically add known functional pieces (eg. IdealGas for hard rods)
        E = self.functional.get_energy(data.n) + (data.V_minus_mu ^ data.n)
        Verr = torch.autograd.grad(E.sum(), data.n.data, create_graph=True)[0]
        return Verr.square().sum()  # converted to MSE loss over epoch below

    def train_loop(self, optimizer: torch.optim.Optimizer, batch_size: int) -> float:
        """Run training loop and return mean loss (over epoch)."""
        loss_total = 0.0
        n_perturbations = 0
        n_batches = qp.utils.ceildiv(self.n_perturbations_train_tot, batch_size)
        for data_batch in random_batch_split(self.data_train, n_batches):
            qp.rc.comm.Barrier()
            # Step using total gradient over batch:
            optimizer.zero_grad()
            for data in data_batch:
                loss = self(data)
                loss.backward()
                loss_total += loss.item()
                n_perturbations += data.n_perturbations
            self.functional.allreduce_parameters_grad(self.comm)
            optimizer.step()
            self.functional.bcast_parameters(self.comm)
        # Collect total loss statistics:
        if self.comm.size > 1:
            loss_total = self.comm.allreduce(loss_total)
            n_perturbations = self.comm.allreduce(n_perturbations)
        return loss_total / n_perturbations

    def test_loop(self) -> float:
        """Run test loop and return mean loss."""
        loss_total = sum(self(data).item() for data in self.data_test)
        n_perturbations = sum(data.n_perturbations for data in self.data_test)
        # Collect total loss statistics:
        if self.comm.size > 1:
            loss_total = self.comm.allreduce(loss_total)
            n_perturbations = self.comm.allreduce(n_perturbations)
        return loss_total / n_perturbations


def load_data(
    comm: MPI.Comm,
    functional: Functional,
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
    return Trainer(comm, functional, filenames_expanded, train_fraction)


def get_optimizer(
    params: Iterable[torch.Tensor], method: str, **method_kwargs
) -> torch.optim.Optimizer:
    qp.log.info(f"\nCreating {method} optimizer with {method_kwargs}")
    Optimizer = getattr(torch.optim, method)  # select optimizer class
    return Optimizer(params, **key_cleanup(method_kwargs))


def run_training_loop(
    trainer: Trainer,
    *,
    loss_curve: str,
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
    loss_history = np.zeros((epochs, 2))
    qp.log.info(f"Initial TestLoss: {best_loss_test:>7f}  t[s]: {qp.rc.clock():.1f}")
    for epoch in range(1, epochs + 1):
        loss_train = trainer.train_loop(optimizer, batch_size)
        loss_test = trainer.test_loop()
        loss_history[epoch - 1] = (loss_train, loss_test)
        qp.log.info(
            f"Epoch: {epoch:3d}  TrainLoss: {loss_train:>7f}"
            f"  TestLoss: {loss_test:>7f}  t[s]: {qp.rc.clock():.1f}"
        )
        if epoch % save_interval == 0:
            np.savetxt(loss_curve, loss_history[:epoch], header="TrainLoss TestLoss")
            if loss_test < best_loss_test:
                best_loss_test = loss_test
                trainer.functional.save(save_file, trainer.comm)
                qp.log.info(f"Saved parameters to '{save_file}'")
            else:
                qp.log.info(f"Skipped save because TestLoss >= {best_loss_test} (best)")
    qp.log.info("Done!")


def run(
    comm: MPI.Comm,
    *,
    functional: dict,
    data: dict,
    train: dict,
) -> None:
    torch.random.manual_seed(0)
    trainer = load_data(
        comm,
        Functional.load(comm, **key_cleanup(functional), cache_w=False),
        **key_cleanup(data),
    )
    run_training_loop(trainer, **key_cleanup(train))


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m hardrods1d.trainer <input.yaml>")
        exit(1)
    in_file = sys.argv[1]

    qp.utils.log_config()  # default set up to log from MPI head alone
    qp.log.info("Using QimPy " + qp.__version__)
    qp.rc.init()

    input_dict = key_cleanup(qp.utils.yaml.load(in_file))
    run(qp.rc.comm, **input_dict)


if __name__ == "__main__":
    main()