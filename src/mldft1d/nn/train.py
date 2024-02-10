from __future__ import annotations
import sys
import glob
import torch
import numpy as np
import qimpy as qp
from qimpy.io.dict import key_cleanup
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
    weight_V_by_n: bool  #: Whether to weight error in V by density n during training

    def __init__(
        self,
        comm: MPI.Comm,
        functional: Functional,
        filenames: Sequence[str],
        train_fraction: float,
        weight_V_by_n: float,
    ) -> None:
        super().__init__()
        self.comm = comm
        self.functional = functional
        self.weight_V_by_n = weight_V_by_n

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
                assert functional.layers[0].n_in[0] == data.n.data.shape[0]
                qp.log.info(f"  {data}")
            return n_perturbations_tot

        self.n_perturbations_train_tot = report("Training", self.data_train)
        self.n_perturbations_test_tot = report("Testing", self.data_test)

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute loss function for one complete perturbation data-set"""
        # Update scalar attributes from current data set:
        for i_attr, attr_name in enumerate(self.functional.attr_names):
            self.functional.attrs[i_attr] = data.attrs[attr_name]
        # Compute energy and gradient errors:
        data.n.data.requires_grad = True
        data.n.data.grad = None
        Eerr = self.functional.get_energy(data.n) - data.E
        V = torch.autograd.grad(Eerr.sum(), data.n.data, create_graph=True)[0]
        Verr = V - data.dE_dn.data * data.grid1d.dz  # error in partial derivative dE/dn
        if self.weight_V_by_n:
            Verr *= data.n.data
        return Eerr.square().sum(), Verr.square().sum()  # converted to MSE loss below

    def train_loop(
        self,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        energy_loss_scale: float = 1.0,
    ) -> tuple[float, float]:
        """Run training loop and return mean losses (over epoch)."""
        lossE_total = 0.0
        lossV_total = 0.0
        n_perturbations = 0
        n_batches = qp.math.ceildiv(self.n_perturbations_train_tot, batch_size)
        for data_batch in random_batch_split(self.data_train, n_batches):
            qp.rc.comm.Barrier()
            # Step using total gradient over batch:
            optimizer.zero_grad(set_to_none=False)
            lossE_batch = None
            lossV_batch = None
            for data in data_batch:
                lossE, lossV = self(data)
                lossE_batch = lossE if (lossE_batch is None) else (lossE_batch + lossE)
                lossV_batch = lossV if (lossV_batch is None) else (lossV_batch + lossV)
                n_perturbations += data.n_perturbations
            (lossE_batch * energy_loss_scale + lossV_batch).backward()
            lossE_total += lossE_batch.item()
            lossV_total += lossV_batch.item()
            self.functional.allreduce_parameters_grad(self.comm)
            optimizer.step()
            self.functional.bcast_parameters(self.comm)
        # Collect total loss statistics:
        if self.comm.size > 1:
            lossE_total = self.comm.allreduce(lossE_total)
            lossV_total = self.comm.allreduce(lossV_total)
            n_perturbations = self.comm.allreduce(n_perturbations)
        return lossE_total / n_perturbations, lossV_total / n_perturbations

    def test_loop(self) -> tuple[float, float]:
        """Run test loop and return mean loss."""
        lossE_total = 0.0
        lossV_total = 0.0
        n_perturbations = 0
        for data in self.data_test:
            lossE, lossV = self(data)
            lossE_total += lossE.item()
            lossV_total += lossV.item()
            n_perturbations += data.n_perturbations
        # Collect total loss statistics:
        if self.comm.size > 1:
            lossE_total = self.comm.allreduce(lossE_total)
            lossV_total = self.comm.allreduce(lossV_total)
            n_perturbations = self.comm.allreduce(n_perturbations)
        return lossE_total / n_perturbations, lossV_total / n_perturbations


def load_data(
    comm: MPI.Comm,
    functional: Functional,
    *,
    filenames: Union[str, Sequence[str]],
    train_fraction: float = 0.8,
    weight_V_by_n: bool = False,
) -> Trainer:
    # Expand list of filenames:
    filenames = [filenames] if isinstance(filenames, str) else filenames
    filenames_expanded: list[str] = sum(
        [glob.glob(filename) for filename in filenames], start=[]
    )
    assert len(filenames_expanded)

    # Create trainer with specified functional and data split:
    return Trainer(comm, functional, filenames_expanded, train_fraction, weight_V_by_n)


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
    energy_loss_scale: float = 1.0,
    **method_kwargs,
) -> None:
    optimizer = get_optimizer(trainer.functional.parameters(), method, **method_kwargs)
    qp.log.info(f"Training for {epochs} epochs")
    lossE_test, lossV_test = trainer.test_loop()
    best_loss_test = lossE_test * energy_loss_scale + lossV_test
    loss_history = np.zeros((epochs, 4))
    qp.log.info(
        f"Initial  TestLoss: E: {lossE_test:>7f}  V: {lossV_test:>7f}"
        f"  t[s]: {qp.rc.clock():.1f}"
    )
    for epoch in range(1, epochs + 1):
        lossE_train, lossV_train = trainer.train_loop(
            optimizer, batch_size, energy_loss_scale=energy_loss_scale
        )
        lossE_test, lossV_test = trainer.test_loop()
        loss_history[epoch - 1] = (lossE_train, lossV_train, lossE_test, lossV_test)
        qp.log.info(
            f"Epoch: {epoch:3d}"
            f"  TrainLoss: E: {lossE_train:>7f}  V: {lossV_train:>7f}"
            f"  TestLoss: E: {lossE_test:>7f}  V: {lossV_test:>7f}"
            f"  t[s]: {qp.rc.clock():.1f}"
        )
        if epoch % save_interval == 0:
            np.savetxt(
                loss_curve,
                loss_history[:epoch],
                header="TrainLossE TrainLossV TestLossE TestLossV",
            )

            # Save if weighted test loss combination decreased:
            loss_test = lossE_test * energy_loss_scale + lossV_test
            if loss_test < best_loss_test:
                best_loss_test = loss_test
                trainer.functional.save(save_file, trainer.comm)
                qp.log.info(f"Saved parameters to '{save_file}'")
            else:
                qp.log.info(
                    f"Skipped save because weighted TestLoss >= {best_loss_test} (best)"
                )

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
        Functional.load(comm, **key_cleanup(functional)),
        **key_cleanup(data),
    )
    run_training_loop(trainer, **key_cleanup(train))


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m hardrods1d.trainer <input.yaml>")
        exit(1)
    in_file = sys.argv[1]

    qp.io.log_config()  # default set up to log from MPI head alone
    qp.log.info("Using QimPy " + qp.__version__)
    qp.rc.init()

    input_dict = key_cleanup(qp.io.yaml.load(in_file))
    run(qp.rc.comm, **input_dict)


if __name__ == "__main__":
    main()
