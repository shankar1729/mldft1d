from __future__ import annotations
import qimpy as qp
import torch
import os
from mpi4py import MPI
from typing import Optional, Sequence
from qimpy.utils.dict import key_cleanup
from .layer import Layer


class Functional(torch.nn.Module):  # type: ignore
    """Machine-learned DFT in 1D."""

    layers: torch.nn.ModuleList  #: List of layers
    attr_names: tuple[str, ...]  #: Names of scalar attributes to use as inputs
    attrs: torch.Tensor  #: Values of scalar attributes used as inputs

    def __init__(
        self,
        comm: MPI.Comm,
        *,
        layers: list[dict],
        attr_names: Sequence[str] = tuple(),
        attrs: Optional[Sequence[float]] = None,
    ) -> None:
        """Initializes functional with specified sizes (and random parameters)."""
        super().__init__()

        # Setup scalar attributes:
        self.attr_names = tuple(attr_names)
        if attrs is None:
            self.attrs = torch.zeros(len(attr_names), device=qp.rc.device)
        else:
            assert len(attrs) == len(attr_names)
            self.attrs = torch.tensor(attrs, device=qp.rc.device, dtype=torch.double)

        # Setup layers:
        self.layers = torch.nn.ModuleList()
        n_outputs_prev = 1
        for layer_params in layers:
            merge_and_check_dicts(
                layer_params, dict(n_inputs=n_outputs_prev, n_attrs=len(self.attrs))
            )
            layer = Layer(**layer_params)
            self.layers.append(layer)
            n_outputs_prev = layer.f.n_out
        assert n_outputs_prev == 1

        if comm.size > 1:
            self.bcast_parameters(comm)

    @classmethod
    def load(
        cls,
        comm: MPI.Comm,
        *,
        load_file: str = "",
        layers: Optional[list[dict]] = None,
        **kwargs,
    ) -> Functional:
        """
        Initialize functional from `kwargs` or from params file if given `load_file`.
        Any parameter in `kwargs` must be consistent with the params file, if given.
        """
        params = dict(**kwargs)
        if layers is not None:
            params["layers"] = [key_cleanup(layer) for layer in layers]
        state = {}

        # Merge / check parameters from file, if available:
        if load_file and os.path.isfile(load_file):
            params_in = torch.load(load_file, map_location=qp.rc.device)
            for key, value_in in params_in.items():
                if key == "state":
                    state = value_in
                elif key in params:
                    value = params[key]
                    if key == "layers":
                        assert len(value) == len(value_in)  # sequences of same length
                        for value_i, value_in_i in zip(value, value_in):
                            assert isinstance(value_i, dict)
                            assert isinstance(value_in_i, dict)
                            merge_and_check_dicts(value_i, value_in_i)
                    else:
                        assert value == value_in
                else:
                    params[key] = value_in

        # Create functional and load state if available:
        functional = Functional(comm, **params)
        if state:
            functional.load_state_dict(state)
        return functional

    def save(self, filename: str, comm: MPI.Comm) -> None:
        """Save parameters to specified filename."""
        params = dict(
            layers=[layer.asdict() for layer in self.layers],
            attr_names=self.attr_names,
            state=self.state_dict(),
        )
        if comm.rank == 0:
            torch.save(params, filename)

    def get_energy(self, n: qp.grid.FieldR) -> torch.Tensor:
        channels = n[None]
        for layer in self.layers:
            channels = layer.compute(channels, self.attrs)
        return n ^ channels[0]

    def get_energy_bulk(self, n: torch.Tensor) -> torch.Tensor:
        channels = n[None]
        for layer in self.layers:
            channels = layer.compute_bulk(channels, self.attrs)
        return n * channels[0]

    def bcast_parameters(self, comm: MPI.Comm) -> None:
        """Broadcast i.e. synchronize module parameters over `comm`."""
        if comm.size > 1:
            for parameter in self.parameters():
                comm.Bcast(qp.utils.BufferView(parameter.data))

    def allreduce_parameters_grad(self, comm: MPI.Comm) -> None:
        """Sum module parameter gradients over `comm`."""
        if comm.size > 1:
            for i_param, parameter in enumerate(self.parameters()):
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter.data)
                comm.Allreduce(MPI.IN_PLACE, qp.utils.BufferView(parameter.grad))


def merge_and_check_dicts(d: dict, d_in: dict) -> None:
    """Merge entries from `d_in` into `d`. Any entry already in `d` must match `d_in`"""
    for key, value_in in d_in.items():
        if key in d:
            value = d[key]
            if isinstance(value, dict):
                merge_and_check_dicts(value, value_in)
            else:
                assert value == value_in
        else:
            d[key] = value_in
