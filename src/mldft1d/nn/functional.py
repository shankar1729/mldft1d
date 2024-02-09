from __future__ import annotations
from typing import Optional, Sequence
import os

import torch

from qimpy import rc, log, MPI
from qimpy.io.dict import key_cleanup
from qimpy.mpi import BufferView
from qimpy.grid import Grid, FieldR
from .layer import Layer
from .function import activation_map, Function, Linear
from .. import Grid1D


class Functional(torch.nn.Module):  # type: ignore
    """Machine-learned DFT in 1D."""

    layers: torch.nn.ModuleList  #: List of convolution layers
    attr_names: tuple[str, ...]  #: Names of scalar attributes to use as inputs
    attrs: torch.Tensor  #: Values of scalar attributes used as inputs
    activation: torch.nn.Module  #: Activation between layers and within readout
    activation_weights: torch.nn.ModuleList  #: Weights for inter-layer activation
    readout: Function  #: Final read-out function to compute energy density
    wda: bool  #: If true, readout per-particle energy (WDA), else rank-2 approximation
    grid_bulk: Grid  #: Trivial grid for bulk free energy calculations

    def __init__(
        self,
        comm: MPI.Comm,
        *,
        n_sites: int = 1,
        layers: list[dict],
        attr_names: Sequence[str] = tuple(),
        attrs: Optional[Sequence[float]] = None,
        activation: str = "softplus",
        readout: list[int],
        wda: bool = False,
    ) -> None:
        """Initializes functional with specified sizes (and random parameters)."""
        super().__init__()

        # Setup scalar attributes:
        self.attr_names = tuple(attr_names)
        if attrs is None:
            self.attrs = torch.zeros(len(attr_names), device=rc.device)
        else:
            assert len(attrs) == len(attr_names)
            self.attrs = torch.tensor(attrs, device=rc.device, dtype=torch.double)

        # Setup activation and convolution layers:
        self.layers = torch.nn.ModuleList()
        self.activation_weights = torch.nn.ModuleList()
        self.activation = activation_map[activation.lower()]()
        n_even = n_sites
        n_odd = 0
        for i_layer, layer_params in enumerate(layers):
            merge_and_check_dicts(layer_params, dict(n_in=(n_even, n_odd)))
            layer = Layer(**layer_params)
            self.layers.append(layer)
            n_even, n_odd = layer.n_out

            # Weights to control activation between layers
            if i_layer < len(layers) - 1:  # Does not apply to final layer
                self.activation_weights.append(
                    Linear(n_in=n_even, n_out=(n_even + n_odd), device=rc.device)
                )

        # Setup readout:
        self.wda = wda
        if wda:
            if n_odd:
                raise ValueError("WDA should only have even densities from final layer")
            n_readout = n_sites
        else:
            self.readout_indices, self.n_readout_split = zip(
                pair_unpack_indices(n_even), pair_unpack_indices(n_odd)
            )
            n_readout = sum(self.n_readout_split)
        self.readout = Function(
            n_in=(n_even + len(self.attrs)),
            n_out=n_readout,
            n_hidden=readout,
            activation=activation,
        )

        if comm.size > 1:
            self.bcast_parameters(comm)

        log.info("\n----- Making trivial grid for bulk energy calculations -----")
        self.grid_bulk = Grid1D(L=2.0, dz=0.5, parallel=False).grid

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
            params_in = torch.load(load_file, map_location=rc.device)
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
            activation=self.activation.__class__.__name__.lower(),
            readout=self.readout.n_hidden,
            wda=self.wda,
            state=self.state_dict(),
        )
        if comm.rank == 0:
            torch.save(params, filename)

    def get_energy(self, n: FieldR) -> torch.Tensor:
        x = n  # current operand (several channels of scalar fields)
        # Convolution and activation for all but the last layer:
        for layer, activation_weight in zip(self.layers, self.activation_weights):
            x = layer.compute(x)
            # Nonlinear activation: function of scalars, applied to all
            channels_even = x.data[: layer.n_out[0]]
            x.data = x.data * self.activation(activation_weight(channels_even))
        # Final convolution layer:
        layer = self.layers[-1]
        x = layer.compute(x)
        # Read out:
        n_even, n_odd = layer.n_out
        x_even, x_odd = x.data.split((n_even, n_odd))
        scalars = x_even
        if len(self.attrs):
            attrs = repeat_end(self.attrs, scalars.shape[1:])
            scalars = torch.cat((scalars, attrs), dim=0)
        f = self.readout(scalars)
        if self.wda:
            return (n ^ FieldR(n.grid, data=f)).sum(dim=0)
        else:
            # Rank-2 decomposition: free energy function for each scalar pair product
            e: list[torch.Tensor] = []  # energy densities from even, odd
            for xi, fi_flat, pair_unpack in zip(
                (x_even, x_odd), f.split(self.n_readout_split), self.readout_indices
            ):
                fi = fi_flat[pair_unpack]
                e.append(((fi * xi).sum(dim=1) * xi).sum(dim=0))
            return FieldR(n.grid, data=(e[0] + e[1])).integral()

    def get_energy_bulk(self, n: torch.Tensor) -> torch.Tensor:
        grid = self.grid_bulk
        n_batched = n if (len(n.shape) == 1) else n.swapaxes(0, 1)
        n_field = FieldR(grid, data=repeat_end(n_batched, grid.shape))
        return self.get_energy(n_field) / self.grid_bulk.lattice.volume

    def bcast_parameters(self, comm: MPI.Comm) -> None:
        """Broadcast i.e. synchronize module parameters over `comm`."""
        if comm.size > 1:
            for parameter in self.parameters():
                comm.Bcast(BufferView(parameter.data))

    def allreduce_parameters_grad(self, comm: MPI.Comm) -> None:
        """Sum module parameter gradients over `comm`."""
        if comm.size > 1:
            for i_param, parameter in enumerate(self.parameters()):
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter.data)
                comm.Allreduce(MPI.IN_PLACE, BufferView(parameter.grad))


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


def repeat_end(x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Repeat `x` with dimensions `shape` to a result of shape `x.shape + shape`."""
    return x.view(x.shape + (1,) * len(shape)).tile(shape)


def pair_unpack_indices(N: int) -> tuple[torch.Tensor, int]:
    """Indices to unpack {i, j} from flat dimension, and its length N(N+1)/2."""
    ij = torch.arange(N, device=rc.device)
    i = ij[:, None]
    j = ij[None, :]
    i_min = torch.minimum(i, j)
    i_max = torch.maximum(i, j)
    return (i_max * (i_max + 1)) // 2 + i_min, (N * (N + 1)) // 2
