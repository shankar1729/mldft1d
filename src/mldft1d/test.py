from __future__ import annotations
from typing import Callable, Sequence, Union, Optional
import sys
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import h5py

import qimpy
from qimpy import rc, log, io, Energy
from qimpy.grid import FieldR
from qimpy.profiler import StopWatch
from . import Grid1D, get1D, Minimizer, protocols, hardrods, kohnsham, hf, ising, nn
from .data import v_shape
from .kohnsham import Schrodinger, ThomasFermi


def make_dft_data(
    *,
    grid1d: Grid1D,
    filename: str,
    i_data: int,
    n_bulk: torch.Tensor,
    T: float,
    mu: Union[float, Sequence[float]],
) -> protocols.DFT:
    """Make a DFT that poses a data file as a DFT for comparison.
    Most useful for comparing to MD data."""
    mus = torch.tensor([mu] if isinstance(mu, float) else mu, device=rc.device)
    return DataDFT(grid1d, filename, i_data, T, mus)


make_dft_map: dict[str, Callable[..., protocols.DFT]] = {
    "hardrods_exact": hardrods.make_dft.exact,
    "hardrods_lda": hardrods.make_dft.lda,
    "hardrods_ml": hardrods.make_dft.ml,
    "numerical_lda": hardrods.make_dft.numerical_lda,
    "kohnsham_exact": kohnsham.make_dft.exact,
    "kohnsham_lda": kohnsham.make_dft.lda,
    "kohnsham_ml": kohnsham.make_dft.ml,
    "hf_exact": hf.make_dft.exact,
    "hf_lda": hf.make_dft.lda,
    "hf_ml": hf.make_dft.ml,
    "ising_exact": ising.make_dft.exact,
    "ising_ml": ising.make_dft.ml,
    "data": make_dft_data,
}  #: Recognized DFTs that can be loaded from YAML input


def run(
    *,
    L: float = 0.0,
    dz: float = 0.0,
    n_bulk: Union[float, Sequence[float]],
    Vshape: Union[dict, Sequence[dict]],
    lbda: float,
    functionals: dict,
    run_name: str,
    n_bulk_range: Sequence[tuple[float, float]] = ((0.0, 1.0),),
    data_file: str = "",
    nc=0.0,
    **dft_common_args,
):
    # Check site density counts:
    n_bulks = torch.tensor(
        [n_bulk] if isinstance(n_bulk, float) else n_bulk, device=rc.device
    )
    n_sites = len(n_bulks)

    # Create grid:
    if data_file:
        with h5py.File(data_file) as fp:
            z = np.array(fp["z"])
        dz = z[1] - z[0]
        L = z[-1] - z[0] + dz
    assert dz
    assert L
    grid1d = Grid1D(L=L, dz=dz)

    # Get potential from specified data file or shape:
    if data_file:
        with h5py.File(data_file) as fp:
            Vdata = torch.tensor(fp["V"], device=rc.device)[:, None, None]
    else:
        Vshape = [Vshape] if isinstance(Vshape, dict) else Vshape
        assert len(Vshape) == n_sites
        Vdata = torch.stack(
            [
                v_shape.get(grid1d, **io.dict.key_cleanup(Vshape_i)).data
                for Vshape_i in Vshape
            ]
        )
    V = FieldR(grid1d.grid, data=(lbda * Vdata))

    # Create DFTs:
    dfts = dict[str, protocols.DFT]()
    for label, dft_dict in functionals.items():
        for dft_name, dft_args in io.dict.key_cleanup(dft_dict).items():
            dfts[label] = make_dft_map[dft_name](
                grid1d=grid1d, n_bulk=n_bulks, **dft_args, **dft_common_args
            )

    for dft in dfts.values():
        dft.V = V
        if isinstance(dft, Minimizer):
            dft.finite_difference_test(dft.random_direction())
        dft.minimize()  # equilibrium results in dft.energy and dft.n

    # Report final energies etc.:
    for dft_name, dft in dfts.items():
        E = float(dft.energy)
        if isinstance(dft, hf.DFT) and (n_bulk * L > 1):
            homo = dft.eig[dft.eig < dft.mu].max().item()
            lumo = dft.eig[dft.eig > dft.mu].min().item()
            gap = lumo - homo
        else:
            gap = np.nan
        log.info(f"{dft_name:>14s}:  mu: {io.fmt(dft.mu)}  E: {E:>9f}  gap: {gap:>9f}")

    if rc.is_head:
        hf_dfts = {
            dft_name: dft for dft_name, dft in dfts.items() if isinstance(dft, hf.DFT)
        }
        Vnuc = next(iter(hf_dfts.values())).Vnuc if hf_dfts else None
        for i_site in range(n_sites):
            # Plot density and potential:
            plt.figure()
            z1d = get1D(grid1d.z)
            for dft_name, dft in dfts.items():
                plt.plot(z1d, get1D(dft.n.data[i_site]), label=f"$n$ ({dft_name})")
            plt.axhline(n_bulks[i_site].item(), color="k", ls="dotted")
            plt.xlabel("z")
            plt.legend()
            # Plot external potential shape (on separate axis) for comparison
            plt.sca(plt.gca().twinx())
            plt.plot(z1d, get1D(V.data[i_site]), "k--", label="$V$")
            if Vnuc is not None:
                plt.plot(z1d, get1D(Vnuc.data), "k:", label="$V_{nuc}$")
            plt.ylabel(f"Site {i_site} applied potential shape")
            plt.title(f"Site {i_site}")
            plt.savefig(f"{run_name}_site{i_site}.pdf", bbox_inches="tight")

        # Compare bulk free energy densities:
        plt.figure()
        n_bulks = torch.stack(
            [
                torch.linspace(*n_site_range, 101, device=rc.device)
                for n_site_range in n_bulk_range
            ],
            dim=-1,
        )
        for dft_name, dft in dfts.items():
            if isinstance(dft, Minimizer):
                f_bulks = dft.functionals[-1].get_energy_bulk(n_bulks)
            elif isinstance(dft, Schrodinger):
                f_bulks = ThomasFermi(dft.T).get_energy_bulk(n_bulks)
            elif isinstance(dft, hf.DFT):
                if dft.exchange_functional is None:
                    f_bulks = dft.bulk_exchange.get_energy_bulk(n_bulks)
                else:
                    f_bulks = dft.exchange_functional.get_energy_bulk(n_bulks)
            elif isinstance(dft, ising.Exact):
                f_bulks = ising.BulkExcess(dft.T, dft.J).get_energy_bulk(n_bulks)
            else:
                continue
            plt.plot(
                n_bulks.detach().to(rc.cpu).numpy()[..., 0],
                f_bulks.detach().to(rc.cpu).numpy(),
                label=dft_name,
            )
        plt.xlabel(r"$n_{\mathrm{bulk}0}$")
        plt.ylabel("Free-energy density")
        plt.legend()
        plt.savefig(f"{run_name}_EOS.pdf", bbox_inches="tight")

        # Visualize weight functions:
        def get_nn_functional(dft: protocols.DFT) -> Optional[nn.Functional]:
            if isinstance(dft, Minimizer):
                if isinstance(nn_functional := dft.functionals[-1], nn.Functional):
                    return nn_functional
            if isinstance(dft, hf.DFT):
                if isinstance(nn_functional := dft.exchange_functional, nn.Functional):
                    return nn_functional
            return None

        if any(get_nn_functional(dft) for dft in dfts.values()):
            for dft_name, dft in dfts.items():
                if (functional := get_nn_functional(dft)) is not None:
                    for i_layer, layer in enumerate(functional.layers):
                        plt.figure()
                        w_tilde = layer.get_w_tilde(grid1d.grid, n_dim_tot=3)
                        w = torch.fft.irfft(w_tilde).real / dz
                        for w_i in w.detach().to(rc.cpu).flatten(0, 1).numpy():
                            is_even = np.abs(w_i[0]) > 1e-6 * np.max(np.abs(w_i))
                            plt.plot(z1d, w_i, color=("r" if is_even else "b"))
                        plt.xlim(0, 0.5 * L)
                        plt.xlabel("$z$")
                        plt.ylabel("$w(z)$")
                        plt.legend(
                            [Line2D([0], [0], color=color) for color in "rb"],
                            ["Even weights", "Odd weights"],
                        )
                        plt.title(f"Weight functions in layer {i_layer} for {dft_name}")

        # Plot band structures:
        if hf_dfts:
            plt.figure()
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            for (dft_name, dft), color in zip(hf_dfts.items(), colors):
                k = dft.k.numpy() if dft.periodic else np.array([0.0, 1.0])
                eig = dft.eig.detach().to(rc.cpu).numpy()
                if not dft.periodic:
                    eig = np.repeat(eig, 2, axis=0)
                mu = dft.mu.item()
                plt.plot(k, eig - mu, "-", color=color, label=dft_name)
                # n_plot = np.max(np.where(eig < mu), axis=1)[1] + 4
                # plt.plot(k, eig[:, :n_plot] - mu, "-", color=color, label=dft_name)
            plt.xlabel("$k$")
            plt.ylabel(r"$\epsilon_{nk}-\mu$")
            plt.legend(
                [Line2D([0], [0], color=color, linestyle="-") for color in colors],
                hf_dfts.keys(),
            )
            plt.axhline(0, color="k", linestyle="dotted")
            plt.savefig(f"{run_name}_eigs.pdf", bbox_inches="tight")

        # Plot target potentials:
        plt.figure()
        for dft_name, dft in dfts.items():
            _, Vtarget = dft.training_targets()
            weight = (0.5 * torch.erfc(-torch.log(dft.n.data[0] / nc))) if nc else 1.0
            plt.plot(z1d, get1D(Vtarget.data[0] * weight), label=dft_name)
        plt.xlabel("z")
        plt.ylabel(rf"$V_{{target}}$ (Weighted with $n_c$ = {nc:.1e})")
        plt.legend()
        plt.savefig(f"{run_name}_Vtarget.pdf", bbox_inches="tight")

        rc.report_end()
        StopWatch.print_stats()
        # plt.show()
        plt.close()


class DataDFT:
    """Pose a read-in (MD) data set as a DFT for comparison."""

    def __init__(
        self, grid1d: Grid1D, filename: str, i_data: int, T: float, mu: torch.Tensor
    ) -> None:
        with h5py.File(filename) as fp:
            n_data = torch.tensor(fp["n"][i_data])
            E_n_data = torch.tensor(fp["dE_dn"][i_data])
            self.n = FieldR(grid1d.grid, data=n_data[:, None, None])
            self.Vtarget = FieldR(grid1d.grid, data=E_n_data[:, None, None])
            self.Etarget = float(fp["E"][i_data])
            self.mu = mu
            self.V = self.n.zeros_like()
            self.energy = Energy()
            self.ideal = hardrods.IdealGas(T)

    def minimize(self) -> Energy:
        """Solve Euler-Lagrange equation and return equilibrium energy."""
        V_minus_mu = FieldR(self.V.grid, data=(self.V.data - self.mu.view(-1, 1, 1, 1)))
        self.energy["Ext"] = (self.n ^ V_minus_mu).sum(dim=-1)
        self.energy["Excess"] = self.Etarget
        self.energy["Ideal"] = self.ideal.get_energy(self.n)
        print(self.energy)
        return self.energy

    def training_targets(self) -> tuple[float, FieldR]:
        return self.Etarget, self.Vtarget


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m mldft1d.test <input.yaml>")
        exit(1)
    in_file = sys.argv[1]
    run_name = os.path.splitext(in_file)[0]

    io.log_config()  # default set up to log from MPI head alone
    log.info("Using QimPy " + qimpy.__version__)
    rc.init()

    input_dict = io.dict.key_cleanup(io.yaml.load(in_file))
    run(**input_dict, run_name=run_name)


if __name__ == "__main__":
    main()
