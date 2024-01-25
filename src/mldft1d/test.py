from __future__ import annotations
from typing import Callable, Sequence, Union
import sys
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

import qimpy
from qimpy import rc, log, io
from qimpy.grid import FieldR
from . import Grid1D, get1D, Minimizer, protocols, hardrods, kohnsham, hf, ising, nn
from .data import v_shape
from .kohnsham import Schrodinger, ThomasFermi


make_dft_map: dict[str, Callable[..., protocols.DFT]] = {
    "hardrods_exact": hardrods.make_dft.exact,
    "hardrods_ml": hardrods.make_dft.ml,
    "kohnsham_exact": kohnsham.make_dft.exact,
    "kohnsham_ml": kohnsham.make_dft.ml,
    "hf_exact": hf.make_dft.exact,
    "hf_lda": hf.make_dft.lda,
    "hf_ml": hf.make_dft.ml,
    "ising_exact": ising.make_dft.exact,
    "ising_ml": ising.make_dft.ml,
}  #: Recognized DFTs that can be loaded from YAML input


def run(
    *,
    L: float,
    dz: float,
    n_bulk: Union[float, Sequence[float]],
    Vshape: Union[dict, Sequence[dict]],
    lbda: float,
    functionals: dict,
    run_name: str,
    n_bulk_range: Sequence[tuple[float, float]] = ((0.0, 1.0),),
    **dft_common_args,
):
    # Check site density / potential counts:
    Vshape = [Vshape] if isinstance(Vshape, dict) else Vshape
    n_bulks = torch.tensor(
        [n_bulk] if isinstance(n_bulk, float) else n_bulk, device=rc.device
    )
    n_sites = len(n_bulks)
    assert len(Vshape) == n_sites

    # Create grid and external potential:
    grid1d = Grid1D(L=L, dz=dz)
    for Vshape_i in Vshape:
        if Vshape_i["shape"] == "coulomb1d":
            Vshape_i["a"] = dft_common_args["a"]
            Vshape_i["periodic"] = dft_common_args["periodic"]
    Vdata = lbda * torch.stack(
        [
            v_shape.get(grid1d, **io.dict.key_cleanup(Vshape_i)).data
            for Vshape_i in Vshape
        ]
    )
    V = FieldR(grid1d.grid, data=Vdata)

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
        mu = dft.mu
        if isinstance(dft, hf.DFT):
            eig = dft.eig.detach().numpy()
            homo = np.max(np.where(eig < mu), axis=1)
            lumo = np.min(np.where(eig > mu), axis=1)
            gap = eig[lumo[0], lumo[1]] - eig[homo[0], homo[1]]
        else:
            gap = np.nan
        log.info(f"{dft_name:>14s}:  mu: {io.fmt(mu)}  E: {E:>9f}  gap: {gap:>9f}")

    if rc.is_head:
        for i_site in range(n_sites):
            # Plot density and potential:
            plt.figure(figsize=(10, 6))
            z1d = get1D(grid1d.z)
            plt.plot(z1d, get1D(V.data[i_site]), label="$V$")
            for dft_name, dft in dfts.items():
                plt.plot(z1d, get1D(dft.n.data[i_site]), label=f"$n$ ({dft_name})")
            plt.axhline(n_bulks[i_site], color="k", ls="dotted")
            plt.axhline(0.0, color="k", ls="dotted")
            plt.xlabel("z")
            plt.title(f"Site {i_site}")
            plt.legend()
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
                f_bulks = ThomasFermi(dft.T).get_energy_bulk(n_bulks)
                f_bulks += dft.bulk_exchange(n_bulks)
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
        plot_weights = any(
            (
                isinstance(dft, Minimizer)
                and isinstance(dft.functionals[-1], nn.Functional)
            )
            for dft in dfts.values()
        )
        if plot_weights:
            for dft_name, dft in dfts.items():
                if isinstance(dft, Minimizer):
                    if isinstance((functional := dft.functionals[-1]), nn.Functional):
                        for i_layer, layer in enumerate(functional.layers):
                            plt.figure()
                            w_tilde = layer.get_w_tilde(grid1d.grid, n_dim_tot=3)
                            w = torch.fft.irfft(w_tilde).real / dz
                            for label, style, w_set, n_nonlocal in zip(
                                ("Even", "Odd"),
                                ("r", "b"),
                                w.detach().to(rc.cpu).split(layer.n_weights, dim=1),
                                layer.n_nonlocal,
                            ):
                                w_np = w_set[:, :n_nonlocal].flatten(0, 1).numpy()
                                label = f"{label} weights"
                                for i_w, w_i in enumerate(w_np):
                                    plt.plot(
                                        z1d, w_i, style, label=("" if i_w else label)
                                    )
                            plt.xlim(0, 0.5 * L)
                            plt.xlabel("$z$")
                            plt.ylabel("$w(z)$")
                            plt.legend()
                            plt.title(
                                f"Weight functions in layer {i_layer} for {dft_name}"
                            )

        plot_bandstructures = any(isinstance(dft, hf.DFT) for dft in dfts.values())
        if plot_bandstructures:
            plt.figure()
            styles = ["b-", "r:"]
            names = []
            for (dft_name, dft), style in zip(dfts.items(), styles):
                names.append(dft_name)
                if isinstance(dft, hf.DFT):
                    k = dft.k.numpy() if dft.periodic else np.array([0.0, 1.0])
                    eig = dft.eig.detach().numpy()
                    if not dft.periodic:
                        eig = np.repeat(eig, 2, axis=0)
                    mu = dft.mu
                    n_plot = np.max(np.where(eig < mu), axis=1)[1] + 4
                    plt.plot(k[:, ...], eig[:, :n_plot] - mu, style, label=dft_name)
            plt.xlabel("$k$")
            plt.ylabel(r"$\epsilon_{nk}-\mu$")
            plt.legend(
                [
                    Line2D([0], [0], color=styles[0][0], linestyle=styles[0][1]),
                    Line2D([0], [0], color=styles[1][0], linestyle=styles[1][1]),
                ],
                names,
            )
            plt.axhline(0, color="k", linestyle="--")
            plt.savefig(f"{run_name}-eigs.pdf", bbox_inches="tight")

        plt.show()


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
