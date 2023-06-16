import os
import sys
import qimpy as qp
import torch
from . import Grid1D, get1D, Minimizer, protocols, hardrods, kohnsham, ising, nn
from .data import v_shape
from .kohnsham import Schrodinger, ThomasFermi
from typing import Callable
import matplotlib.pyplot as plt


make_dft_map: dict[str, Callable[..., protocols.DFT]] = {
    "hardrods_exact": hardrods.make_dft.exact,
    "hardrods_ml": hardrods.make_dft.ml,
    "kohnsham_exact": kohnsham.make_dft.exact,
    "kohnsham_ml": kohnsham.make_dft.ml,
    "ising_exact": ising.make_dft.exact,
    "ising_ml": ising.make_dft.ml,
}  #: Recognized DFTs that can be loaded from YAML input


def run(
    *,
    L: float,
    dz: float,
    n_bulk: float,
    Vshape: dict,
    lbda: float,
    functionals: dict,
    run_name: str,
    n_bulk_range: tuple[float, float] = (0.0, 1.0),
    **dft_common_args,
):
    # Create grid and external potential:
    grid1d = Grid1D(L=L, dz=dz)
    V = lbda * v_shape.get(grid1d, **qp.utils.dict.key_cleanup(Vshape))

    # Create DFTs:
    dfts = dict[str, protocols.DFT]()
    for label, dft_dict in functionals.items():
        for dft_name, dft_args in qp.utils.dict.key_cleanup(dft_dict).items():
            dfts[label] = make_dft_map[dft_name](
                grid1d=grid1d, n_bulk=n_bulk, **dft_args, **dft_common_args
            )

    for dft in dfts.values():
        dft.V = V
        if isinstance(dft, Minimizer):
            dft.finite_difference_test(dft.random_direction())
        dft.minimize()  # equilibrium results in dft.energy and dft.n

    if qp.rc.is_head:
        # Plot density and potential:
        plt.figure(1, figsize=(10, 6))
        z1d = get1D(grid1d.z)
        plt.plot(z1d, get1D(V.data), label="$V$")
        for dft_name, dft in dfts.items():
            E = float(dft.energy)
            qp.log.info(f"{dft_name:>14s}:  mu: {dft.mu:>7f}  E: {E:>9f}")
            plt.plot(z1d, get1D(dft.n.data), label=f"$n$ ({dft_name})")
        plt.axhline(n_bulk, color="k", ls="dotted")
        plt.axhline(0.0, color="k", ls="dotted")
        plt.xlabel("z")
        plt.legend()
        plt.savefig(f"{run_name}.pdf", bbox_inches="tight")

        # Compare bulk free energy densities:
        plt.figure(2)
        n_bulks = torch.linspace(*n_bulk_range, 101, device=qp.rc.device)
        for dft_name, dft in dfts.items():
            if isinstance(dft, Minimizer):
                f_bulks = dft.functionals[-1].get_energy_bulk(n_bulks)
            elif isinstance(dft, Schrodinger):
                f_bulks = ThomasFermi(dft.T).get_energy_bulk(n_bulks)
            elif isinstance(dft, ising.Exact):
                f_bulks = ising.HomogeneousIsing(dft.T, dft.J).get_energy_bulk(n_bulks)
            else:
                continue
            plt.plot(
                n_bulks.detach().to(qp.rc.cpu).numpy(),
                f_bulks.detach().to(qp.rc.cpu).numpy(),
                label=dft_name,
            )
        plt.xlabel(r"$n_{\mathrm{bulk}}$")
        plt.ylabel("Free-energy density")
        plt.legend()

        # Visualize weight functions:
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
                            w.detach().to(qp.rc.cpu).split(layer.n_weights, dim=1),
                            layer.n_nonlocal,
                        ):
                            w_np = w_set[:, :n_nonlocal].flatten(0, 1).numpy()
                            label = f"{label} weights"
                            for i_w, w_i in enumerate(w_np):
                                plt.plot(z1d, w_i, style, label=("" if i_w else label))
                        plt.xlim(0, 0.5 * L)
                        plt.xlabel("$z$")
                        plt.ylabel("$w(z)$")
                        plt.legend()
                        plt.title(f"Weight functions in layer {i_layer} for {dft_name}")
        plt.show()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m mldft1d.test <input.yaml>")
        exit(1)
    in_file = sys.argv[1]
    run_name = os.path.splitext(in_file)[0]

    qp.utils.log_config()  # default set up to log from MPI head alone
    qp.log.info("Using QimPy " + qp.__version__)
    qp.rc.init()

    input_dict = qp.utils.dict.key_cleanup(qp.utils.yaml.load(in_file))
    run(**input_dict, run_name=run_name)


if __name__ == "__main__":
    main()
