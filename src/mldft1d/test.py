import os
import sys
import qimpy as qp
from . import Grid1D, get1D, Minimizer, protocols, hardrods
from .data import v_shape
from typing import Callable
import matplotlib.pyplot as plt


make_dft_map: dict[str, Callable[..., protocols.DFT]] = {
    "hardrods_exact": hardrods.make_dft.exact,
    "hardrods_ml": hardrods.make_dft.ml,
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

    # Plot density and potential:
    if qp.rc.is_head:
        plt.figure(1, figsize=(10, 6))
        z1d = get1D(grid1d.z)
        plt.plot(z1d, get1D(V.data) / lbda, label=f"$V/V_0$ (with $V_0 = {lbda}$)")
        for dft_name, dft in dfts.items():
            E = float(dft.energy)
            qp.log.info(f"{dft_name:>14s}:  mu: {dft.mu:>7f}  E: {E:>9f}")
            plt.plot(z1d, get1D(dft.n.data), label=f"$n$ ({dft_name})")
        plt.axhline(n_bulk, color="k", ls="dotted")
        plt.xlabel("z")
        plt.ylim(0, None)
        plt.legend()
        plt.savefig(f"{run_name}.pdf", bbox_inches="tight")
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
