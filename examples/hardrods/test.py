import os
import sys
import qimpy as qp
from mldft1d import Grid1D, get1D, Minimizer, protocols
from mldft1d.data import v_shape
from mldft1d.nn import Functional
from mldft1d.hardrods import IdealGas, FMT
import matplotlib.pyplot as plt


def run(
    *,
    L: float,
    dz: float,
    R: float,
    T: float,
    n_bulk: float,
    Vshape: dict,
    lbda: float,
    functionals: dict,
    run_name: str,
):
    # Create grid and external potential:
    grid1d = Grid1D(L=L, dz=dz)
    V = lbda * v_shape.get(grid1d, **qp.utils.dict.key_cleanup(Vshape))

    # Create functionals:
    dfts = dict[str, protocols.Variational]()
    dfts["Exact"] = Minimizer(
        functionals=(IdealGas(T), FMT(grid1d, T=T, R=R)),
        grid1d=grid1d,
        n_bulk=n_bulk,
        name="FMT",
    )
    for label, filename in functionals.items():
        dfts[f"ML {label}"] = Minimizer(
            functionals=(
                IdealGas(T),
                Functional.load(qp.rc.comm, load_file=filename),
            ),
            grid1d=grid1d,
            n_bulk=n_bulk,
            name="MLCDFT",
        )

    for dft in dfts.values():
        dft.V = V
        if isinstance(dft, qp.utils.Minimize):
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
        print("Usage: python test.py <input.yaml>")
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
