import sys
import qimpy as qp
import hardrods1d as hr
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
):
    # Create grid and external potential:
    grid1d = hr.Grid1D(L=L, dz=dz)
    V = lbda * hr.v_shape.get(grid1d, **qp.utils.dict.key_cleanup(Vshape))

    # Create functionals:
    cdfts = {"Exact": hr.HardRodsFMT(grid1d, R=R, T=T, n_bulk=n_bulk)}
    for label, filename in functionals.items():
        cdfts[f"ML {label}"] = hr.mlcdft.Minimizer(
            functional=hr.mlcdft.Functional.load(qp.rc.comm, load_file=filename),
            grid1d=grid1d,
            n_bulk=n_bulk,
        )

    for cdft in cdfts.values():
        cdft.V = V
        cdft.finite_difference_test(cdft.random_direction())
        cdft.E = cdft.minimize()

    # Plot density and potential:
    if qp.rc.is_head:
        plt.figure(1, figsize=(10, 6))
        z1d = hr.get1D(grid1d.z)
        plt.plot(z1d, hr.get1D(V.data) / lbda, label=f"$V/V_0$ (with $V_0 = {lbda}$)")
        for cdft_name, cdft in cdfts.items():
            DeltaE = float(cdft.E) - cdft.e_bulk * grid1d.L
            qp.log.info(f"{cdft_name:>14s}:  mu: {cdft.mu:>7f} DeltaE: {DeltaE:>9f}")
            plt.plot(z1d, hr.get1D(cdft.n.data), label=f"$n$ ({cdft_name})")
        plt.axhline(n_bulk, color="k", ls="dotted")
        plt.xlabel("z")
        plt.ylim(0, None)
        plt.legend()
        plt.savefig("test.pdf", bbox_inches="tight")
        plt.show()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python test.py <input.yaml>")
        exit(1)
    in_file = sys.argv[1]

    qp.utils.log_config()  # default set up to log from MPI head alone
    qp.log.info("Using QimPy " + qp.__version__)
    qp.rc.init()

    input_dict = qp.utils.dict.key_cleanup(qp.utils.yaml.load(in_file))
    run(**input_dict)


if __name__ == "__main__":
    main()
