import torch
import qimpy as qp
import hardrods1d as hr
import matplotlib.pyplot as plt


def main():
    qp.utils.log_config()  # default set up to log from MPI head alone
    qp.log.info("Using QimPy " + qp.__version__)
    qp.rc.init()

    grid1d = hr.Grid1D(L=20.0, dz=0.05)
    n_bulk = 0.6
    T = 1.0

    # Create external potential:
    V0 = 0.5 * T
    V = V0 * hr.v_shape.get(grid1d, shape="rectangular", sigma=0.1, duty=0.4)

    # Create exact functional:
    cdfts = {}
    cdfts["Exact"] = hr.HardRodsFMT(grid1d, R=0.5, T=T, n_bulk=n_bulk)

    # Create MLCDFT approximation:
    functional = hr.mlcdft.Functional(
        T=T,
        w=hr.mlcdft.NNFunction(1, 2, [10, 10]),
        f_ex=hr.mlcdft.NNFunction(2, 2, [10, 10]),
    )
    functional.load_state_dict(
        torch.load("mlcdft_params.dat", map_location=qp.rc.device)
    )
    cdfts["ML"] = hr.mlcdft.Minimizer(
        functional=functional, grid1d=grid1d, n_bulk=n_bulk
    )

    for cdft in cdfts.values():
        cdft.V = V
        cdft.finite_difference_test(cdft.random_direction())
        cdft.E = cdft.minimize()

    # Plot density and potential:
    if qp.rc.is_head:
        plt.figure(1, figsize=(10, 6))
        z1d = hr.get1D(grid1d.z)
        plt.plot(z1d, hr.get1D(V.data) / V0, label=f"$V/V_0$ (with $V_0 = {V0}$)")
        for cdft_name, cdft in cdfts.items():
            DeltaE = float(cdft.E) - cdft.mu * n_bulk * grid1d.L
            qp.log.info(f"{cdft_name:>7s}:  mu: {cdft.mu:>7f} DeltaE: {DeltaE:>9f}")
            plt.plot(z1d, hr.get1D(cdft.n.data), label=f"$n$ ({cdft_name})")
        plt.axhline(n_bulk, color="k", ls="dotted")
        plt.xlabel("z")
        plt.ylim(0, None)
        plt.legend()
        plt.savefig("test.pdf", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    main()
