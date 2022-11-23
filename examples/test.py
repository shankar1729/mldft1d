import torch
import qimpy as qp
import hardrods1d as hr
import matplotlib.pyplot as plt


def main():
    qp.utils.log_config()  # default set up to log from MPI head alone
    qp.log.info("Using QimPy " + qp.__version__)
    qp.rc.init()

    grid1d = hr.Grid1D(L=40.0, dz=0.01)
    n_bulk = 0.6
    T = 1.0

    # Create external potential:
    V0 = 2 * T
    Vsigma = 0.1
    V = qp.grid.FieldR(
        grid1d.grid,
        data=(0.5 * V0)
        * ((0.4 * grid1d.L - (grid1d.z - 0.5 * grid1d.L).abs()) / Vsigma).erfc(),
    )

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
        cdft.minimize()

    # Plot density and potential:
    if qp.rc.is_head:
        z1d = hr.get1D(grid1d.z)
        plt.plot(z1d, hr.get1D(V.data), label="V")
        for cdft_name, cdft in cdfts.items():
            qp.log.info(f"mu ({cdft_name}) = {cdft.mu}")
            plt.plot(z1d, hr.get1D(cdft.n.data), label=f"n ({cdft_name})")
        plt.axhline(n_bulk, color="k", ls="dotted")
        plt.xlabel("z")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
