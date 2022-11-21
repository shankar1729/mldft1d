import torch
import qimpy as qp
import hardrods1d as hr
import matplotlib.pyplot as plt


def main():
    qp.utils.log_config()  # default set up to log from MPI head alone
    qp.log.info("Using QimPy " + qp.__version__)
    qp.rc.init()

    grid1d = hr.Grid1D(L=40.0, dz=0.01)
    # cdft = hr.HardRodsFMT(grid1d, R=0.5, T=0.1, n_bulk=0.6)

    cdft = hr.mlcdft.Minimizer(
        functional=hr.mlcdft.Functional(
            T=1.0, w=hr.mlcdft.NNFunction(1, 2, []), f_ex=hr.mlcdft.NNFunction(2, 2, [])
        ),
        grid1d=grid1d,
        n_bulk=0.6,
    )
    cdft.functional.load_state_dict(torch.load("mlcdft_params.dat"))
    qp.log.info(f"mu = {cdft.mu}")

    # Set external potential:
    V0 = 10 * cdft.T
    Vsigma = 0.1
    cdft.V.data = (0.5 * V0) * (
        (0.4 * grid1d.L - (grid1d.z - 0.5 * grid1d.L).abs()) / Vsigma
    ).erfc()

    # Finite difference test
    cdft.finite_difference_test(cdft.random_direction())

    cdft.minimize()
    n = cdft.n

    # Plot density and potential:
    if qp.rc.is_head:
        z1d = hr.get1D(grid1d.z)
        plt.plot(z1d, hr.get1D(cdft.V.data), label="V")
        plt.plot(z1d, hr.get1D(n.data), label="n")
        # plt.plot(z1d, get1D(n.convolve(w0_tilde).data), label="n0")
        # plt.plot(z1d, get1D(n.convolve(w1_tilde).data), label="n1")
        plt.axhline(cdft.n_bulk, color="k", ls="dotted")
        plt.xlabel("z")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
