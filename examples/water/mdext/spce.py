from dataclasses import dataclass
import random
import sys

import numpy as np
from lammps import PyLammps

import mdext
from mdext import MPI, log


def main(U0ev, startfile, potential_seed) -> None:

    # Current simulation parameters:
    T = 300.0  # K
    P = None  # NVT
    lammps_seed = random.randint(1, 10000)  # mdext lammps seed
    U0 = (
        U0ev * 23.06
    )  # Convert amplitude of the external potential to kcal/mol since units water classical

    np.random.seed(potential_seed)
    sigma = np.random.uniform(0.5, 5)  # Angstrom (increasing range to have
    B = np.random.randn()
    if B > 0.5:
        A = -1.0
    elif B < 0.0:
        A = 1.0
    else:
        A = np.random.choice([1, -1])

    coeffs = A * np.array([1, B])
    setup = Setup(startfile)

    # Initialize and run simulation:
    md = mdext.md.MD(
        setup=setup,
        T=T,
        P=P,
        seed=lammps_seed,
        potential=mdext.potential.Gaussian(U0, sigma, coeffs),
        geometry_type=mdext.geometry.Planar,
        n_atom_types=2,
        potential_type=2,
        pe_collect_interval=100,
    )
    md.run(5, "equilibration")
    md.reset_stats()
    md.run(10, "collection", f"data-U{U0ev:+.2f}.h5")
    md.lmp.write_data(f"U{U0ev:+.2f}.step.data nocoeff")


@dataclass
class Setup:
    startfile: str

    def __call__(self, lmp: PyLammps, seed: int) -> int:
        """Setup initial atomic configuration and interaction potential."""

        file_liquid = self.startfile

        # Construct water box:
        L = np.array([30.0, 30.0, 40.0])  # overall box dimensions
        is_head = MPI.COMM_WORLD.rank == 0
        if is_head and file_liquid == "liquid.data":
            mdext.make_liquid.make_water(
                pos_min=[-L[0] / 2, -L[1] / 2, -L[2] / 2],
                pos_max=[+L[0] / 2, +L[1] / 2, +L[2] / 2],
                out_file=file_liquid,
            )
        lmp.atom_style("full")
        lmp.read_data(file_liquid)

        # Interaction potential (SPC/E, long-range):
        lmp.pair_style("lj/long/coul/long long long 10")
        lmp.kspace_style("pppm/disp 1e-5")
        lmp.kspace_modify("force/disp/real 0.0001 force/disp/kspace 0.002")
        lmp.bond_style("harmonic")
        lmp.angle_style("harmonic")
        lmp.set("type 1 charge  0.4238")
        lmp.set("type 2 charge -0.8476")
        lmp.pair_coeff("1 *2 0.000 0.000")  # No LJ for H
        lmp.pair_coeff("2 2 0.1553 3.166")  # O-O
        lmp.bond_coeff("1 1000 1.0")  # H-O
        lmp.angle_coeff("1 100 109.47")  # H-O-H

        # Initial minimize:
        if file_liquid == "liquid.data":
            log.info("Minimizing initial structure")
            lmp.neigh_modify("exclude molecule/intra all")
            lmp.minimize("1E-4 1E-6 10000 100000")

        # Rigid molecule constraints for dynamics:
        lmp.neigh_modify("exclude none")
        lmp.fix("BondConstraints all shake 0.001 20 0 b 1 a 1")


if __name__ == "__main__":

    potential_seed = int(sys.argv[1])  # np seed for generating the potential shape
    # each seed would be run in a separate folder to prevent conflict and run in parallel
    # sweep through potential amplitudes option:
    endRange = 0.30  # eV
    stepSize = 0.02  # eV

    # Ui in eV
    Uis = np.around(
        np.arange(0, endRange + stepSize, stepSize), decimals=2
    )  # positive only
    print("Uis to run: ", Uis)

    for i, Ui in enumerate(Uis):
        if Ui == 0.0:  # initial run
            startfile = "liquid.data"
        else:  # normal sequence within a run
            prev_Ui = Uis[i - 1]
            startfile = f"U{prev_Ui:+.2f}.step.data"

        print(f"launching seed {potential_seed}, Ui {Ui:+.2f}")
        main(Ui, startfile, potential_seed)

    print("Done!")
