#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from os import listdir
from os.path import isfile, join
import yaml

Z = 1  # nuclear charge
os.environ["Z"] = f"{Z:g}"

if (len(sys.argv) != 2) or (sys.argv[1] != "run" and sys.argv[1] != "plot"):
    print('Must pass "run" or "plot" to Cohesive.py')

rootdir = "./"


def run():
    L = 20
    n = 1.0 * Z / L
    os.environ["n_a"] = f"{n:g}"
    os.system("python3 -m mldft1d.test %s/atom.yaml > atom%d.out" % (rootdir, Z))
    for ext in ["site0", "EOS", "eigs", "Vtarget"]:
        os.system("mv %s/atom_%s.pdf atom%d_%s.pdf" % (rootdir, ext, Z, ext))
    for x in np.linspace(0.76, 8, 30):
        x_str = f"{x:g}"
        n = 1.0 * Z / x
        os.environ["lat_a"] = f"{x:g}"
        os.environ["n_a"] = f"{n:g}"
        os.system("echo $lat_a $n_a")
        os.system(
            "python3 -m mldft1d.test %s/chain.yaml > chain%d_%s.out"
            % (rootdir, Z, x_str)
        )
        for ext in ["site0", "EOS", "eigs", "Vtarget"]:
            os.system(
                "mv %s/chain_%s.pdf chain%d_%s_%s.pdf" % (rootdir, ext, Z, ext, x_str)
            )


def plot(workdir="./"):
    with open("test_common.yaml", "r") as f:
        input_dict = yaml.safe_load(f)
    fcnls = list(input_dict["functionals"].keys())
    outfiles = [
        f
        for f in listdir(workdir)
        if isfile(join(workdir, f)) and ".out" in f and "chain%d" % (Z) in f
    ]
    R = np.zeros(len(outfiles))
    H1_energy = {}
    H2_energy = {}
    H2_gap = {}
    read_fcnls = {}
    for _fcnl in fcnls:
        H2_energy[_fcnl] = 0.0
        H2_energy[_fcnl] = np.zeros(len(outfiles))
        H2_gap[_fcnl] = np.zeros(len(outfiles))
        read_fcnls[_fcnl] = 0
    H1_energy = {}
    with open("atom%d.out" % (Z)) as f:
        fread = f.read()
        if fread.count("Converged") != len(fcnls) + 2:
            print("WARNING: H1 DID NOT CONVERGE ON SCF AND/OR OEP (or +/- dEx/dN)")
        f.seek(0)
        flines = f.readlines()
        start = 0
        for _fcnl in fcnls:
            if np.prod(list(read_fcnls.values())):
                break
            for line, i in zip(flines[start:], range(len(flines[start:]))):
                if "%s:" % (_fcnl) in line:
                    start = i
                    H1_energy[_fcnl] = float(line.split()[4])  # total energy
                    read_fcnls[_fcnl] = 1
    for _out, o in zip(outfiles, range(len(outfiles))):
        read_fcnls = dict.fromkeys(read_fcnls, 0)
        with open(workdir + _out) as f:
            fread = f.read()
            if fread.count("Converged") != len(fcnls) + 2:
                print(
                    "WARNING: ",
                    _out,
                    " DID NOT CONVERGE ON SCF AND/OR OEP (or +/- dEx/dN)",
                )
            R[o] = float(_out[7:-4])
            f.seek(0)
            flines = f.readlines()
            start = 0
            for _fcnl in fcnls:
                if np.prod(list(read_fcnls.values())):
                    break
                for line, i in zip(flines[start:], range(len(flines[start:]))):
                    if "%s:" % (_fcnl) in line:
                        start = i
                        H2_energy[_fcnl][o] = float(line.split()[4])
                        H2_gap[_fcnl][o] = float(line.split()[6])
                        read_fcnls[_fcnl] = 1
    sort = np.argsort(R)
    R = R[sort]
    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    ax.set_title("Cohesize energy of 1D atom chain (Z=%d)" % (Z))
    ax1.set_title("HOMO-LUMO gap")
    for _fcnl in fcnls:
        ax.plot(R, H2_energy[_fcnl][sort] - H1_energy[_fcnl], "-", label=_fcnl)
        ax1.plot(R, H2_gap[_fcnl][sort], "-", label=_fcnl)
        print(_fcnl, H2_gap[_fcnl][sort])
    ax.axhline(0, color="k", linestyle="--")
    ax.legend()
    ax.set_xlabel("Lattice constant [$a_0$]")
    ax.set_ylabel("Cohesive energy [$E_H$]")
    ax1.set_ylabel("Gap [$E_H$]")
    ax1.set_xlabel("Lattice constant [$a_0$]")
    plt.legend()
    # plt.savefig("atomChain-CohesiveEnergy.pdf", transparent=True, bbox_inches="tight")
    plt.show()


if sys.argv[1] == "run":
    run()
if sys.argv[1] == "plot":
    plot()
