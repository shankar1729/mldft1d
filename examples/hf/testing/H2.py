#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from os import listdir
from os.path import isfile, join
import yaml


if (len(sys.argv) != 2) or (sys.argv[1] != "run" and sys.argv[1] != "plot"):
    print('Must pass "run" or "plot" to H2.py')

rootdir = "./"


def run():
    os.system("python3 -m mldft1d.test %s/H1.yaml > H1.out" % (rootdir))
    for ext in ["site0", "EOS", "eigs", "Vtarget"]:
        os.system("mv %s/H1_%s.pdf H1_%s.pdf" % (rootdir, ext, ext))
    for x in np.linspace(0.1, 4, 32):
        p1 = 20 - x  # position of atom 1
        p2 = 20 + x  # position of atom 2
        x_str = f"{x:g}"
        os.environ["p1"] = f"{p1:g}"  # str(p1)
        os.environ["p2"] = f"{p2:g}"  # str(p2)
        os.system("echo $p1 $p2")
        os.system("python3 -m mldft1d.test %s/H2.yaml > H2_%s.out" % (rootdir, x_str))
        for ext in ["site0", "EOS", "eigs", "Vtarget"]:
            os.system("mv %s/H2_%s.pdf H2_%s_%s.pdf" % (rootdir, ext, ext, x_str))


def plot(workdir="./"):
    with open("test_common.yaml", "r") as f:
        input_dict = yaml.safe_load(f)
    fcnls = list(input_dict["functionals"].keys())
    outfiles = [
        f
        for f in listdir(workdir)
        if isfile(join(workdir, f)) and ".out" in f and "H2" in f
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
    with open("H1.out") as f:
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
            R[o] = float(_out[3:-4])
            f.seek(0)
            flines = f.readlines()
            start = 0
            for _fcnl in fcnls:
                if np.prod(list(read_fcnls.values())):
                    break
                for line, i in zip(flines[start:], range(len(flines[start:]))):
                    if "%s:" % (_fcnl) in line:
                        start = i
                        H2_energy[_fcnl][o] = float(line.split()[4])  # total energy
                        H2_gap[_fcnl][o] = float(line.split()[6])  # total energy
                        read_fcnls[_fcnl] = 1  # total energy
    sort = np.argsort(R)
    R = 2 * R[sort]
    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    ax.set_title("H$_2$ binding energy in 1D")
    ax1.set_title("HOMO-LUMO gap")
    for _fcnl in fcnls:
        ax.plot(R, H2_energy[_fcnl][sort] - 2 * H1_energy[_fcnl], "-", label=_fcnl)
        ax1.plot(R, H2_gap[_fcnl][sort], "-", label=_fcnl)
    ax.axhline(0, color="k", linestyle="--")
    ax.legend()
    ax.set_ylabel("Binding energy [$E_H$]")
    ax.set_xlabel("Distance $[a_0$]")
    ax1.set_ylabel("HOMO-LUMO gap [$E_H$]")
    ax1.set_xlabel("Distance $[a_0$]")
    plt.legend()
    # plt.savefig("H2bindingE.pdf", transparent=True, bbox_inches="tight")
    plt.show()


if sys.argv[1] == "run":
    run()
if sys.argv[1] == "plot":
    plot()
