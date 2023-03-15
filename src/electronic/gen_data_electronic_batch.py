import os
from gen_data_electronic import main
#
##!/bin/bash
##SBATCH -n 1 -N 1
#
#module load venv/qimpy
#
#for n_bulk in 0.4 0.5 0.6 0.7 0.8; do
#	export n_bulk
#	for L in 1 2 5 10; do
#		export L
#		for sigma in 0.1 0.2 0.5 1.0 2.0; do
#			export sigma
#			for i_seed in {1..10}; do
#				export seed=$((seed + 1))
#				python gen_data_electronic.py random.yaml
#			done
#		done
#	done
#done

n_bulks = [0.4, 0.5, 0.6, 0.7, 0.8,]
Ls = [1, 2, 5, 10,]
sigmas = [0.1, 0.2, 0.5, 1.0, 2.0,]
seed = 0
for n_bulk in n_bulks:
    for L in Ls:
        for sigma in sigmas:
            for i_seed in range(1, 10):
                seed += 1
                os.environ['n_bulk'] = str(n_bulk)
                os.environ['L'] = str(L)
                os.environ['sigma'] = str(sigma)
                os.environ['seed'] = str(seed)
                main()
