# Installation

Python 3.11 or below is currently required for installing this software, as `setuptools` was removed in Python 3.12. This repository requires installation of [pytorch](https://pytorch.org/) followed by [qimpy](https://github.com/shankar1729/qimpy/).

Once these steps are complete, run `python setup.py develop` in the root of this repository to install this repository to your current python environment (alternatively `python setup.py install` if you would like a static installation and do not plan to do make active modifications to the code).

# Examples

The general workflow for this software is

1. Generate data for testing/training of a model for a particular system.
2. Perform training.
3. Inference trained models on particular potentials, comparing known or exact functionals to machine learned functionals.

## Data Generation

Using a Hartree-Fock system as an example, you can generate training data for this system by referencing `examples/hf`. 
Data corresponding to random bandwidth-limited potentials can be generated with `generate_random.py` and data for molecule-like potentials can be generated with `generate_random_molecules.py`.
The data-generation process can be performance-intensive, so these scripts are MPI enabled. Examples for running these generation scripts as jobs with MPI on a HPC with a SLURM scheduler can be seen in the `.job` files.

## Training

A good example for training may be seen in `examples/hf/gaussian/3layer/train.yaml`. Modify `data/filenames` within this file to point to the data you have generated. Within your python environment, you may now run `python -m mldft.nn.train train.yaml`.
Training is performance intensive, so, if you are on an HPC, it is recommended to run this script as a SLURM job with MPI enabled. It is also recommended to ensure that GPU acceleration is being used if it is available on your system.

## Testing

If training is successful, it will produce data files for parameters and loss history. You can use these trained parameters to test on new systems. Edit `examples/hf/test_common.yaml` to point to the parameters file you have generated (under `load_file`), and specify any other functionals you would like to test.
The settings at `mldft1d/examples/hf/test_random.yaml` and `mldft1d/examples/hf/test_molecule.yaml` will allow you to test these models on a new randomly generated potential or molecule respectively, plotting the density, equation of state, and corresponding weight functions.

You can now run e.g. `python -m mldft1d.test test_random.yaml` to produce these results.
