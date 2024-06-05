Steps
-----

1. Install pylammps and [mdext](https://github.com/shankar1729/mdext) in a virtual environment, which can say be loaded using a module called `venv/mdext` (assumed in the job files)
2. Edit `run.job` as needed for your compute resources and then run `launch.py` (this should launch several hundered MD jobs)
3. Run `convert.py` to convert mdext outputs to train/test data files for mldft1d (create destination directory Data/ in advance)
4. Run `make-bulk.py` to produce auxiliary files to help train the equation of state

