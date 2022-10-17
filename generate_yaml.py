
import yaml

Params = {'L': 40, 'dz': 0.01, 'R': 0.5, 'T': 0.1, 'n_bulk': 0.6, 'vsigma': 0.1, 'minimum': 1, 'maximum': 101, 'step': 1}

with open(r'Params.yaml', 'w') as file:
    documents = yaml.dump(Params, file)