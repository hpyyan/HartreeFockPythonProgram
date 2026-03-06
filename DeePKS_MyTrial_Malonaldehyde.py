import sgdml
import torch
import numpy as np

dataset = sgdml.load_dataset('Malonaldehyde_dataset_sGDML/malonaldehyde_ccsd_t-test.npz')

data = np.load('malonaldehyde_ccsd_t.npz')

R = data['R'] #geometry
E = data['E'] #energy
F = data['F'] #structure

print(R, E, F)