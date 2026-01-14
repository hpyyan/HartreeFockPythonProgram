from pyscf import dft
from pyscf import gto #(Gasussian type basis)

# Assign the geometry and basis to create Mole object.
# use .build() func to initialize the molecule

# (first method)
'''
mol = gto.Mole()
mol.atom = '''O 0 0 0; H 0 1 0; H 0 0 1''' # define molecules geometry
mol.basis = 'ccpvdz' #double zeta; select its basis func
mol.build()
'''

# (second method: shortcut func)
mol_h2o = gto.M(atom = 'O 0 0 0; H 0 1 0; H 0 0 1', basis = 'ccpvdz')

rks_hwo = dft.RKS(mol_h2o)
