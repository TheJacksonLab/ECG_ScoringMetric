import numpy as np
from scipy.linalg import fractional_matrix_power
import sys


############ Set file path & parameters ############
path = './'
QC_output_filename = 'BTBT_MO.out' #'TestMoleculeSmall.inp.out'


########## Read general information ##########
f = open((path+QC_output_filename), "r")
full_content = f.readlines()
f.close()
full_content_numpy = np.array(full_content)

line = np.argwhere(full_content_numpy=="General Settings:\n").flatten()[0]
no_electron = int(full_content[line+5].split()[5])
no_basis = int(full_content[line+6].split()[4])
HOMO_level = int(no_electron/2)
line_1 = np.argwhere(full_content_numpy=="INTERNAL COORDINATES (A.U.)\n").flatten()[0]
line_2 = np.argwhere(full_content_numpy=="BASIS SET INFORMATION\n").flatten()[0]
no_atoms = line_2 - line_1 - 4
print("Number of Electrons: %s" %(no_electron))
print("Basis Dimension: %s" %(no_basis))
print("Inex of HOMO: %s" %(HOMO_level))
print("Number of atoms: %s" %(no_atoms))


########## Read nuclear charges ZA##########
ZA = np.zeros(no_atoms)
line = np.argwhere(full_content_numpy=="CARTESIAN COORDINATES (A.U.)\n").flatten()[0]
for i in range(no_atoms):
  ZA[i] = float(full_content_numpy[line+3+i].split()[2])


########## Identify number of section of overlap matrix (molecular orbitals) ##########
no_column_last_section = no_basis%6
if (no_column_last_section==0):
  no_section = int(no_basis/6)
else:
  no_section = int(no_basis/6)+1


########## Read overlap matrix ##########
overlap_matrix = np.zeros((no_basis,no_basis))
line = np.argwhere(full_content_numpy=="OVERLAP MATRIX\n").flatten()[0]
for sec in range(no_section-1):
  #print(full_content[line+2+sec*(no_basis+1)])
  #print(sec*6,(sec+1)*6-1)
  for i in range(no_basis):
    overlap_matrix[i,sec*6:(sec+1)*6] = full_content[line+3+i+(sec*(no_basis+1))].split()[1:]
### Last section ###
if (no_column_last_section==0):
  for sec in range(no_section-1):
    for i in range(no_basis):
      overlap_matrix[i,sec*6:(sec+1)*6] = full_content[line+3+i+(sec*(no_basis+1))].split()[1:]
else:
  for sec in range((no_section-1),no_section):
    for i in range(no_basis):
      overlap_matrix[i,sec*6:(sec*6+no_column_last_section)] = full_content[line+3+i+(sec*(no_basis+1))].split()[1:]


########## Read molecular orbitals ##########
C_mn = np.zeros((no_basis,no_basis)) #[AO,MO]
line = np.argwhere(full_content_numpy=="MOLECULAR ORBITALS\n").flatten()[0]

for sec in range(no_section-1):
  #print(sec*6,(sec+1)*6-1)
  for i in range(no_basis):
    C_mn[i,sec*6:(sec+1)*6] = full_content[line+6+i+(sec*(no_basis+4))].split()[2:]
    #print(full_content[line+6+i+(sec*(no_basis+4))])
### Last section ###
if (no_column_last_section==0):
  for sec in range(no_section-1):
    for i in range(no_basis):
      C_mn[i,sec*6:(sec+1)*6] = full_content[line+6+i+(sec*(no_basis+4))].split()[2:]
else:
  for sec in range((no_section-1),no_section):
    for i in range(no_basis):
      C_mn[i,sec*6:(sec*6+no_column_last_section)] = full_content[line+6+i+(sec*(no_basis+4))].split()[2:]


########## Identify number of basis per atom ##########
no_basis_per_atom = np.zeros(no_atoms, dtype=int)
line = np.argwhere(full_content_numpy=="MOLECULAR ORBITALS\n").flatten()[0]
check = []
for i in range(no_basis):
  check.append(int(full_content[line+6+i][:3]))
check = np.array(check)
for i in range(no_atoms):
  no_basis_per_atom[i] = np.sum(check==i)
#print(no_basis_per_atom.sum())

"""
no_basis = 2
HOMO_level = 1
C_mn = np.array([[0.8019, -0.7823],[0.3368, 1.0684]])
#print(C_mn)
overlap_matrix = np.array([[1.0, 0.4508],[0.4508, 1.0]])
#print(overlap_matrix)
"""
for MO in range((HOMO_level-1),HOMO_level): 
  print(MO)

########## Calculate density matrix ##########
density_matrix = np.zeros((no_basis,no_basis))
for mu in range(no_basis):
  for nu in range(no_basis):
    #for MO in range(HOMO_level): #Consider all the electron occupied MO
    for MO in range((HOMO_level-1),HOMO_level): # Only consider HOMO
    #for MO in range(HOMO_level,(HOMO_level+1)): # Only consider LUMO
      density_matrix[mu,nu] = density_matrix[mu,nu] + C_mn[mu,MO]*C_mn[nu,MO]

density_matrix = 2* density_matrix
print(density_matrix.shape)


########## Mulliken population analysis ##########
PS_matrix = np.matmul(density_matrix,overlap_matrix)
diagonal_element = np.diagonal(PS_matrix)
#print(PS_matrix)

### Calculate number of electrons ###
#N = np.trace(PS_matrix)
N = diagonal_element.sum()
print("Number of electrons calculated by Mulliken approach: %s" %(N))

### Calculate atomic partial charges ###
atom_charge = np.zeros(no_atoms)
index_start = 0
index_end = 0
for i in range(no_atoms):
  index_end = index_end + no_basis_per_atom[i]
  atom_charge[i] = ZA[i] - diagonal_element[index_start:index_end].sum()
  index_start = index_start + no_basis_per_atom[i]
print("Mulliken charges:")
print(atom_charge)


########## Loewdin population analysis ##########
S_half = fractional_matrix_power(overlap_matrix, 0.5)
Loewdin_matrix = np.matmul((np.matmul(S_half,density_matrix)), S_half)
#print(Loewdin_matrix)
diagonal_element = np.diagonal(Loewdin_matrix)

### Calculate number of electrons ###
#N = np.trace(PS_matrix)
N = diagonal_element.sum()
print("Number of electrons calculated by Loewdin approach: %s" %(N))

### Calculate atomic partial charges ###
atom_charge = np.zeros(no_atoms)
MO_pop = np.zeros(no_atoms)
index_start = 0
index_end = 0
for i in range(no_atoms):
  index_end = index_end + no_basis_per_atom[i]
  atom_charge[i] = ZA[i] - diagonal_element[index_start:index_end].sum()
  MO_pop[i] = diagonal_element[index_start:index_end].sum()
  index_start = index_start + no_basis_per_atom[i]
print("Loewdin charges:")
print(atom_charge)

print("Loewdin MO population:")
MO_pop = MO_pop/2
print(MO_pop)
print(MO_pop.sum())
#np.savetxt('BTBT_HOMO_Population.txt', MO_pop)

