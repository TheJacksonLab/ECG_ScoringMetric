import numpy as np

################ Set Parameters ################
molecule_name = "BTBT"    # Abreviation of molecule: two characters
huang_rhys_folder = "./"
huang_rhys_path = huang_rhys_folder + molecule_name + ".out"
gaussian_path = huang_rhys_folder + molecule_name + "_g.log"
MO_population_path = 'BTBT_HOMO_Population.txt'
CG_mapping_folder = "CG_mapping/"
CG_mapping_file = ['GBCG', 'CG_Random-1', 'CG_Random-2', 'CG_Improve'] 
frequency_criteria = [200] 

#metric_method = 'NM' #Normal mode displacement without any weighting 
metric_method = 'MO-NM' #Normal mode displacement weighted by HOMO density
#metric_method = 'HR-NM' #Normal mode displacement weighted by Huang-Rhys factor

no_toatal_atom = 62 #62    #total number of atoms per molecule  
normal_mode_criteria = 0.0 #Define the lower bondary of Huang-Rhys factor to selecte the effective normal mode 


################### Open Files ###################
def openfile(filename):
  f = open((filename), "r")
  full_content = f.readlines()
  f.close()
  full_content_numpy = np.array(full_content)
  print('Loading file: %s' %(filename))
  return full_content,full_content_numpy


################### Generate Random Vector  ###################
def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    np.random.seed(10)
    vec *= np.random.randn(1, npoints)*10.0
    return vec   
#print(sample_spherical(2).T)


################ Read Normal Mode Frequency Displacement ################
########## Identify the number of normal mode saved in Gaussian output file ##########
keyword = "Frequencies ---"
full_content,full_content_numpy = openfile(gaussian_path)
line_no = []
for i in range(len(full_content_numpy)):
  if (full_content[i].__contains__(keyword)):
    line_no.append(i)
no_save_mode = int(full_content[line_no[-1]-2].split()[-1])
#print(no_save_mode)

########## Read normal mode displacement & frequency from Gaussian output file ##########
freq_gaussian = np.zeros(no_save_mode,dtype=float)
mode_dis = np.zeros((no_save_mode,no_toatal_atom,3),dtype=float)
##### The major section ######
for index, i in np.ndenumerate(np.array(line_no)):
  #print(index[0],full_content[i-2])
  if(i!=line_no[-1]):
    for m in range(5):
      ### Frequency ###
      freq_gaussian[m+index[0]*5] = float(full_content[i].split()[m+2])
    
      ### normal mode displacement ###
      counter = 0
      for n in range(no_toatal_atom):
        mode_dis[(m+index[0]*5),counter,0] = float(full_content[i+5+n*3].split()[m+3]) # x component
        mode_dis[(m+index[0]*5),counter,1] = float(full_content[i+5+n*3+1].split()[m+3]) # y component
        mode_dis[(m+index[0]*5),counter,2] = float(full_content[i+5+n*3+2].split()[m+3]) # z component
        counter += 1
  elif(i==line_no[-1]):
    no_column = no_save_mode - index[0]*5
    for m in range(no_column):
      ### Frequency ###
      freq_gaussian[m+index[0]*5] = float(full_content[i].split()[m+2])
    
      ### normal mode displacement ###
      counter = 0
      for n in range(no_toatal_atom):
        mode_dis[(m+index[0]*5),counter,0] = float(full_content[i+5+n*3].split()[m+3]) # x component
        mode_dis[(m+index[0]*5),counter,1] = float(full_content[i+5+n*3+1].split()[m+3]) # y component
        mode_dis[(m+index[0]*5),counter,2] = float(full_content[i+5+n*3+2].split()[m+3]) # z component
        counter += 1
#print(freq_gaussian)
#print(mode_dis[-2,:,:])

################ Specify Atom Mass ################
########## Read atomic number ##########
keyword = " Center     Atomic      Atomic             Coordinates (Angstroms)"
line_no = []
for i in range(len(full_content_numpy)):
  if (full_content[i].__contains__(keyword)):
    line_no.append(i)
atom_number = []
for i in range(no_toatal_atom):
  atom_number.append(int(full_content[line_no[0]+3+i].split()[1]))
#print(atom_number)
########## Define atom mass arry ##########
atom_mass = []
for i in range(no_toatal_atom):
  if (atom_number[i]==6):    #C
    atom_mass.append(12.011)
  elif (atom_number[i]==1):  #H
    atom_mass.append(1.008)
  elif (atom_number[i]==16): #S
    atom_mass.append(32.060)    
  elif (atom_number[i]==8):  #O
    atom_mass.append(15.999)    
#print(atom_mass)


################ Read HOMO Population ################
MO_pop = np.loadtxt(MO_population_path)
#print(MO_pop.sum())

################ Read Normal Mode Frequency & Huang-Rhys factor ################
freq_all = np.zeros(no_toatal_atom*3, dtype=float)
lambda_all = np.zeros(no_toatal_atom*3, dtype=float)

########## Read charge state information: cation ##########
keyword = "Mode    Frequency (cm-1)             lambda+"
full_content,full_content_numpy = openfile(huang_rhys_path)
for i in range(len(full_content_numpy)):
  if (full_content[i].__contains__(keyword)):
    line_no = i
line_begin = line_no +2
line_end = line_begin + no_toatal_atom*3
counter = 0
for i in range(line_begin,line_end):
  freq_all[counter] = float(full_content[i].split()[1])
  lambda_all[counter] = float(full_content[i].split()[2])
  #print(freq_all[counter],lambda_all[counter])
  counter += 1  

########## Calculate Huang-Rhys factor ##########
freq_select = freq_all[-no_save_mode:]
if not(np.array_equal(freq_select.astype(int),freq_gaussian.astype(int))):
  raise ValueError("The frequencies obtained from Gaussian and Huang-Rhys analysis are not matched!")
HR_factor =  lambda_all[-no_save_mode:]**2
#print(HR_factor)
#print(freq_select)


########## Find the effective normal modes ##########
eff_mode_index = np.argwhere(HR_factor>normal_mode_criteria).flatten()
#print(HR_factor[eff_mode_index])
#print(freq_select[eff_mode_index].astype(int).flatten())
#print(freq_gaussian[eff_mode_index].astype(int).flatten())
#print(mode_dis[eff_mode_index,:,:])


################ Read Atom Index of CG Representation ################
score_sum = np.zeros((len(CG_mapping_file),len(frequency_criteria)))
save_CG_bead = []
for idx_CG_map, path in enumerate(CG_mapping_file):
  for idx_freq, freq in enumerate(frequency_criteria):
    CG_mapping_path = CG_mapping_folder + path + '.map'
    full_content,full_content_numpy = openfile(CG_mapping_path)
    map_index = []
    for line in full_content: 
      no_index = len(line.split())
      map_index.append(line.split()[4:no_index])
    no_CG_bead = len(map_index)
    #print(map_index)
    print("Total number of CG beads: %s\n" %no_CG_bead)
    save_CG_bead.append(no_CG_bead)

    ################ Read Number of Atoms per CG Bead ################
    no_atom_per_CG = []
    for i in range(no_CG_bead):
      no_atom_per_CG.append(len(map_index[i]))
    no_atom_per_CG = np.array(no_atom_per_CG)



    ################ Evaluate CG Mapping Score ################
    mode_dis_CG = np.zeros((no_CG_bead,3),dtype=float)
    #mode_dis_CG = np.zeros((no_CG_bead,1),dtype=float)
    CG_mapping_score = np.zeros(no_CG_bead,dtype=float)

    ########## Sum up all the displacement of effective mode to each CG bead ##########
    if (metric_method=='NM'):
      print('Evaluation metric: Normal mode displacement without any weighting')
    elif (metric_method=='HR-NM'):
      print('Evaluation metric: Normal mode displacement weighted by Huang-Rhys factor')
    elif (metric_method=='MO-NM'):
      print('Evaluation metric: Normal mode displacement weighted by HOMO density')
      
    for i in range(no_CG_bead):
      for m, mode_index in np.ndenumerate(eff_mode_index):
        #if (freq_select[mode_index]>1200 and freq_select[mode_index]<1800):
        if (freq_select[mode_index]<freq):
          #print(i,freq_select[mode_index])
          for j, atom_index in enumerate(map_index[i]):
            if((atom_mass[int(atom_index)-1])!=1.008): #Exclude the displacement of H aotms              
              # Displacement only
              if (metric_method=='NM'):
                mode_dis_CG[i,:] += mode_dis[mode_index,(int(atom_index)-1),:] #maping index starts from 1
              
              # Displacement multiplied by Huang-Rhys factor
              if (metric_method=='HR-NM'):
                mode_dis_CG[i,:] += mode_dis[mode_index,(int(atom_index)-1),:]* HR_factor[mode_index] #maping index starts from 1
              
              # MO population weighted displacement
              if (metric_method=='MO-NM'):
                mode_dis_CG[i,:] += mode_dis[mode_index,(int(atom_index)-1),:] * MO_pop[int(atom_index)-1]*20 #maping index starts from 1
              
              # MO population only
              #mode_dis_CG[i,:] +=  MO_pop[int(atom_index)-1] #maping index starts from 1
              

    ########## Calculate CG mapping score ##########
    #CG_mapping_score = np.abs(mode_dis_CG[:,0])
    CG_mapping_score = (mode_dis_CG[:,0]**2 + mode_dis_CG[:,1]**2 +mode_dis_CG[:,2]**2)**0.5
    CG_mapping_score = CG_mapping_score/no_atom_per_CG
    CG_mapping_score_sum = CG_mapping_score.sum()
    #print(CG_mapping_score)
    print('----- Score of each CG particle -----')
    for i in range(no_CG_bead):
      print(i+1,CG_mapping_score[i])
    CG_mapping_score_nonzero = CG_mapping_score[np.nonzero(CG_mapping_score)]
    print("Global CG score: %s" %CG_mapping_score_sum)
    score_sum[idx_CG_map,idx_freq] = CG_mapping_score_sum

print('\n----- Summary of Global CG score -----')
for i in range(len(CG_mapping_file)):
  print("%s: %s" %(CG_mapping_file[i],score_sum[i][0]))


