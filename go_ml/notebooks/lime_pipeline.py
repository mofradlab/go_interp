import torch
import numpy as np

def mutate_ind(seq, key_ind, lambda_=15):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    mut_seq = np.array(list(seq))
    num_changes = np.random.poisson(lambda_)
    num_changes = min(num_changes, len(key_ind))
    indices_to_change = np.random.choice(len(key_ind), num_changes, replace=False)
    if num_changes == 1:
        indices_to_change = indices_to_change[0]
        mut_seq[key_ind[indices_to_change]] = np.random.choice(list(amino_acids)) 
    else:
        mut_seq[key_ind[indices_to_change]] = np.random.choice(list(amino_acids), num_changes)
    mut_seq = ''.join(mut_seq)
    return mut_seq, num_changes