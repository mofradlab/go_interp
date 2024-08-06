from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# TODO: NORMALIZE OUTPUT
# Concatenate dataset
# fit_transform on whole dataset
# TODO: Possible fit_transfrom on mutants

def embedding(sequence: str) -> np.array:
    sequence_embeddings_list = []

    for residue in sequence:
        amino_acid = ProteinAnalysis(residue)
        aromaticity = amino_acid.aromaticity()
        isoelectric = amino_acid.isoelectric_point()
        molecular_weight = amino_acid.molecular_weight()

        sequence_embeddings_list.append([aromaticity, isoelectric, molecular_weight])
    sequence_embeddings = np.array(sequence_embeddings_list)

    # Normalize:
    sequence_embeddings = StandardScaler().fit_transform(sequence_embeddings)
    return sequence_embeddings

    for i