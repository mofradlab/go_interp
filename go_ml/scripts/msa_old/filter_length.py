from Bio import SeqIO
import numpy as np
import os

def filter_length(input_fasta, output_fasta):
    sequences = list(SeqIO.parse(input_fasta, "fasta"))

    if not sequences:
        print("Fasta empty")
        return 


    lengths = [len(seq.seq) for seq in sequences]
    max_length = max(lengths)

    print(f"{input_fasta} max length: {max_length}")

    # assumption that the first length is the reference length 
    # double check this later

    reference_length = len(sequences[0].seq)
    print(reference_length)
    lengths = [len(seq.seq) for seq in sequences]
    two_SDs = np.std(lengths) * 2
    print(two_SDs)

    filtered_sequences = [seq for seq in sequences if abs(len(seq.seq) - reference_length) <= two_SDs]

    SeqIO.write(filtered_sequences, output_fasta, 'fasta')

fasta_dir = "../../data/fastas_filtered"
fasta_filtered_length = "../../data/fastas_filtered_len"

for fasta_file in os.listdir(fasta_dir):
    if fasta_file.endswith(".fasta"):
        fasta_path = os.path.join(fasta_dir, fasta_file)
        output_fasta = os.path.join(fasta_filtered_length, fasta_file)
        filter_length(fasta_path, output_fasta)

print("Done")



