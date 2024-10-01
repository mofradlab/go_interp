from Bio import SeqIO
import numpy as np
import os

fasta_dir = "../../data/fastas_filtered"

for fasta_file in os.listdir(fasta_dir):
    if fasta_file.endswith(".fasta"):
        print('hi')
        fasta_path = os.path.join(fasta_dir, fasta_file)
        fasta = list(SeqIO.parse(fasta_path, "fasta"))
        
        if not fasta:
            print(f"{fasta_path} is empty")
            continue

        seed_id = fasta_file.split('_')[1].split('.')[0]
        seed_len = None
        for record in fasta:
            print(record.id)

        break
