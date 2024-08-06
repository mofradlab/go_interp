import json
import os
from Bio import SeqIO

# Ensuring not missing proteins from batch downlaoding
# If the difference printed is positive, we are missing proteins 
# Negative means some of the uniprot IDs had mulitple sequences

with open("../../data/catalytic_residues_homologues.json") as f:
    data = json.load(f)

homolog_counts = {}
for enzyme in data:
    residue_sequences = enzyme["residue_sequences"]
    uniprot_ids = []
    reference_id = None
    for residue in residue_sequences:
        if residue["is_reference"]:
            reference_id = residue["uniprot_id"]
        else:
            uniprot_ids.append(residue["uniprot_id"])
    if reference_id:
        homolog_counts[reference_id] = len(uniprot_ids) + 1

fasta_dir = "../../data/msa_old/fastas/"
results = []

for fasta_file in os.listdir(fasta_dir):
    if fasta_file.endswith(".fasta"):
        enzyme_id = fasta_file.split('_')[1].split('.')[0]
        fasta_path = os.path.join(fasta_dir, fasta_file)
        fasta_sequences = list(SeqIO.parse(fasta_path, "fasta"))
        fasta_count = len(fasta_sequences)
        homolog_count = homolog_counts.get(enzyme_id, None)
        
        if homolog_count is not None:
            match = fasta_count == homolog_count
            results.append((enzyme_id, fasta_count, homolog_count, match))

for enzyme_id, fasta_count, homolog_count, match in results:
    print(f"Enzyme: {enzyme_id}, Homolog - FASTA Count: {homolog_count - fasta_count}")

print("Comparison complete.")

