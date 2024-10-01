import json
import subprocess
import os

def run_cd_hit(input_file, output_file, identity=0.95):
    try:
        subprocess.run(['../../cdhit/cd-hit', '-i', input_file, '-o', output_file, '-c', str(identity)], check=True)
        print(f"CD-HIT completed. Filtered sequences saved in {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during CD-HIT: {e}")

with open("../../data/catalytic_residues_homologues.json") as f:
    data = json.load(f)

homologs = {}

for enzyme in data:
    residue_sequences = enzyme["residue_sequences"]
    uniprot_ids = []
    for residue in residue_sequences:
        if residue["is_reference"]:
            reference = residue["uniprot_id"]
            continue
        uniprot_ids.append(residue["uniprot_id"])
    homologs[reference] = uniprot_ids

for enzyme, enzyme_homologs in homologs.items():
    fasta_filename = f'../../data/fastas/enzyme_{enzyme}.fasta'
    output_filename = f'../../data/fastas_filtered/enzyme_{enzyme}.fasta'
    if os.path.exists(fasta_filename):
        run_cd_hit(fasta_filename, output_filename)
    else:
        print(f"FASTA file {fasta_filename} does not exist. Skipping.")

print("Done")

