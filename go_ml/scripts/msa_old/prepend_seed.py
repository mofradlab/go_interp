import json
import requests
import os

def fetch_fasta(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return f"Error: {response.status_code} - {response.text}"

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
    if os.path.exists(fasta_filename):
        with open(fasta_filename, 'r') as infile:
            existing_fasta = infile.read()

        reference_fasta = fetch_fasta(enzyme)
        if "Error" in reference_fasta:
            print(f"Error fetching reference sequence for {enzyme}: {reference_fasta}")
            continue

        with open(fasta_filename, 'w') as outfile:
            outfile.write(reference_fasta + existing_fasta)
    else:
        print(f"FASTA file {fasta_filename} does not exist. Skipping.")

print("Done")

