import json
import requests
import os
import subprocess
import numpy as np
from Bio import SeqIO
from io import StringIO

with open("../../data/catalytic_residues_homologues.json") as f:
    data = json.load(f)

homologs = {}
for enzyme in data:
    residue_sequences = enzyme["residue_sequences"]
    uniprot_ids = []
    reference = None
    for residue in residue_sequences:
        if residue["is_reference"]:
            reference = residue["uniprot_id"]
        else:
            uniprot_ids.append(residue["uniprot_id"])
    if reference:
        homologs[reference] = uniprot_ids

def batch_fetch(uniprot_ids, batch_size=25):
    fasta_sequences = ""
    for i in range(0, len(uniprot_ids), batch_size):
        batch = uniprot_ids[i:i+batch_size]
        url = "https://rest.uniprot.org/uniprotkb/search"
        headers = {
            "Accept": "text/x-fasta"
        }
        query = ' OR '.join(batch)
        params = {
            'query': query,
            'format': 'fasta'
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            fasta_sequences += response.text
        else:
            print(f"Error: {response.status_code} - {response.text}")
    return fasta_sequences

def run_cd_hit(input_file, output_file, identity=0.95):
    try:
        subprocess.run(['../../cdhit/cd-hit', '-i', input_file, '-o', output_file, '-c', str(identity)], check=True)
        print(f"CD-HIT completed. Filtered sequences saved in {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during CD-HIT: {e}")

def fetch_protein_sequence(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise ValueError(f"Error fetching sequence for UniProt ID {uniprot_id}: {response.status_code}")


# If no longer, in file after CD-Hit, need to add seed 
def add_seed(input_file, seed_id):
    existing_sequences = list(SeqIO.parse(input_file, "fasta"))
    if not any(seed_id in record.id for record in existing_sequences):
        try:
            fetched_sequence = fetch_protein_sequence(seed_id)
            fetched_seq_record = list(SeqIO.parse(StringIO(fetched_sequence), "fasta"))[0]
            existing_sequences = list(SeqIO.parse(input_file, "fasta"))
            all_sequences = [fetched_seq_record] + existing_sequences
            SeqIO.write(all_sequences, input_file, "fasta")
            print(f"Processed {fasta_file}")
        except ValueError as e:
            print(f"Error processing {input_file}: {e}")


def filter_length(input_fasta, output_fasta, seed_id):
    fasta = list(SeqIO.parse(input_fasta, "fasta"))
    if not fasta:
        print(f"{input_fasta} is empty")
        return

    seed_len = None
    for record in fasta:
        if seed_id in record.id:
            seed_len = len(record.seq)
            break

    if seed_len is None:
        print(f"Reference sequence not found in {input_fasta}")
        return

    lengths = [len(seq.seq) for seq in fasta]
    two_SDs = np.std(lengths) * 2

    filtered_sequences = [seq for seq in fasta if abs(len(seq.seq) - seed_len) <= two_SDs]
    SeqIO.write(filtered_sequences, output_fasta, 'fasta')

i = 0
for enzyme, enzyme_homologs in homologs.items():
    try:
        uniprot_ids = [enzyme] + enzyme_homologs
        fasta = batch_fetch(uniprot_ids)
        
        fasta_filename = f'../../data/msa_files/fastas/enzyme_{enzyme}.fasta'
        fasta_cd_hit = f'../../data/msa_files/fastas_cd_hit/enzyme_{enzyme}.fasta'
        fasta_len = f'../../data/msa_files/fastas_len/enzyme_{enzyme}.fasta'
        
        with open(fasta_filename, 'w') as outfile:
            outfile.write(fasta)
        
        num_homologs = len(uniprot_ids)
        num_sequences = fasta.count('>')
        print(f"Enzyme {i}: {num_homologs} homologs (including reference), {num_sequences} sequences included in FASTA")
        
        run_cd_hit(fasta_filename, fasta_cd_hit)
        add_seed(fasta_cd_hit, enzyme)
        filter_length(fasta_cd_hit, fasta_len, enzyme)
        
        i += 1
    except ValueError as e:
        print(f"Error processing enzyme {enzyme}: {e}")

fasta_dir = "../../data/msa_files/fastas_len/"
msa_dir = "../../data/msa_files/msa/"

def run_muscle(input_fasta, output_aligned):
    muscle_args = ["/home/andrew/GO_interp/muscle", "-super5", input_fasta, "-output", output_aligned]
    subprocess.run(muscle_args, check=True)

for fasta_file in os.listdir(fasta_dir):
    if fasta_file.endswith(".fasta"):
        fasta_path = os.path.join(fasta_dir, fasta_file)
        output_aligned = os.path.join(msa_dir, f"{os.path.splitext(fasta_file)[0]}_aligned.fasta")
        run_muscle(fasta_path, output_aligned)
        print(f"Alignment done for {fasta_file}")

print("Finished obtaining MSAs")
