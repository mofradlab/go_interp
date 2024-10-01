import json
import requests
f = open("../../data/catalytic_residues_homologues.json")
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

f.close()

def fetch_fasta(uniprot_ids):
    url = "https://rest.uniprot.org/uniprotkb/search"
    headers = {
        "Accept": "text/x-fasta"
    }
    query = ' OR '.join(uniprot_ids)
    params = {
        'query': query,
        'format': 'fasta'
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.text
    else:
        return f"Error: {response.status_code} - {response.text}"

def batch_fetch(uniprot_ids, batch_size=25):
    fasta_sequences = ""
    for i in range(0, len(uniprot_ids), batch_size):
        batch = uniprot_ids[i:i+batch_size]
        batch_fasta = fetch_fasta(batch)
        # print(f"Batch {i // batch_size + 1} fetched with {batch_fasta.count('>')} sequences")
        fasta_sequences += batch_fasta
    return fasta_sequences

i = 0
for enzyme, enzyme_homologs in homologs.items():
    with open(f'../../data/fastas/enzyme_{enzyme}.fasta', 'w') as outfile:
        fasta = batch_fetch(enzyme_homologs)
        outfile.write(fasta)
        print(f"Enzyme {i}: {fasta.count('>')} homologs")
    i += 1

print("Done")

