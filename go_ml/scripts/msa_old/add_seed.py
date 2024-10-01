import os
import requests
from Bio import SeqIO
from io import StringIO

def fetch_protein_sequence(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise ValueError(f"Error fetching sequence for UniProt ID {uniprot_id}: {response.status_code}")

fasta_dir = "../../data/fastas_filtered_len"
fasta_filtered_length = "../../data/fastas_filtered_with_ref"

os.makedirs(fasta_filtered_length, exist_ok=True)

for fasta_file in os.listdir(fasta_dir):
    if fasta_file.endswith(".fasta"):
        uniprot_id = fasta_file.split('_')[1].split('.')[0]
        fasta_path = os.path.join(fasta_dir, fasta_file)
        output_fasta = os.path.join(fasta_filtered_length, fasta_file)

        try:
            fetched_sequence = fetch_protein_sequence(uniprot_id)
            fetched_seq_record = list(SeqIO.parse(StringIO(fetched_sequence), "fasta"))[0]
            existing_sequences = list(SeqIO.parse(fasta_path, "fasta"))
            all_sequences = [fetched_seq_record] + existing_sequences
            SeqIO.write(all_sequences, output_fasta, "fasta")
            print(f"Processed {fasta_file}")
        except ValueError as e:
            print(f"Error processing {fasta_file}: {e}")
        except IndexError:
            print(f"No valid sequence found for UniProt ID {uniprot_id} in {fasta_file}")

print("Done")

