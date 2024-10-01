import os
import subprocess

fasta_dir = "/home/andrew/GO_interp/data/fastas_filtered_with_ref/"
msa_dir = "/home/andrew/GO_interp/data/msa/"

def run_muscle(input_fasta, output_aligned):
    muscle_args = ["/home/andrew/GO_interp/muscle", "-super5", input_fasta, "-output", output_aligned]
    subprocess.run(muscle_args, check=True)

for fasta_file in os.listdir(fasta_dir):
    if fasta_file.endswith(".fasta"):
        fasta_path = os.path.join(fasta_dir, fasta_file)
        output_aligned = os.path.join(msa_dir, f"{os.path.splitext(fasta_file)[0]}_aligned.fasta")
        run_muscle(fasta_path, output_aligned)
        print(f"Alignment done for {fasta_file}")

print("Done")

