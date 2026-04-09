# MSA Pipeline

Generates Multiple Sequence Alignment (MSA) baselines for the evaluation datasets.
MSA entropy is used as a position-conservation baseline in `dataset_eval/eval_models.ipynb`.

## Datasets

MSA baselines are computed for 4 datasets: **csa**, **llps**, **elms**, **ip_domain**

## Prerequisites

- [MMseqs2](https://github.com/soedinglab/MMseqs2) installed and on `PATH`
- [MUSCLE v5](https://github.com/rcedgar/muscle) binary available (used for alignment)
- UniRef sequence database built in `uniref_data/`:
  ```bash
  mmseqs createdb <path/to/uniref.fasta> uniref_data/uniref_db
  mmseqs makepaddedseqdb uniref_data/uniref_db uniref_data/uniref_db_gpu
  ```

## Pipeline Steps

### Step 1 — Extract dataset FASTAs
**Notebook:** `build_fasta.ipynb`

Reads each dataset CSV from `gen_datasets/datasets/` and writes a FASTA file.
- `csa`, `llps`, `elms`: sequence IDs are UniprotIDs
- `ip_domain`: sequence IDs are `ipdom{i}` (integer index) to avoid colon characters

**Outputs:** `../gen_datasets/datasets/{dataset}.fasta`

> Pre-computed FASTAs already exist. Only re-run if dataset CSVs change.

---

### Step 2 — Homolog search with MMseqs2
**Script:** `run_mmseqs2.sh`

Searches each dataset FASTA against the UniRef database to find homologous sequences.

```bash
# Run from go_ml/msa_pipeline/
bash run_mmseqs2.sh
```

**Outputs:** `{dataset}_aln.m8` — tab-separated alignment hits (one per dataset)

> Pre-computed `.m8` files already exist. Only re-run if datasets change.

---

### Step 3 — Parse hits into per-protein FASTAs
**Notebook:** `build_homologues.ipynb`

Filters alignment hits (≤95% identity, within 2 SD of mean length, max 450 homologs,
max 1500 AA) and writes one unaligned FASTA per query protein.

**Outputs:** `uniref_msa/{dataset}_msa/{protein_id}_homologues.fasta`

> Pre-computed files already exist in `uniref_msa/`.

---

### Step 4 — Align homologs with MUSCLE
For each per-protein FASTA, run MUSCLE to produce an aligned version:

```bash
# Example for a single protein
muscle -align uniref_msa/csa_msa/P12345_homologues.fasta \
       -output uniref_msa/csa_msa_output/P12345_homologues_aligned.fasta

# Batch alignment (all proteins in a dataset)
for f in uniref_msa/csa_msa/*.fasta; do
    name=$(basename "$f" .fasta)
    muscle -align "$f" -output "uniref_msa/csa_msa_output/${name}_aligned.fasta"
done
```

**Outputs:** `uniref_msa/{dataset}_msa_output/{protein_id}_homologues_aligned.fasta`

> Pre-computed aligned FASTAs already exist in `uniref_msa/`.

---

### Step 5 — Compute PSSM entropy and save results
**Notebook:** `dataset_eval/msa_gen.ipynb`

Loads aligned FASTAs, computes position-specific scoring matrices (PSSM) and
per-position entropy, saves to `eval_files/{dataset}/msa.pkl`.

Run from `go_ml/dataset_eval/`.

## Directory Structure

```
msa_pipeline/
├── build_fasta.ipynb          # Step 1
├── run_mmseqs2.sh             # Step 2
├── build_homologues.ipynb     # Step 3
├── msa_entropy.ipynb          # Exploratory MSA entropy analysis
├── {dataset}_aln.m8           # MMseqs2 alignment results (Step 2 output)
├── query_db/                  # MMseqs2 query databases (intermediate)
├── results_db/                # MMseqs2 result databases (intermediate)
├── tmp/                       # MMseqs2 temp files
├── uniref_data/               # UniRef MMseqs2 database
└── uniref_msa/
    ├── {dataset}_msa/         # Unaligned homolog FASTAs (Step 3 output)
    └── {dataset}_msa_output/  # Aligned homolog FASTAs (Step 4 output)
```
