# Function-Conditioned Protein Language Models

Semi-supervised identification of functionally important residues using
GO-term-conditioned masked language modeling.

## Overview

Protein language models (PLMs) like ESM assign per-residue token probabilities.
Positions with lower entropy (higher model confidence) tend to be more conserved
and functionally constrained. This work tests whether explicitly conditioning a PLM
on a protein's Gene Ontology (GO) functional annotations sharpens residue-level
predictions at known functional sites (active sites, binding sites, motifs, domains).

**Model:** `FuncCondESMC` extends ESM Cambrian with a learned embedding
for each GO term. These embeddings are projected into the token embedding space and
added before the transformer encoder, conditioning all attention layers on the
protein's annotated function.

**Evaluation:** Models are evaluated on 8 site-annotation datasets spanning catalytic
residues (CSA), phase-separation regions (LLPS), linear motifs (ELMs), binding sites
(BioLiP), and InterPro domains/active sites/binding sites/repeats. Metrics include
mean reciprocal rank (MRR), mean AUC, and top-30 conservation score, compared against
MSA entropy, attribution methods, and unconditioned baselines.

## Repository Structure

```
GO_interp/
├── go_ml/
│   ├── models/
│   │   ├── func_cond_esmc.py       # ESMC conditioned model (main)
│   │   └── func_cond_esm.py        # ESM2 conditioned model
│   ├── scripts/
│   │   ├── train_func_cond_esmc.py # Training entry point
│   │   ├── train_func_cond.py      # ESM2 training entry point
│   │   └── train.sh                # Exact commands for paper checkpoints
│   ├── dataset_eval/
│   │   ├── datasets/               # 8 evaluation dataset CSVs
│   │   ├── eval_files/             # Precomputed model outputs (pkl)
│   │   ├── cond_bert_gen_esmc.py   # Generate ESMC-cond logits
│   │   ├── cond_bert_gen_esm2.ipynb
│   │   ├── bert_gen*.ipynb         # Baseline logit generation
│   │   ├── msa_gen.ipynb           # MSA PSSM baseline
│   │   ├── annot_bert_gen.ipynb    # ProteinInfer baseline
│   │   ├── run_eval.sh             # Regenerate all eval_files/
│   │   └── eval_models.ipynb       # Main results notebook
│   ├── msa_pipeline/               # MSA homolog search pipeline
│   │   └── README.md               # Step-by-step MSA instructions
│   ├── model_interp/
│   │   ├── gen_interp.ipynb        # Attribution methods (lga_attr, ldl_attr)
│   │   └── attention_interp.ipynb  # Attention-based attribution
│   ├── gen_datasets/
│   │   ├── README.md               # Dataset sources and column schema
│   │   └── dataset_*.ipynb         # One notebook per dataset
│   ├── data_utils.py               # Dataset classes and tokenization
│   ├── eval_utils.py               # Evaluation metrics
│   ├── masking.py                  # Masking strategies and logit generation
│   └── train_utils.py              # LR scheduler and data loaders
├── checkpoints/                    # 5 trained ESMC checkpoints
├── data/
│   ├── go-basic.obo                # GO ontology
│   ├── go_terms.json               # GO term list (from CAFA5)
│   ├── train_esm_datasets/         # Pre-split training/val pickles
│   └── [large annotation files]   # goa_uniprot_all.gaf, processed_gaf.csv, etc.
└── archive/                        # Old code, unused checkpoints, large data
```

## Installation

```bash
git clone <repo>
cd GO_interp
pip install -r requirements.txt
pip install -e .
```

The project also depends on `go_bench` (a local package for GO evaluation utilities):
```bash
pip install -e git+https://github.com/amdson/go_bench.git@950aa81#egg=go_bench
```

ESM Cambrian requires a separate install:
```bash
pip install esm
```

## Data Setup

The training data pickles (`data/train_esm_datasets/`) and large GO annotation files
(`data/goa_uniprot_all.gaf`, `data/processed_gaf.csv`) are not tracked in git due to
size. Contact the authors for access, or rebuild from scratch using
`go_ml/gen_datasets/` notebooks. 

Evaluation datasets are small CSVs in `go_ml/dataset_eval/datasets/` and are tracked
in git. See `go_ml/gen_datasets/README.md` for construction details.

## Training

```bash
cd go_ml/scripts/

# Train a single configuration
python train_func_cond_esmc.py \
    --gpu_id 0 \
    --mask_func span \
    --context_length 100 \
    --span_mask_length 5 \
    --data_dir ../../data/train_esm_datasets/ \
    --output_dir ../../checkpoints/

# Reproduce all 5 paper checkpoints in order
bash train.sh 0   # pass GPU id
```

See `scripts/train.sh` for the checkpoint→hyperparameter mapping.

## Evaluation

Precomputed outputs for all models are in `go_ml/dataset_eval/eval_files/`.
To reproduce them from scratch:

```bash
cd go_ml/dataset_eval/

# Regenerate ESMC-cond outputs (all 5 configs) + instructions for all baselines
bash run_eval.sh 0   # pass GPU id

# View results
jupyter notebook eval_models.ipynb
```

For the MSA baseline, run the pipeline in `go_ml/msa_pipeline/` first
(see `msa_pipeline/README.md`).

## Attribution Methods

Gradient-based and attention-based attributions (`lga_attr`, `ldl_attr`) are generated by
`go_ml/model_interp/gen_interp.ipynb` and `attention_interp.ipynb`, respectively.

## Checkpoints

Five ESMC checkpoints are provided (span masking, varying context/span length):

| Checkpoint | context_len | span_len |
|-----------|-------------|----------|
| `func_cond_finetune_esmc.ckpt` | 100 | 5 |
| `func_cond_finetune_esmc-v4.ckpt` | 200 | 5 |
| `func_cond_finetune_esmc-v5.ckpt` | 100 | 10 |
| `func_cond_finetune_esmc-v6.ckpt` | 100 | 2 |
| `func_cond_finetune_esmc-v7.ckpt` | 50 | 5 |

Checkpoints are not tracked in git (see `.gitignore`). Contact the authors for access.
