"""
cond_bert_gen_esmc.py — Generate logits for function-conditioned ESMC models.

For each evaluation dataset and each trained checkpoint, this script:
  1. Loads the dataset CSV and expands each protein's GO term to include ancestors
  2. Builds a GO label vector and runs span-masked inference with the conditioned model
  3. Normalizes logits to per-position AA probability distributions
  4. Saves bert_mat, seq_len_mask, and bert_entropy to a pkl file

Inputs
------
  --param_index   Index into the 5 (context_len, span_len) configurations (0–4)
                  0: (100, 2)   1: (100, 5)   2: (100, 10)   3: (50, 5)   4: (200, 5)
  --gpu_id        GPU index (default: 0)
  --data_dir      Dataset CSV directory (default: ../gen_datasets/datasets)
  --checkpoint_dir  Checkpoint directory (default: ../../checkpoints)
  --go_terms_path   Path to go_terms.json (default: ../../data/go_terms.json)
  --go_obo_path     Path to go-basic.obo  (default: ../../data/go-basic.obo)
  --eval_dir      Output directory for pkl files (default: eval_files)

Outputs
-------
  {eval_dir}/{dataset}/{path_label}.pkl
  where path_label = esmc_cond_span_{context_len}_{span_len}

Example
-------
  # Run from go_ml/dataset_eval/
  python cond_bert_gen_esmc.py --param_index 0 --gpu_id 0
  # Or run all 5 configs via the run_eval.sh wrapper
"""

import json, os, pickle
from argparse import ArgumentParser
from itertools import chain

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import goatools.obo_parser as obo

from go_ml.eval_utils import (filter_annot_df, gen_annot_mat, gen_bert_mat,
                               get_bert_entropy, gen_seq_len_mask)
from go_ml.masking import get_logits_cond, mask_span, SEQUENCE_MASK_TOKEN
from go_ml.models.func_cond_esmc import FuncCondESMCFinetune

# ── Arguments ────────────────────────────────────────────────────────────────

parser = ArgumentParser()
parser.add_argument("--gpu_id",        default=0,   type=int)
parser.add_argument("--param_index",   default=0,   type=int,
                    help="Index into param_list: 0=(100,2), 1=(100,5), 2=(100,10), 3=(50,5), 4=(200,5)")
parser.add_argument("--data_dir",      default="../gen_datasets/datasets",   type=str)
parser.add_argument("--checkpoint_dir",default="../../checkpoints",          type=str)
parser.add_argument("--go_terms_path", default="../../data/go_terms.json",   type=str)
parser.add_argument("--go_obo_path",   default="../../data/go-basic.obo",    type=str)
parser.add_argument("--eval_dir",      default="eval_files",                 type=str)
args = parser.parse_args()

# ── Hyperparameter configurations ─────────────────────────────────────────────
# Each tuple is (context_len, span_len) for the span masking strategy.
# context_len: radius of unmasked context visible around each masked span
# span_len:    number of consecutive tokens per masked span

param_list = [(100, 2), (100, 5), (100, 10), (50, 5), (200, 5)]
model_dict = {
    (100, 2):  "func_cond_finetune_esmc-v6.ckpt",
    (100, 5):  "func_cond_finetune_esmc.ckpt",
    (100, 10): "func_cond_finetune_esmc-v5.ckpt",
    (50,  5):  "func_cond_finetune_esmc-v7.ckpt",
    (200, 5):  "func_cond_finetune_esmc-v4.ckpt",
}

context_len, span_len = param_list[args.param_index]
checkpoint_path = os.path.join(args.checkpoint_dir, model_dict[(context_len, span_len)])
path_label = f"esmc_cond_span_{context_len}_{span_len}"

# ── Datasets ──────────────────────────────────────────────────────────────────

dataset_labels = ["csa", "llps", "elms", "biolip",
                  "ip_repeat", "ip_domain", "ip_binding_site", "ip_active_site"]
dataset_dfs = [
    filter_annot_df(pd.read_csv(f"{args.data_dir}/{label}_dataset.csv", sep="\t"))
    for label in dataset_labels
]

# ── GO ontology ───────────────────────────────────────────────────────────────

with open(args.go_terms_path) as f:
    go_terms = json.load(f)
go_dag = obo.GODag(args.go_obo_path)
go_ind_map = {term: i for i, term in enumerate(go_terms)}

def list_ancestors(term, godag):
    """Yield all ancestor GO terms of a given term."""
    if term not in godag:
        return
    ancestors = list(godag[term]._parents)
    seen = set()
    while ancestors:
        ancestor = ancestors.pop()
        if ancestor in seen:
            continue
        seen.add(ancestor)
        if ancestor in godag:
            ancestors.extend(godag[ancestor]._parents)
        yield ancestor

def expand_go_terms(go_terms_for_protein, go_dag):
    """Return the union of go_terms_for_protein and all their ancestors."""
    return (set(chain.from_iterable(list_ancestors(t, go_dag) for t in go_terms_for_protein))
            | set(go_terms_for_protein))

# ── Model ─────────────────────────────────────────────────────────────────────

device = torch.device(f"cuda:{args.gpu_id}")
model = FuncCondESMCFinetune.load_from_checkpoint(checkpoint_path, map_location=device)
model.eval()

vi = {i: a for a, i in model.tokenizer.get_vocab().items()}
span_mask_func = lambda seq, mask_token: mask_span(
    seq, mask_token,
    residue_coverage=5,
    span_rad=context_len // 2,
    run_len=span_len,
    mask_per=0.3,
)

# ── Inference ─────────────────────────────────────────────────────────────────

for ds_label, annot_df in zip(dataset_labels, dataset_dfs):
    save_path = os.path.join(args.eval_dir, ds_label, f"{path_label}.pkl")
    if os.path.exists(save_path):
        print(f"Skipping {ds_label} — {save_path} already exists")
        continue

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Building {save_path}")

    df_logits = {}
    for seq_id, seq, annot_term in tqdm(
        zip(annot_df["UniprotID"], annot_df["Sequence"], annot_df["GOTerm"]),
        total=len(annot_df), desc=ds_label,
    ):
        func_labels = torch.zeros(len(go_terms))
        for go_term in expand_go_terms(annot_term, go_dag):
            if go_term in go_ind_map:
                func_labels[go_ind_map[go_term]] = 1
        df_logits[seq_id] = get_logits_cond(
            seq, func_labels, model, batch_size=20, mask_func=span_mask_func
        ).float()

    # Slice AA token columns [4:24], normalize to probabilities
    bert_map = {k: v[:, 4:24] for k, v in df_logits.items()}
    bert_map = {k: v / (v.sum(dim=1, keepdim=True) + 1e-10) for k, v in bert_map.items()}
    bert_map = {k: v.numpy() for k, v in bert_map.items()}

    annot_mat    = gen_annot_mat(annot_df["AnnotatedIndices"], [len(s) for s in annot_df["Sequence"]])
    seq_len_mask = gen_seq_len_mask(annot_df["Sequence"])
    bert_mat     = gen_bert_mat(annot_df["UniprotID"], bert_map, max_len=850)
    bert_entropy = get_bert_entropy(bert_mat, seq_len_mask)

    with open(save_path, "wb") as f:
        pickle.dump({
            "UniprotID":    annot_df["UniprotID"],
            "bert_mat":     bert_mat,
            "seq_len_mask": seq_len_mask,
            "bert_entropy": bert_entropy,
        }, f)
    print(f"  Saved {ds_label}")
