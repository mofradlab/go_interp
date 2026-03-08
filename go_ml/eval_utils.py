"""
eval_utils.py — Evaluation metrics and data utilities for residue-level prediction.

All evaluation functions operate on padded matrix representations:
  - score_mat  (N, L)      — per-residue scores (higher = more likely functional)
  - seq_len_mask (N, L)    — True at valid (non-padded) positions
  - annot_mat  (N, L)      — True at annotated functional residue positions

Sequences are 1-indexed in these matrices (position 0 is unused / reserved for BOS).
Maximum sequence length is 850 AA throughout.

Public API (used in eval_models.ipynb and cond_bert_gen_esmc.py):
  filter_annot_df     — load and validate a dataset CSV
  gen_annot_mat       — build annotation matrix from AnnotatedIndices column
  gen_seq_len_mask    — build sequence length mask from sequence strings
  gen_bert_mat        — pack per-protein logit dicts into a padded matrix
  get_bert_entropy    — per-position Shannon entropy from AA probability matrix
  get_pssm_entropy    — per-position entropy from PSSM (MSA baseline)
  mean_reciprocal_rank_mat — MRR: rank of the best-ranked annotated residue
  mean_auc            — mean per-protein ROC AUC
  bulk_auc            — single AUC pooling all residues across proteins
  top_30_score        — fraction of annotated residues in top-30 predictions
  roc_average         — interpolated mean ROC curve across proteins
"""

import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score
from scipy.stats import percentileofscore
import transformers
from collections import defaultdict
from Bio import SeqIO
import os

vocab = transformers.AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D").get_vocab()
vocab = {k: v-4 for i, (k, v) in enumerate(vocab.items()) if i >= 4 and i < 24}  # Filter to single-character tokens
vocab['-'] = 20
vocab = defaultdict(lambda: 20, vocab)  # Default to 20 for unknown characters

def load_msa_dict(prot_id_l, msa_dir):
    msa_map = {}
    for prot_id in prot_id_l:
        fasta_path = os.path.join(msa_dir, f"{prot_id}_homologues_aligned.fasta")
        if not os.path.exists(fasta_path):
            print(f"Warning: MSA file {fasta_path} does not exist.")
            continue
        sequences = list(SeqIO.parse(fasta_path, "fasta"))
        msa_map[prot_id] = sequences
    return msa_map

def get_pssm_entropy(pssm_mat, seq_len_mask, dash_override=True):
    dash_mean = pssm_mat[:, :, -1]
    entropy = (-np.log(pssm_mat[:,:, :]+1e-6) * pssm_mat[:,:, :]).sum(axis=2)
    #set values over inverse of mask to 3.0
    entropy[~seq_len_mask.astype(bool)] = 3.0
    if(dash_override):
        entropy[dash_mean > 0.3] = 3
    return entropy

#sequences = list(SeqIO.parse(msa_path, "fasta"))
def msa_to_pssm(msa_sequences, seed_seq_id):
    seed_seq = [str(record.seq) for record in msa_sequences if seed_seq_id in record.id]
    if not seed_seq:
        print(f"Warning: Seed sequence {seed_seq_id} not found in MSA.")
    seed_seq = seed_seq[0]
    num_sequences = len(msa_sequences)
    sequences_array = np.array([[20] + [vocab[s] for s in str(record.seq)] for record in msa_sequences])
    seed_seq_mask = np.array([s != '-' for s in 'A' + seed_seq])
    sequences_array = sequences_array[:, seed_seq_mask]
    msa_array = np.eye(21)[sequences_array].sum(axis=0)
    msa_array = msa_array / msa_array.sum(axis=1, keepdims=True)
    return msa_array

def gen_pssm_mat(prot_id_l, msa_map, max_len=850, return_msa=False):
    pssm_mat = np.zeros((len(prot_id_l), max_len, 21), dtype=np.float32)
    # print(pssm_mat.shape)
    for i, prot_id in enumerate(prot_id_l):
        if prot_id in msa_map:
            pssm = msa_to_pssm(msa_map[prot_id], prot_id)
            # print(pssm.shape, 'pssm')
            pssm_mat[i, :pssm.shape[0], :] = pssm
    return (pssm_mat, msa_map) if return_msa else pssm_mat

def gen_bert_mat(prot_id_l, bert_map, max_len=850):
    pssm_mat = np.zeros((len(prot_id_l), max_len, 20), dtype=np.float32)
    # print(pssm_mat.shape)
    for i, prot_id in enumerate(prot_id_l):
        if prot_id in bert_map:
            pssm = bert_map[prot_id]
            # print(pssm.shape, 'pssm')
            pssm_mat[i, :pssm.shape[0], :] = pssm
    return pssm_mat

def get_bert_entropy(bert_mat, seq_len_mask):
    entropy = (-np.log(bert_mat+1e-6) * bert_mat).sum(axis=2)
    #set values over inverse of mask to 3.0
    entropy[~seq_len_mask.astype(bool)] = 3.0
    return entropy

def gen_logit_map(prot_id_l, logit_map, max_len=850):
    score_mat = np.zeros((len(prot_id_l), max_len, logit_map[prot_id_l[0]].shape[-1]), dtype=np.float32)
    for i, prot_id in enumerate(prot_id_l):
        if prot_id in logit_map:
            prot_scores = logit_map[prot_id]
            score_mat[i, :prot_scores.shape[0], :] = prot_scores
    return score_mat

from go_ml.data_utils import gen_annot_mat

def gen_seq_len_mask(sequences, max_len=850):
    mask_mat = np.zeros((len(sequences), max_len), dtype=bool)
    for i, seq in enumerate(sequences):
        mask_mat[i, 1:len(seq)+1] = 1 #Sequences are 1-indexed
    return mask_mat

import ast
def filter_annot_df(annot_df, max_seq_len=850):
    """Load and validate an evaluation dataset CSV.

    Drops rows with missing values or sequences > max_seq_len. Parses
    AnnotatedIndices and GOTerm from string representation. Removes proteins
    where no annotated residues fall within the sequence, or where >75% of
    all residues are annotated (likely a data artifact).
    """
    annot_df = annot_df.dropna()
    annot_df = annot_df[annot_df['Sequence'].str.len() <= max_seq_len]
    annot_df['AnnotatedIndices'] = annot_df['AnnotatedIndices'].apply(ast.literal_eval)
    annot_df['GOTerm'] = annot_df['GOTerm'].apply(ast.literal_eval)
    annot_mat = gen_annot_mat(annot_df['AnnotatedIndices'], [len(s) for s in annot_df['Sequence']])
    seq_len_mask = gen_seq_len_mask(annot_df['Sequence'], max_len=max_seq_len)
    has_annot = (annot_mat*seq_len_mask).sum(axis=1) > 0
    annot_full = annot_mat.sum(axis=1) >= 0.75*annot_df['Sequence'].str.len()
    annot_df = annot_df[has_annot & ~annot_full]
    return annot_df

def auc_score(token_attribution, token_attribution_mask, conserved_token_mat):
    fpr_l, tpr_l, auc_l = [], [], []
    for r in range(token_attribution.shape[0]):
        p_labels = conserved_token_mat[r, token_attribution_mask[r]]
        p_attribution = token_attribution[r, token_attribution_mask[r]]
        fpr, tpr, thresholds = roc_curve(p_labels, p_attribution)
        roc_auc = auc(fpr, tpr)
        auc_l.append(roc_auc)
    # fpr_l = np.array(fpr_l); tpr_l = np.array(tpr_l); 
    auc_l = np.array(auc_l)
    return auc_l.mean()

from sklearn.metrics import f1_score, precision_recall_fscore_support
def bulk_auc(token_attribution, token_attribution_mask, conserved_token_mat):
    """Single ROC AUC pooling all residues across all proteins.

    Flattens all valid residues into one binary classification problem
    (annotated vs. non-annotated) and computes one AUC. Less sensitive to
    per-protein variance than mean_auc; useful as a sanity check.
    """
    mask = token_attribution_mask.flatten()
    labels = conserved_token_mat.flatten()[mask]
    preds = (token_attribution.flatten())[mask]
    fpr, tpr, thresholds = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def top_30_score(token_attribution, token_attribution_mask, conserved_token_mat):
    """Fraction of annotated residues recovered in the top-30 predictions.

    For each protein, takes the top min(30, n_annotated) highest-scoring
    residues and measures what fraction are annotated. Returns the mean
    across proteins. Higher is better; range [0, 1].
    """
    attribution_argsort = np.argsort(token_attribution - 1e5*(~token_attribution_mask), axis=1)[:, ::-1]
    top_conserve_stat = conserved_token_mat[np.arange(0, conserved_token_mat.shape[0]).reshape((-1, 1)), attribution_argsort]
    annot_counts = conserved_token_mat.sum(axis=1, keepdims=False)
    row_norm = np.minimum(30, annot_counts).astype(float)
    conserve_counts = top_conserve_stat[:, :30].sum(axis=1).astype(float)
    # print(row_norm.shape, conserve_counts.shape)
    conserve_counts /= row_norm
    return conserve_counts.mean()


def mean_reciprocal_rank(token_attribution, token_attribution_mask, conserved_tokens):
    attribution_argsort = torch.argsort(token_attribution - 1e5*token_attribution_mask, dim=1, descending=True)
    attribution_ranks = torch.argsort(attribution_argsort, dim=1, descending=False)
    ttr = 0
    tct = 0
    for i, token_ind in enumerate(conserved_tokens):
        token_ranks = attribution_ranks[i, token_ind]
        # print(token_ranks.min())
        ttr += 1 / (token_ranks.min()+1)
        tct += 1
    return ttr / tct

def mean_reciprocal_rank_mat(token_attribution, token_attribution_mask, conserved_token_mat):
    """Mean Reciprocal Rank (MRR) of annotated residues.

    For each protein, ranks all valid residues by score (descending) and
    records the rank of the highest-ranked annotated residue. Returns the
    mean of 1/rank across all proteins. Higher is better; range (0, 1].
    """
    attribution_argsort = np.argsort(token_attribution - 1e5*(~token_attribution_mask), axis=1)[:, ::-1]
    attribution_ranks = np.argsort(attribution_argsort, axis=1)

    # print(token_attribution)
    # print(attribution_argsort)
    # print(attribution_ranks)

    ttr = 0
    tct = 0
    for i in range(conserved_token_mat.shape[0]):
        token_ranks = attribution_ranks[i][conserved_token_mat[i]]
        if(token_ranks.shape[0] <= 0):
            continue
        ttr += 1 / (token_ranks.min()+1)
        tct += 1
    return ttr / tct

# def mean_reciprocal_rank(token_attribution, conserved_tokens):
#     attribution_argsort = torch.argsort(token_attribution, dim=1, descending=True)
#     attribution_ranks = torch.argsort(attribution_argsort, dim=1, descending=False)
#     ttr = 0
#     tct = 0
#     for i, token_ind in enumerate(conserved_tokens):
#         token_ranks = attribution_ranks[i, token_ind]
#         ttr += torch.divide(1, token_ranks+1).sum()
#         tct += token_ranks.shape[0]
#     return ttr / tct

def mean_percent_rank(token_attribution, seq_len, conserved_tokens):
    attribution_argsort = torch.argsort(token_attribution, dim=1, descending=True)
    attribution_ranks = torch.argsort(attribution_argsort, dim=1, descending=False)
    attribution_percent_rank = attribution_ranks / seq_len.reshape(-1, 1)
    ttr = 0
    tct = 0
    for i, token_ind in enumerate(conserved_tokens):
        token_ranks = attribution_percent_rank[i, token_ind]
        ttr += token_ranks.sum()
        tct += token_ranks.shape[0]
    return ttr / tct

def roc_stats(token_attribution: np.ndarray, seq_len: np.ndarray, conserved_tokens):
    token_labels = np.zeros_like(token_attribution)
    for r in range(len(conserved_tokens)):
        token_labels[r, conserved_tokens[r]] = 1
    fpr_l, tpr_l, auc_l = [], [], []
    for r in range(token_attribution.shape[0]):
        p_labels = token_labels[r, :seq_len[r]]
        p_attribution = token_attribution[r, :seq_len[r]]
        fpr, tpr, thresholds = roc_curve(p_labels, p_attribution)
        roc_auc = auc(fpr, tpr)
        auc_l.append(roc_auc)
    # fpr_l = np.array(fpr_l); tpr_l = np.array(tpr_l); 
    auc_l = np.array(auc_l)
    return auc_l.mean()

def mean_average_precision(token_attribution: np.ndarray, seq_len: np.ndarray, conserved_tokens):
    token_labels = np.zeros_like(token_attribution)
    for r in range(len(conserved_tokens)):
        token_labels[r, conserved_tokens[r]] = 1
    ap_l = [] 
    for r in range(len(conserved_tokens)):
        p_labels = token_labels[r, :seq_len[r]]
        p_attribution = token_attribution[r, :seq_len[r]]
        ap = average_precision_score(p_labels, p_attribution)
        ap_l.append(ap)
    ap_l = np.array(ap_l)
    return ap_l.mean()

def mean_auc(token_attribution: np.ndarray, token_attribution_mask: np.ndarray, conserved_tokens, return_roc=False):
    """Mean per-protein ROC AUC for annotated-residue prediction.

    Computes a per-protein binary classification ROC AUC (annotated vs.
    non-annotated residues) and returns the mean. Proteins with no annotated
    residues are skipped. If return_roc=True, also returns (fpr_list, tpr_list,
    auc_list) for plotting.
    """
    token_labels = np.zeros_like(token_attribution)
    for r in range(len(conserved_tokens)):
        token_labels[r, conserved_tokens[r]] = 1
    fpr_l, tpr_l, auc_l = [], [], []
    for r in range(token_attribution.shape[0]):
        if(token_labels[r].sum() <= 0):
            continue
        p_labels = token_labels[r, token_attribution_mask[r]]
        p_attribution = token_attribution[r, token_attribution_mask[r]]
        fpr, tpr, thresholds = roc_curve(p_labels, p_attribution)
        roc_auc = auc(fpr, tpr)
        if(np.isnan(roc_auc)):
            print(f"Warning: NaN AUC for row {r}. Skipping.")
            continue
        fpr_l.append(fpr)
        tpr_l.append(tpr)
        auc_l.append(roc_auc)
    auc_l = np.array(auc_l)
    if return_roc:
        return auc_l.mean(), (fpr_l, tpr_l, auc_l)
    return auc_l.mean()

def roc_average(fpr_l, tpr_l):
    all_fpr = np.unique(np.concatenate(fpr_l))  # Combine and sort all unique FPR values
    mean_tpr = np.zeros_like(all_fpr) # Initialize an array for storing the mean TPR values
    for i in range(len(fpr_l)):
        # Interpolate each TPR curve at the common FPR points
        mean_tpr += np.interp(all_fpr, fpr_l[i], tpr_l[i])  
    mean_tpr /= len(fpr_l) # Calculate the mean TPR
    return all_fpr, mean_tpr

# def mean_auc(token_attribution: np.ndarray, seq_len: np.ndarray, conserved_tokens):
#     token_labels = np.zeros_like(token_attribution)
#     for r in range(len(conserved_tokens)):
#         token_labels[r, conserved_tokens[r]] = 1
#     fpr_l, tpr_l, auc_l = [], [], []
#     for r in range(token_attribution.shape[0]):
#         p_labels = token_labels[r, :seq_len[r]]
#         p_attribution = token_attribution[r, :seq_len[r]]
#         fpr, tpr, thresholds = roc_curve(p_labels, p_attribution)
#         roc_auc = auc(fpr, tpr)
#         auc_l.append(roc_auc)
#     auc_l = np.array(auc_l)
#     return auc_l.mean()