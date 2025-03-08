import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score
from scipy.stats import percentileofscore


def mean_reciprocal_rank_corrected(token_attribution, conserved_tokens):
    attribution_argsort = torch.argsort(token_attribution, dim=1, descending=True)
    attribution_ranks = torch.argsort(attribution_argsort, dim=1, descending=False)
    ttr = 0
    tct = 0
    for i, token_ind in enumerate(conserved_tokens):
        token_ranks = attribution_ranks[i, token_ind]
        # print(token_ranks.min())
        ttr += 1 / (token_ranks.min()+1)
        tct += 1
    return ttr / tct

def mean_reciprocal_rank(token_attribution, conserved_tokens):
    attribution_argsort = torch.argsort(token_attribution, dim=1, descending=True)
    attribution_ranks = torch.argsort(attribution_argsort, dim=1, descending=False)
    ttr = 0
    tct = 0
    for i, token_ind in enumerate(conserved_tokens):
        token_ranks = attribution_ranks[i, token_ind]
        ttr += torch.divide(1, token_ranks+1).sum()
        tct += token_ranks.shape[0]
    return ttr / tct

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

def mean_auc(token_attribution: np.ndarray, seq_len: np.ndarray, conserved_tokens):
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
    auc_l = np.array(auc_l)
    return auc_l.mean()