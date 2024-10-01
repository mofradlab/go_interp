import torch
device = torch.device('cuda:0')
import pandas as pd
from go_ml.train_utils import get_enzyme_df, enzyme_iterator
import transformers

from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoConfig
import torch

def mask_seq(seq_ind, attention_mask, mask_token, residue_coverage=6, mut_per=0.15):
    device = seq_ind.device
    seq_len = attention_mask.sum()-1
    mut_count = torch.floor(seq_len*mut_per).int().item()
    total_muts = (torch.floor(seq_len*residue_coverage/mut_count)*mut_count).int().item()
    
    mut_inds = (torch.randperm(total_muts).reshape(-1, mut_count).to(device) % seq_len) + 1
    batch_inds = torch.tile(torch.arange(0, mut_inds.shape[0]).reshape((-1, 1)), (1, mut_count))
    mut_inds, batch_inds = mut_inds.to(device), batch_inds.to(device)

    batch = torch.tile(seq_ind, (mut_inds.shape[0], 1))
    batch[batch_inds, mut_inds] = mask_token
    return batch, batch_inds, mut_inds

tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
config = AutoConfig.from_pretrained('facebook/esm2_t6_8M_UR50D')
model = AutoModelForMaskedLM.from_pretrained('facebook/esm2_t6_8M_UR50D').to(device)
model.eval()

enzyme_df = get_enzyme_df()
enzyme_l = list(enzyme_iterator(enzyme_df, tokenizer))

import tqdm
with torch.no_grad():
    bert_distr_l = []
    with tqdm.tqdm(total=100) as pbar:
        for test_prot in enzyme_l[:100]:
            seq_ind, mask =  torch.tensor(test_prot['seq_ind']).to(device), torch.BoolTensor(test_prot['mask']).to(device)
            residue_coverage = 6
            seq_batch, batch_inds, mut_inds = mask_seq(seq_ind[:, :], mask[:, :], tokenizer.mask_token_id, residue_coverage=residue_coverage)
            coverage_counts = torch.zeros(seq_ind.shape[1]).to(device)
            for r in mut_inds:
                coverage_counts[r] += 1
            bert_pred = model(seq_batch, torch.tile(mask, (seq_batch.shape[0], 1)))
            mut_distr = torch.softmax(bert_pred.logits, dim=2)
            mut_distr.shape
            N, L, T = bert_pred.logits.shape
            bert_distr = torch.zeros(L, T, device=device)
            for bi, ti in zip(batch_inds.flatten(), mut_inds.flatten()):
                bert_distr[ti, :] += mut_distr[bi, ti, :]
            bert_distr /= residue_coverage
            bert_distr_l.append(bert_distr.cpu())
            pbar.update()

import pickle
with open('/home/andrew/GO_interp/go_ml/notebooks/notebook_cache/test_100_mini_bert_distr.pkl', 'wb') as f:
    pickle.dump(bert_distr_l, f)