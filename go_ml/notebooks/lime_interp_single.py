import torch
import torch.nn as nn
import pickle
from go_ml.models.bert_finetune import BERTFinetune
import pandas as pd

enzyme_df= pd.read_csv('/home/andrew/GO_interp/data/enzyme_dataset_seq.csv')
enzyme_df= enzyme_df[~enzyme_df['Sequence'].isna()]
enzyme_go_terms = [gt.split("'")[1] for gt in enzyme_df['GOTerm']]
import json
train_path = "/home/andrew/cafa5_team/data/"
with open(f"{train_path}/cafa_dataset/go_terms.json", "r") as f:
    go_terms = json.load(f)
term_ind_map = {t:i for i, t in enumerate(go_terms)}
enzyme_df['GOTerm'] = enzyme_go_terms
enzyme_df= enzyme_df[enzyme_df['GOTerm'].isin(term_ind_map)]
enzyme_term_index = [term_ind_map[t] for t in enzyme_df['GOTerm']]
enzyme_df['GOTermIndex'] = enzyme_term_index
annotated_indices = [list(filter(lambda x: x < min(1024, len(seq)), map(int, x[1:-1].split(',')))) for x, seq in zip(enzyme_df['AnnotatedIndices'], enzyme_df['Sequence'])]
enzyme_df['AnnotatedIndices'] = annotated_indices

device = torch.device('cuda:1')

import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
def enzyme_iterator():
     for i, pid, annot_ind, enzyme_cls, goterm, seq, go_ind in enzyme_df.itertuples():
          # print(pid, annot_ind, enzyme_cls, goterm, seq, go_ind)
          inputs = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding='max_length',
                                             truncation=True, return_attention_mask=True, max_length=1024)
          yield {'prot_id': pid, 'annot_ind': annot_ind, 'go_ind': go_ind, 'seq': seq, 'seq_ind': inputs['input_ids'], 'mask': inputs['attention_mask']}
seq_len = torch.LongTensor([len(r['seq']) for r in enzyme_iterator()])

checkpoint_dir = "/home/andrew/GO_interp/checkpoints"
# with open(f"{checkpoint_dir}/esm_finetune_hparams.pkl", "rb") as f:
#     hparams = pickle.load(f)
model = BERTFinetune.load_from_checkpoint(f"{checkpoint_dir}/esm_finetune-v1.ckpt", map_location=device)
model.eval()

mask_l = []
for r in enzyme_iterator():
    m = torch.zeros(1024, dtype=bool)
    m[1:len(r['seq'])+1] = True
    mask_l.append(m)
enzyme_mask = torch.stack(mask_l)

import numpy as np
import matplotlib.pyplot as plt
import pickle, torch
with open('/home/andrew/GO_interp/data/bert_distr.pkl', 'rb') as f:
    bert_distr = pickle.load(f)

high_eval_set = []

top_k_ind = torch.topk(bert_attr, k=75, dim=1, largest=True).indices - 1

def mutate(seq, lambda_=15):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    mut_seq = np.array(list(seq))
    num_changes = np.random.poisson(lambda_)
    indices_to_change = np.random.choice(len(seq), num_changes, replace=False)
    mut_seq[indices_to_change] = np.random.choice(list(amino_acids), num_changes)
    mut_seq = ''.join(mut_seq)
    return mut_seq, num_changes

def mutate_ind(seq, key_ind, lambda_=15):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    mut_seq = np.array(list(seq))
    num_changes = np.random.poisson(lambda_)
    indices_to_change = np.random.choice(len(key_ind), num_changes, replace=False)
    mut_seq[key_ind[indices_to_change]] = np.random.choice(list(amino_acids), num_changes)
    mut_seq = ''.join(mut_seq)
    return mut_seq, num_changes

def mut_batch(prot_id, seq, key_ind, lambda_=15, batch_size=100):
    mut_seq = [mutate_ind(seq, key_ind, lambda_=lambda_)[0] for _ in range(batch_size)]
    inputs = tokenizer.batch_encode_plus(mut_seq, add_special_tokens=True, padding='max_length',
                    truncation=True, return_attention_mask=True, max_length=1024) 
    return {'seq': mut_seq, 'prot_id': [prot_id]*batch_size, 'seq_ind': inputs['input_ids'], 'mask': inputs['attention_mask']}

mut_seq_dl = []
for i, r in enumerate(enzyme_iterator()):
    mut_seq_batch = mut_batch(r['prot_id'], r['seq'], top_k_ind[i, :min(75, len(r['seq']))], lambda_=15, batch_size=100)
    mut_seq_dl.append(mut_seq_batch)

def collate_dict(data_dict_l):
    ex = data_dict_l[0]
    dd = {}
    for k, v in ex.items():
        if(type(v) is torch.Tensor):
            dd[k] = torch.cat([data_dict_l[i][k] for i in range(len(data_dict_l))])
        else:
            dd[k] = []
            for i in range(len(data_dict_l)):
                dd[k].extend(data_dict_l[i][k])              
    return dd
mut_seq_dataset = collate_dict(mut_seq_dl)

import pickle

from scipy.sparse import csr_array, vstack
def get_evals(model, mut_seq_dataset, batch_size=60):
    d = mut_seq_dataset
    device = model.device
    eval_l = []
    with torch.no_grad():
        for i in range(0, len(d['seq_ind']), batch_size):
            seq_ind, mask =  torch.tensor(d['seq_ind'][i:i+batch_size]).to(device), torch.BoolTensor(d['mask'][i:i+batch_size]).to(device)
            logit_preds = model.forward(seq_ind, mask)
            logit_preds = logit_preds * (logit_preds > -9)
            sparse_preds = csr_array(logit_preds.cpu().numpy())
            eval_l.append(sparse_preds)
            if(i % (12*batch_size) == 0 and i > 0):
                print(f"{100*i/len(d['seq_ind'])}% Eval")
    eval_results = vstack(eval_l)
    return eval_results

model_evals = get_evals(model, mut_seq_dataset, batch_size=75)
with open('/home/andrew/GO_interp/data/enzyme_mut_evals.pkl', 'wb') as f:
    pickle.dump((mut_seq_dataset, model_evals), f)