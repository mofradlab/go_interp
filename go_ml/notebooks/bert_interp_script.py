from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSeq2SeqLM, AutoConfig
import torch
device = torch.device('cuda:1')

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
config = AutoConfig.from_pretrained('facebook/esm2_t33_650M_UR50D')
model = AutoModelForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D').to(device)
model.eval()
print("Loaded BERT Model")

import pandas as pd
enzyme_df= pd.read_csv('~/GO_interp/data/enzyme_dataset_seq.csv')
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

import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
def enzyme_iterator():
     for i, pid, annot_ind, enzyme_cls, goterm, seq, go_ind in enzyme_df.itertuples():
          # print(pid, annot_ind, enzyme_cls, goterm, seq, go_ind)
          inputs = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding='max_length',
                                             truncation=True, return_attention_mask=True, max_length=1024)
          yield {'prot_id': pid, 'annot_ind': annot_ind, 'go_ind': go_ind, 'seq': seq, 'seq_ind': inputs['input_ids'], 'mask': inputs['attention_mask']}

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

bert_distr_l = []
with torch.no_grad():
    for i, r in enumerate(enzyme_iterator()):
        seq_ind, mask =  torch.tensor(r['seq_ind']).to(device), torch.BoolTensor(r['mask']).to(device)
        residue_coverage = 6
        seq_batch, batch_inds, mut_inds = mask_seq(seq_ind[:, :], mask[:, :], tokenizer.mask_token_id, residue_coverage=residue_coverage)
        bert_pred = model(seq_batch, torch.tile(mask, (seq_batch.shape[0], 1)))
        mut_distr = torch.softmax(bert_pred.logits, dim=2)
        N, L, T = bert_pred.logits.shape
        bert_distr = torch.zeros(L, T, device=device)
        for bi, ti in zip(batch_inds.flatten(), mut_inds.flatten()):
            bert_distr[ti, :] += mut_distr[bi, ti, :]
        bert_distr /= residue_coverage
        # seq_entropy = (-bert_distr*torch.log(bert_distr)).sum(dim=1)
        bert_distr_l.append(bert_distr.cpu())
        if(i % 3 == 0):
            print(f"{100*i/len(enzyme_go_terms)}")

import pickle
with open('/home/andrew/GO_interp/data/bert_distr.pkl', 'wb') as f:
    pickle.dump(bert_distr_l, f)