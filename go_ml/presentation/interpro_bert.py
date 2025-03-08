import os, json, pickle
import torch
import numpy as np
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSeq2SeqLM, AutoConfig
import torch

device = torch.device('cuda:0')
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
config = AutoConfig.from_pretrained('facebook/esm2_t33_650M_UR50D')
model = AutoModelForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D').to(device)
model.eval()
print("Loaded BERT Model")

from go_ml.train_utils import get_enzyme_df, enzyme_iterator
enzyme_df = get_enzyme_df()
enzyme_l = list(enzyme_iterator(enzyme_df, tokenizer))
SEQUENCE_MASK_TOKEN = tokenizer.mask_token_id

from go_ml.masking import *
def get_logits(seq, batch_size=8, mask_func=mask_indiv):
    seq_ind = torch.LongTensor(tokenizer.encode(seq)).to(device)
    ln = len(seq)
    batch, batch_inds, mut_inds = mask_func(seq_ind, SEQUENCE_MASK_TOKEN)
    bert_eval_l = []
    with torch.no_grad():
        for si in range(0, batch.shape[0], batch_size):
            ei = min(batch.shape[0], si+batch_size)
            x = batch[si:ei, :]
            model_eval = model(x)
            bert_eval = model_eval.logits
            bert_eval_l.append(bert_eval.cpu())
    bert_eval = torch.cat(bert_eval_l)
    # bert_eval = torch.softmax(bert_eval, dim=2)
    bert_mask = (batch == SEQUENCE_MASK_TOKEN).cpu()
    eval_avg, eval_support = mask_avg(bert_mask, bert_eval)
    return eval_avg

def get_unmasked_logits(seq, function_tokens=None):
    seq_ind = torch.LongTensor(tokenizer.encode(seq)).to(device).unsqueeze(0)
    ln = len(seq)
    with torch.no_grad():
        model_eval = model(seq_ind)
    return model_eval.logits[0]

from collections import defaultdict
logit_eval_dict = defaultdict(list)

# from tqdm import tqdm
# for prot in tqdm(enzyme_l, total=len(enzyme_l)):
#     seq = prot['seq']
#     if(len(prot['seq']) > 800):
#         seq = prot['seq'][:800]
#     batch_size = 60
#     logit_eval_dict['base'].append(get_logits(seq, mask_func=mask_indiv, batch_size=batch_size).cpu())
#     logit_eval_dict['mask_nc3'].append(
#         get_logits(seq, mask_func=lambda a, b: mask_indiv_neighborhood(a, b, n_rad=3), batch_size=batch_size).cpu())
#     logit_eval_dict['mask_nc10'].append(
#         get_logits(seq, mask_func=lambda a, b: mask_indiv_neighborhood(a, b, n_rad=10), batch_size=batch_size).cpu())
#     logit_eval_dict['perc'].append(
#         get_logits(seq, mask_func=lambda a, b: mask_perc(a, b, 6, 0.15), batch_size=8).cpu())
#     logit_eval_dict['unmasked'].append(get_unmasked_logits(seq))
    
# with open('esm2_interpro_logits.pkl', 'wb') as f:
#    pickle.dump(logit_eval_dict, f)

from Bio import SeqIO
sequences = []
acc_codes = []
fasta_file_path = '../../data/interpro_superfamilies.fasta'
for record in SeqIO.parse(fasta_file_path, "fasta"):
    accession_code = record.id.split("|")[1]
    acc_codes.append(accession_code)
    sequences.append(str(record.seq))

def protein_iterator():
     for pid, seq in zip(acc_codes, sequences):
          inputs = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding='max_length',
                                             truncation=True, return_attention_mask=True, max_length=800)
          yield {'prot_id': pid, 'seq': seq, 'seq_ind': inputs['input_ids'], 'mask': inputs['attention_mask']}

protein_l = list(protein_iterator())

from tqdm import tqdm
for prot in tqdm(protein_l, total=len(protein_l)):
    pid = prot['prot_id']
    seq = prot['seq']
    if(len(prot['seq']) > 800):
        seq = prot['seq'][:800]
    batch_size = 60
    logit_eval_dict['perc'].append(
        get_logits(seq, mask_func=lambda a, b: mask_perc(a, b, 6, 0.15), batch_size=60).cpu())
    
with open('esm2_interpro_logits.pkl', 'wb') as f:
   pickle.dump(logit_eval_dict, f)