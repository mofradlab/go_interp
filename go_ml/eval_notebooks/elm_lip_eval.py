import matplotlib.pyplot as plt
import numpy as np

import json, pickle
train_path = "/home/andrew/cafa5_team/data/"
with open(f"{train_path}/cafa_dataset/go_terms.json", "r") as f:
    go_terms = json.load(f)
# with open(f"{train_path}/cafa_dataset/go_terms.json", "r") as f:
#         go_terms = json.load(f)
# with open(f"{train_path}/cafa_dataset/prot_ids.json", "r") as f:
#     prot_ids = json.load(f)
# with open(f"{train_path}/cafa_dataset/rev_annot.pkl", "rb") as f:
#     labels = pickle.load(f)
# labels[:, go_terms.index('GO:0035615')].sum()

from go_ml.data_utils import *
from go_ml.train_utils import cls_seq_encode

train_path = "/home/andrew/GO_interp/data/elm"
prot_sequences, seq_ids = load_protein_sequences(f"{train_path}/lig_elm_instances.fasta")
prot_sequences = [seq.upper() for seq in prot_sequences]
f_ind = [i for i in range(len(prot_sequences)) if len(prot_sequences[i]) <= 800]
prot_sequences = [prot_sequences[ind] for ind in f_ind]
seq_ids = [seq_ids[ind] for ind in f_ind]

import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
from go_ml.go_utils import godag, go2parents_isa, get_ancestors

aa_str = 'LAGVSERTIDPKQNFYMHWC'
aa_ind = [tokenizer.get_vocab()[c] for c in aa_str]

import torch
device = torch.device('cuda:0')
from go_ml.models.bert_finetune import BERTFinetune
checkpoint_dir = "/home/andrew/GO_interp/checkpoints"
model = BERTFinetune.load_from_checkpoint(f"{checkpoint_dir}/esm_finetune-v1.ckpt", map_location=device)
model.eval()
print('model loaded')

from tqdm import tqdm
def get_eval_mat(prot_seq):
    print(prot_seq)
    eval_ind = list(range(1, 1+len(prot_seq)))
    eval_dict = {}
    seq_data = cls_seq_encode(prot_seq.upper(), tokenizer)
    with torch.no_grad():
        aa_str = 'LAGVSERTIDPKQNFYMHWC'
        aa_ind = [tokenizer.get_vocab()[c] for c in aa_str]
        aa_ind = torch.tensor(aa_ind, device=device)
        seq_ind, mask =  torch.tensor(seq_data['seq_ind']).to(device), torch.BoolTensor(seq_data['mask']).to(device)
        seq_ind = seq_ind[:, :len(prot_seq) + 2]
        mask = mask[:, :len(prot_seq) + 2]
        base_batch_seq = torch.tile(seq_ind, (aa_ind.shape[0], 1))
        batch_mask = torch.tile(mask, (aa_ind.shape[0], 1))
        for res_ind in tqdm(eval_ind):
            batch_seq = base_batch_seq.clone()
            batch_seq[:, res_ind] = aa_ind
            logits = model.forward(batch_seq, batch_mask)
            eval_dict[res_ind] = logits.cpu()
    eval_mat = torch.stack([eval_dict[i] for i in range(1, len(eval_dict)+1)])
    return eval_mat

for prot_id, prot_seq in zip(seq_ids, prot_sequences):
    eval_mat = get_eval_mat(prot_seq)
    torch.save(eval_mat, f'lip_mutation_scan/{prot_id}.pt')

