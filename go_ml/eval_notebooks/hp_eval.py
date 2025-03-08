
import pickle
with open('handpicked_dataset.pkl', 'rb') as f:
    go_domain_df, prot_dict = pickle.load(f)

import numpy as np
import pandas as pd
from go_ml.train_utils import get_enzyme_df, enzyme_iterator, cls_seq_encode
import transformers
import matplotlib.pyplot as plt
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
print("Model loaded")

pid_eval_dict = {}
from tqdm import tqdm
for prot_id, prot_seq in prot_dict.items():
    eval_ind = list(range(1, 1+len(prot_seq)))
    aa_str = 'LAGVSERTIDPKQNFYMHWC'
    aa_ind = [tokenizer.get_vocab()[c] for c in aa_str]
    eval_dict = {}
    seq_data = cls_seq_encode(prot_seq.upper(), tokenizer)
    with torch.no_grad():
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
            eval_dict[res_ind] = logits.cpu().numpy()
    eval_mat = np.stack([eval_dict[i] for i in range(1, len(eval_dict)+1)])
    pid_eval_dict[prot_id] = eval_mat
    
with open('hp_eval.pkl', 'wb') as f:
    pickle.dump(pid_eval_dict, f)