
import torch
device = torch.device('cuda:0')
import pandas as pd
from go_ml.train_utils import get_enzyme_df, enzyme_iterator
from go_ml.models.bert_finetune import BERTFinetune
from transformers import AutoTokenizer, AutoConfig


checkpoint_dir = "/home/andrew/GO_interp/checkpoints"
# with open(f"{checkpoint_dir}/esm_finetune_hparams.pkl", "rb") as f:
#     hparams = pickle.load(f)
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = BERTFinetune.load_from_checkpoint(f"{checkpoint_dir}/esm_finetune-v1.ckpt", map_location=device)
model.eval()

enzyme_df = get_enzyme_df()
enzyme_l = list(enzyme_iterator(enzyme_df, tokenizer))

import numpy as np
def mutate_ind(seq, lambda_=15):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    mut_seq = np.array(list(seq))
    num_changes = np.random.poisson(lambda_)
    indices_to_change = np.random.choice(len(seq), num_changes, replace=False)
    mut_seq[indices_to_change] = np.random.choice(list(amino_acids), num_changes)
    mut_seq = ''.join(mut_seq)
    return mut_seq, num_changes

from go_ml.data_utils import ProtDataset, get_seq_collator
from torch.utils.data import DataLoader
from scipy.sparse import csr_array, vstack

import tqdm.notebook
TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'
with tqdm.tqdm(total=100*1000) as pbar:
    eval_mut = []
    eval_l = []
    with torch.no_grad():
        for test_prot in enzyme_l[:100]:
            mut_seq = [mutate_ind(test_prot['seq'], lambda_=45) for _ in range(1000)]
            mut_ds = ProtDataset(['csatest']*len(mut_seq), [x[0] for x in mut_seq])
            collate_seqs = get_seq_collator(tokenizer, max_length=1024, add_special_tokens=True)
            mut_dl = DataLoader(mut_ds, shuffle=False, batch_size=60, collate_fn=collate_seqs)
            logit_l = []
            for batch in mut_dl:
                seq_ind, mask = batch['seq_ind'].to(device), batch['mask'].to(device)
                logit_preds = model.forward(seq_ind, mask)
                logit_preds = logit_preds * (logit_preds > -9)
                logit_l.append(logit_preds)
                pbar.update(logit_preds.shape[0])
            logit_preds = torch.cat(logit_l)
            sparse_preds = csr_array(logit_preds.cpu().numpy())
            eval_mut.extend(mut_seq)
            eval_l.append(sparse_preds)
        eval_results = vstack(eval_l)

import pickle
with open('/home/andrew/GO_interp/go_ml/notebooks/notebook_cache/test_1000_mut_eval.pkl', 'wb') as f:
    pickle.dump((eval_mut, eval_results), f)

    