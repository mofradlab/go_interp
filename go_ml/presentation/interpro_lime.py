import pandas as pd
import torch
from Bio import SeqIO
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

def create_sequence_dict(fasta_file):
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        accession_code = record.id.split("|")[1]
        sequences[accession_code] = str(record.seq)
    return sequences

fasta_file_path = '../../data/interpro_superfamilies.fasta'

sequences_dict = create_sequence_dict(fasta_file_path)

def protein_iterator():
     for pid, seq in sequences_dict.items():
          inputs = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding='max_length',
                                             truncation=True, return_attention_mask=True, max_length=800)
          yield {'prot_id': pid, 'seq': seq, 'seq_ind': inputs['input_ids'], 'mask': inputs['attention_mask']}

protein_l = list(protein_iterator())

import numpy as np
def mutate_ind(seq, lambda_=15):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    mut_seq = np.array(list(seq))
    num_changes = np.random.poisson(lambda_)
    num_changes = min(num_changes, len(seq))
    indices_to_change = np.random.choice(len(seq), num_changes, replace=False)
    mut_seq[indices_to_change] = np.random.choice(list(amino_acids), num_changes)
    mut_seq = ''.join(mut_seq)
    return mut_seq, num_changes

from go_ml.data_utils import ProtDataset, get_seq_collator
from torch.utils.data import DataLoader
from scipy.sparse import csr_array, vstack

import tqdm.notebook
TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'
NUM_MUT = 1500

lime_dict = {}
i = 0
with tqdm.tqdm(total=len(protein_l)*NUM_MUT) as pbar:
    eval_mut = []
    eval_l = []
    with torch.no_grad():
        for prot in protein_l:
            # print(prot)
            mut_seq = [mutate_ind(prot['seq']) for _ in range(NUM_MUT)]
            mut_ds = ProtDataset(['interpro']*len(mut_seq), [x[0] for x in mut_seq])
            collate_seqs = get_seq_collator(tokenizer, max_length=800, add_special_tokens=True)
            mut_dl = DataLoader(mut_ds, shuffle=False, batch_size=60, collate_fn=collate_seqs)
            logit_l = []
            for batch in mut_dl:
                seq_ind, mask = batch['seq_ind'].to(device), batch['mask'].to(device)
                logit_preds = model.forward(seq_ind, mask)
                logit_preds = logit_preds * (logit_preds > -9)
                logit_l.append(logit_preds)
                pbar.update(logit_preds.shape[0])
            logit_preds = torch.cat(logit_l)
            # save the whole array, not the csr sparse array
            sparse_preds = csr_array(logit_preds.cpu().numpy())
            # eval_mut.append(mut_seq)
            # eval_l.append(logit_preds)
            lime_dict[prot["prot_id"]] = (mut_seq, sparse_preds)
            i += 1
            # if i == 5:
            #     break

        
import pickle
with open('/home/andrew/GO_interp/go_ml/notebooks/notebook_cache/interpro_lime.pkl', 'wb') as f:
    pickle.dump(lime_dict, f)
