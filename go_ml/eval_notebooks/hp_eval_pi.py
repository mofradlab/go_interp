import matplotlib.pyplot as plt

import onnxruntime as rt
rt.get_device()
import json
train_path = "/home/andrew/cafa5_team/data/"
with open(f"{train_path}/cafa_dataset/go_terms.json", "r") as f:
    go_terms = json.load(f)
with open('/home/andrew/GO_interp/data/proteinf_go_terms.json', 'r') as f:
    go_terms_pi = json.load(f)
sess = rt.InferenceSession("/home/andrew/GO_interp/data/proteinf_model.onnx")
import pickle
with open('handpicked_dataset.pkl', 'rb') as f:
    go_domain_df, prot_dict = pickle.load(f)

AMINO_ACID_VOCABULARY = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
RESIDUE_TO_INT = {aa: idx for idx, aa in enumerate(AMINO_ACID_VOCABULARY)}
import numpy as np
oh_mat = np.eye(len(RESIDUE_TO_INT), dtype=np.float32)
def get_onehot(seq_ind):
    seq_oh = oh_mat[seq_ind]
    return seq_oh

def token_pa(seq):
    seq_len = len(seq)
    seq_ind = np.array([RESIDUE_TO_INT[c] for c in seq]).reshape((1, -1))
    mut_ind = np.arange(20)
    eval_dict = {}
    base_batch_seq = np.tile(seq_ind, (mut_ind.shape[0], 1))
    base_batch_len = np.full(mut_ind.shape, seq_len, dtype=np.int32)
    for ri in range(seq_len):
        batch_seq = base_batch_seq.copy()
        batch_seq[:, ri] = mut_ind
        batch_seq_oh = get_onehot(batch_seq)
        logits = sess.run(output_names=['output'], input_feed={'sequence_length': base_batch_len, 'sequence': batch_seq_oh})
        eval_dict[ri] = logits
    eval_mat = np.concatenate([eval_dict[i] for i in range(0, len(eval_dict))])
    return eval_mat

pid_eval_dict = {}
from tqdm import tqdm
for prot_id, prot_seq in tqdm(prot_dict.items()):
    eval_mat = token_pa(prot_seq.upper())
    pid_eval_dict[prot_id] = eval_mat

with open('hp_eval_pi.pkl', 'wb') as f:
    pickle.dump(pid_eval_dict, f)