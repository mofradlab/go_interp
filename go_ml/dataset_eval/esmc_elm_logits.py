from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.utils.constants.esm3 import (
    SEQUENCE_MASK_TOKEN,
)
import torch
from go_ml.train_utils import get_elm_df

device = torch.device('cuda:0')
model = ESMC.from_pretrained("esmc_600m").to(device) # or "cpu"

vi = {i: a for a, i in model.tokenizer.get_vocab().items()}
AA_str = [vi[i] for i in range(4, 24)]

from go_ml.masking import *
def get_logits(seq, batch_size=8, mask_func=mask_indiv):
    seq_ind = model.encode(ESMProtein(sequence=seq)).sequence
    batch, batch_inds, mut_inds = mask_func(seq_ind, SEQUENCE_MASK_TOKEN)
    bert_eval_l = []
    with torch.no_grad():
        for si in range(0, batch.shape[0], batch_size):
            ei = min(batch.shape[0], si+batch_size)
            x = batch[si:ei, :]
            model_eval = model(x)
            bert_eval = model_eval.sequence_logits
            bert_eval_l.append(bert_eval.cpu())
    bert_eval = torch.cat(bert_eval_l)
    bert_eval = torch.softmax(bert_eval, dim=2)
    bert_mask = (batch == SEQUENCE_MASK_TOKEN).cpu()
    eval_avg, eval_support = mask_avg(bert_mask, bert_eval)
    return eval_avg

elm_df = get_elm_df()
from tqdm import tqdm
logit_eval = {}
for pid, seq in tqdm(list(zip(elm_df['Primary_Acc'], elm_df['Sequence']))):
    logit_eval[pid] = get_logits(seq, batch_size=16, mask_func=lambda a, b: mask_perc(a, b, 6, 0.15))

import pickle
with open('elm_logits.pkl', 'wb') as f:
    pickle.dump(logit_eval, f)