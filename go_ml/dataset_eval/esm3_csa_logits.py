import os, json, pickle
import torch
import numpy as np
torch._dynamo.config.suppress_errors = True

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.constants.esm3 import SEQUENCE_VOCAB
from esm.models.esm3 import ESM3
from esm.sdk.api import (
    ESMProtein,
    GenerationConfig,
)
model =  ESM3.from_pretrained("esm3_sm_open_v1", device=torch.device(DEVICE)).eval()

from esm.tokenization.function_tokenizer import (
    InterProQuantizedTokenizer as EsmFunctionTokenizer,
)
from esm.tokenization.sequence_tokenizer import (
    EsmSequenceTokenizer,
)
from esm.utils.constants.esm3 import (
    SEQUENCE_MASK_TOKEN,
)
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.types import FunctionAnnotation

tokenizer = EsmSequenceTokenizer()
function_tokenizer = EsmFunctionTokenizer()

from go_ml.train_utils import get_enzyme_df, enzyme_iterator
enzyme_df = get_enzyme_df()
enzyme_l = list(enzyme_iterator(enzyme_df, tokenizer))

train_path = "/home/andrew/cafa5_team/data/"
with open(f"{train_path}/cafa_dataset/go_terms.json", "r") as f:
    go_terms = json.load(f)

from goatools.obo_parser import GODag
from goatools.godag.go_tasks import get_go2parents
godag = GODag('../go-basic.obo')
go2parents_isa = get_go2parents(godag, set())
def get_ancestors(go, go2parents):
    seen = set()
    b = {go}
    while b:
        next_term = b.pop()
        if(next_term in seen or not next_term in go2parents):
            continue
        seen.add(next_term)
        b.update(go2parents[next_term])
    return seen


from go_ml.masking import *
def get_logits(seq, function_tokens=None, batch_size=8, mask_func=mask_indiv):
    protein_prompt = ESMProtein(sequence=seq)
    protein_tensor = model.encode(protein_prompt)
    seq_ind, ln = protein_tensor.sequence, len(seq)
    batch, batch_inds, mut_inds = mask_func(seq_ind, SEQUENCE_MASK_TOKEN)
    bert_eval_l = []
    with torch.no_grad():
        for si in range(0, batch.shape[0], batch_size):
            ei = min(batch.shape[0], si+batch_size)
            x = batch[si:ei, :]
            model_eval = model(sequence_tokens=x, function_tokens=function_tokens)
            bert_eval = model_eval.sequence_logits
            bert_eval_l.append(bert_eval.cpu())
    bert_eval = torch.cat(bert_eval_l)
    # bert_eval = torch.softmax(bert_eval, dim=2)
    bert_mask = (batch == SEQUENCE_MASK_TOKEN).cpu()
    eval_avg, eval_support = mask_avg(bert_mask, bert_eval)
    return eval_avg

def get_unmasked_logits(seq, function_tokens=None):
    protein_prompt = ESMProtein(sequence=seq)
    protein_tensor = model.encode(protein_prompt)
    seq_ind, ln = protein_tensor.sequence.reshape(1, -1), len(seq)
    with torch.no_grad():
        model_eval = model(sequence_tokens=seq_ind, function_tokens=function_tokens)
    return model_eval.sequence_logits[0]

from collections import defaultdict
logit_eval_dict = defaultdict(list)

from tqdm import tqdm
for prot in tqdm(enzyme_l, total=len(enzyme_l)):
    seq = prot['seq']
    if(len(prot['seq']) > 800):
        seq = prot['seq'][:800]
    go_term = go_terms[prot['go_ind']]
    ancestor_desc = '-'.join(godag[g].name for g in get_ancestors(go_term, go2parents_isa))
    function_annotations = [
        FunctionAnnotation(label=f, start=1, end=len(seq)) for f in function_tokenizer.keyword_vocabulary if f in ancestor_desc
    ]
    function_tokens = function_tokenizer.tokenize(function_annotations, len(seq))
    function_tokens = function_tokenizer.encode(function_tokens)
    function_tokens = function_tokens.to(DEVICE).unsqueeze(0)

    batch_size = 20
    if(len(seq) > 500):
        batch_size = 12
        
    logit_eval_dict['func_cond'].append(get_logits(seq, function_tokens=function_tokens, batch_size=batch_size).cpu())
    logit_eval_dict['base'].append(get_logits(seq, mask_func=mask_indiv, batch_size=batch_size).cpu())
    logit_eval_dict['mask_nc3'].append(
        get_logits(seq, mask_func=lambda a, b: mask_indiv_neighborhood(a, b, n_rad=3), batch_size=batch_size).cpu())
    logit_eval_dict['mask_nc10'].append(
        get_logits(seq, mask_func=lambda a, b: mask_indiv_neighborhood(a, b, n_rad=10), batch_size=batch_size).cpu())
    logit_eval_dict['perc'].append(
        get_logits(seq, mask_func=lambda a, b: mask_perc(a, b, 6, 0.15), batch_size=8).cpu())
    logit_eval_dict['unmasked'].append(get_unmasked_logits(seq))
    
with open('esm3_csa_logits.pkl', 'wb') as f:
   pickle.dump(logit_eval_dict, f)