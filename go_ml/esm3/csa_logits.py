import os
import numpy as np
import torch
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

from esm.pretrained import (
    ESM3_function_decoder_v0,
    ESM3_sm_open_v0,
    ESM3_structure_decoder_v0,
)
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
test_prot = enzyme_l[12]

import json
train_path = "/home/andrew/cafa5_team/data/"
with open(f"{train_path}/cafa_dataset/go_terms.json", "r") as f:
    go_terms = json.load(f)

import goatools
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

import tqdm.notebook
TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'

def get_logits(seq, function_tokens=None, batch_size=1):
  protein_prompt = ESMProtein(sequence=seq)
  protein_tensor = model.encode(protein_prompt)
  x, ln = protein_tensor.sequence, len(seq)
  with torch.no_grad():
    f = lambda x, y: model(sequence_tokens=x, function_tokens=y).sequence_logits[:, 1:(ln+1), 4:24].detach().cpu().numpy()
    logits = np.zeros((ln, 20), dtype=np.float32)
    with tqdm.notebook.tqdm(total=ln, bar_format=TQDM_BAR_FORMAT) as pbar:
      for n in range(0, ln, batch_size):
        m = min(n + batch_size, ln)
        x_h = torch.clone(x).unsqueeze(0).repeat(m - n, 1)
        if(function_tokens is not None):
            y_h = torch.clone(function_tokens).unsqueeze(0).repeat(m - n, 1)
        else:
           y_h = None
        for i in range(m - n):
          x_h[i, n + i + 1] = SEQUENCE_VOCAB.index("<mask>")
        fx_h = f(x_h.to(DEVICE), y_h.to(DEVICE))
        for i in range(m - n):
          logits[n + i] = fx_h[i, n + i]
        pbar.update(m - n)
  return logits

import pickle
logit_l_cond = []
for prot in enzyme_l:
    go_term = go_terms[prot['go_ind']]
    ancestor_desc = '-'.join(godag[g].name for g in get_ancestors(go_term, go2parents_isa))
    function_annotations = [
        FunctionAnnotation(label=f, start=1, end=len(prot['seq'])) for f in function_tokenizer.keyword_vocabulary if f in ancestor_desc
    ]
    function_tokens = function_tokenizer.tokenize(function_annotations, len(prot['seq']))
    function_tokens = function_tokenizer.encode(function_tokens)
    function_tokens = function_tokens.to(DEVICE).unsqueeze(0)
    logit_l_cond.append(get_logits(prot['seq'], function_tokens=function_tokens, batch_size=12))

with open('esm3_csa_logits_cond.pkl', 'wb') as f:
   pickle.dump(logit_l_cond, f)

logit_l_base = []
for prot in enzyme_l:
    logit_l_base.append(get_logits(prot['seq'], batch_size=12))

with open('esm3_csa_logits_base.pkl', 'wb') as f:
   pickle.dump(logit_l_base, f)
