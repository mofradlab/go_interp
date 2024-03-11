from go_ml.train_utils import get_dataloaders
train_loader, val_loader = get_dataloaders()

import json
train_path = "/home/andrew/cafa5_team/data/"
with open(f"{train_path}/cafa_dataset/go_terms.json", "r") as f:
    go_terms = json.load(f)

batch = next(iter(val_loader))
print(batch['prot_id'])
for i in range(batch['labels'].shape[0]):
    print(batch['prot_id'][i], list(batch['labels'][i].nonzero().flatten().numpy()))

import torch
device = torch.device('cuda:1')

import pickle
from go_ml.models.bert_finetune import BERTFinetune
checkpoint_dir = "/home/andrew/GO_interp/checkpoints"
with open(f"{checkpoint_dir}/esm_finetune_hparams.pkl", "rb") as f:
    hparams = pickle.load(f)
model = BERTFinetune.load_from_checkpoint(f"{checkpoint_dir}/esm_finetune.ckpt", model_args=hparams, 
                                          map_location=device)

import torch.nn as nn
class SingleGOModel(nn.Module):
    def __init__(self, model, go_idx):
        super().__init__()
        self.model = model
        self.go_idx = go_idx

    def forward(self, seq, mask):
        return self.model(seq, mask)[:, self.go_idx]
go260_model = SingleGOModel(model, 260)

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSeq2SeqLM, AutoConfig
tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
config = AutoConfig.from_pretrained('facebook/esm2_t6_8M_UR50D')

from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
token_reference = TokenReferenceBase(reference_token_idx=tokenizer.pad_token_id)
lig = LayerIntegratedGradients(model, model.model.embeddings.word_embeddings)

batch_idx = batch['prot_id'].index('Q13422')
seq = batch['seq'][batch_idx:batch_idx+1]
mask = batch['mask'][batch_idx:batch_idx+1]
# gopred = model(seq.to(device), mask.to(device))

text = tokenizer.convert_ids_to_tokens(seq.flatten())
seq_length = seq.shape[1]

reference_indices = token_reference.generate_reference(seq_length, device=device).unsqueeze(0)
pred = go260_model(seq.to(device), mask.to(device)).item()

attributions_ig, delta = lig.attribute(seq.to(device), reference_indices, \
                                           n_steps=2, return_convergence_delta=True, 
                                           additional_forward_args=mask.to(device))

print(text)
print(attributions_ig)