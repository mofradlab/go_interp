"""
train_func_cond_esmc.py — Train a function-conditioned ESMC masked language model.

The model (FuncCondESMC) extends ESM Cambrian (ESMC 300M) with learnable GO-term
embeddings that are injected into the token embedding space before encoding. It is
trained with masked language modeling: the model must reconstruct masked residues
given the unmasked context and the protein's GO functional annotations.

Inputs
------
  --data_dir       Directory containing train_dataset.pkl and val_dataset.pkl.
                   Default: ../../data/train_esm_datasets/
  --output_dir     Directory for checkpoints and TensorBoard logs.
                   Default: ../../checkpoints/
  --gpu_id         GPU index to use (default: 0)
  --mask_func      Masking strategy: 'perc' (random 15% tokens) or 'span'
                   (contiguous spans, controlled by context_length/span_mask_length)
  --context_length  Radius of context window around each masked span (span only)
  --span_mask_length  Number of consecutive tokens per masked span (span only)

Outputs
-------
  {output_dir}/func_cond_finetune_esmc.ckpt  — best checkpoint by val loss
  {output_dir}/logs/func_cond_finetune_esmc/ — TensorBoard training logs

Hardware
--------
  Trained on a single A100/H100 GPU with bf16 mixed precision.
  Effective batch size: 10 * 4 gradient accumulation steps = 40 sequences.

Example commands (see also scripts/train.sh for the exact runs used in the paper)
-------
  python train_func_cond_esmc.py --gpu_id 0 --mask_func span --context_length 100 --span_mask_length 5
  python train_func_cond_esmc.py --gpu_id 0 --mask_func span --context_length 100 --span_mask_length 2
"""

import os, pickle
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser

from go_ml.models.func_cond_esmc import FuncCondESMC, FuncCondESMCFinetune
from go_ml.data_utils import (prot_func_collate_bert, ProtFuncDataset, BertFuncDataset,
                               bert_span_mask_parametrized, bert_mask_alias,
                               aa_tokens, mask_token_id)

# ── Arguments ────────────────────────────────────────────────────────────────

parser = ArgumentParser()
parser.add_argument("--gpu_id", default=0, type=int,
                    help="GPU index to use for training")
parser.add_argument("--mask_func", default="span", type=str,
                    choices=["perc", "span"],
                    help="Masking strategy: 'perc' for random token masking, "
                         "'span' for contiguous span masking")
parser.add_argument("--context_length", default=100, type=int,
                    help="Radius of context window around each masked span (span only)")
parser.add_argument("--span_mask_length", default=5, type=int,
                    help="Number of consecutive tokens per masked span (span only)")
parser.add_argument("--data_dir", default="../../data/train_esm_datasets/", type=str,
                    help="Directory containing train_dataset.pkl and val_dataset.pkl")
parser.add_argument("--output_dir", default="../../checkpoints/", type=str,
                    help="Directory for checkpoints and TensorBoard logs")

parser = FuncCondESMCFinetune.add_model_specific_args(parser)
hparams = parser.parse_args()

# ── Data ─────────────────────────────────────────────────────────────────────

with open(os.path.join(hparams.data_dir, "train_dataset.pkl"), "rb") as f:
    train_dataset = pickle.load(f)
with open(os.path.join(hparams.data_dir, "val_dataset.pkl"), "rb") as f:
    val_dataset = pickle.load(f)

if hparams.mask_func == "perc":
    mask_func = bert_mask_alias
elif hparams.mask_func == "span":
    def mask_func(seq_ind, attention_mask):
        return bert_span_mask_parametrized(
            seq_ind, attention_mask, mask_token_id, aa_tokens,
            mask_prob=0.3,
            context_length=hparams.context_length,
            span_length=hparams.span_mask_length,
        )

train_dataset = BertFuncDataset.from_prot_func_dataset(train_dataset, mask_func=mask_func)
val_dataset   = BertFuncDataset.from_prot_func_dataset(val_dataset,   mask_func=mask_func)

train_loader = DataLoader(train_dataset, shuffle=True,  batch_size=10, collate_fn=prot_func_collate_bert)
val_loader   = DataLoader(val_dataset,   shuffle=False, batch_size=12, collate_fn=prot_func_collate_bert)

# ── Hyperparameters ───────────────────────────────────────────────────────────

hparams.label_counts    = train_dataset.labels.sum(axis=0).A1  # per-label frequency for adaptive filtering
hparams.num_train_steps = len(train_dataset) * 15               # total steps for cosine LR schedule
hparams.learning_rate   = 1e-6                                  # conservative LR for fine-tuning a 300M param model

# ── Model ─────────────────────────────────────────────────────────────────────

model = FuncCondESMCFinetune(hparams)
print("freeze_func_encoder:", hparams.freeze_func_encoder)
print("func_emb requires_grad:", model.model.func_emb.requires_grad)

# ── Training ──────────────────────────────────────────────────────────────────

os.makedirs(hparams.output_dir, exist_ok=True)
log_dir = os.path.join(hparams.output_dir, "logs")

early_stop_callback = EarlyStopping(
    monitor="loss/val",
    min_delta=0.0,
    patience=2,          # stop after 2 epochs without improvement
    verbose=True,
    mode="min",
)
checkpoint_callback = ModelCheckpoint(
    dirpath=hparams.output_dir,
    filename="func_cond_finetune_esmc",
    monitor="loss/val",
    mode="min",
    verbose=True,
)
logger = TensorBoardLogger(log_dir, name="func_cond_finetune_esmc", default_hp_metric=False)
logger.log_hyperparams(hparams)

trainer = pl.Trainer(
    devices=[hparams.gpu_id],
    max_epochs=4,                    # early stopping typically triggers before this
    callbacks=[early_stop_callback, checkpoint_callback],
    logger=logger,
    gradient_clip_val=1.0,           # prevents gradient explosions during fine-tuning
    accumulate_grad_batches=4,       # effective batch size = 10 * 4 = 40
    precision="bf16-mixed",          # bf16 for memory efficiency on A100/H100
)
trainer.fit(model, train_loader, val_loader)
