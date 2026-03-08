"""
train_func_cond.py — Train a function-conditioned ESM2 masked language model.

ESM2-based counterpart to train_func_cond_esmc.py. Uses ESM2 (650M) as the
backbone instead of ESMC. Architecture and training procedure are otherwise
identical: GO-term embeddings are injected into the token embedding space and
the model is trained with masked language modeling.

Inputs
------
  --data_dir    Directory containing train_dataset.pkl and val_dataset.pkl.
                Default: ../../data/train_esm_datasets/
  --output_dir  Directory for checkpoints and TensorBoard logs.
                Default: ../../checkpoints/
  --gpu_id      GPU index to use (default: 0)
  --mask_func   Masking strategy: 'perc' (random 15% tokens) or 'span'
                (contiguous spans)

Outputs
-------
  {output_dir}/func_cond_finetune.ckpt           — best checkpoint by val loss
  {output_dir}/logs/func_cond_finetune/          — TensorBoard training logs

Hardware
--------
  Trained on a single A100/H100 GPU with bf16 mixed precision.
  Effective batch size: 10 * 4 gradient accumulation steps = 40 sequences.
"""

import os, pickle
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser

from go_ml.models.func_cond_esm import FuncCondESM, FuncCondESMFinetune
from go_ml.data_utils import (prot_func_collate_bert, ProtFuncDataset, BertFuncDataset,
                               bert_mask_alias, bert_span_mask_alias)

# ── Arguments ────────────────────────────────────────────────────────────────

parser = ArgumentParser()
parser.add_argument("--gpu_id", default=0, type=int,
                    help="GPU index to use for training")
parser.add_argument("--mask_func", default="span", type=str,
                    choices=["perc", "span"],
                    help="Masking strategy: 'perc' for random token masking, "
                         "'span' for contiguous span masking")
parser.add_argument("--data_dir", default="../../data/train_esm_datasets/", type=str,
                    help="Directory containing train_dataset.pkl and val_dataset.pkl")
parser.add_argument("--output_dir", default="../../checkpoints/", type=str,
                    help="Directory for checkpoints and TensorBoard logs")

parser = FuncCondESMFinetune.add_model_specific_args(parser)
hparams = parser.parse_args()

# ── Data ─────────────────────────────────────────────────────────────────────

with open(os.path.join(hparams.data_dir, "train_dataset.pkl"), "rb") as f:
    train_dataset = pickle.load(f)
with open(os.path.join(hparams.data_dir, "val_dataset.pkl"), "rb") as f:
    val_dataset = pickle.load(f)

if hparams.mask_func == "perc":
    mask_func = bert_mask_alias
elif hparams.mask_func == "span":
    mask_func = bert_span_mask_alias

train_dataset = BertFuncDataset.from_prot_func_dataset(train_dataset, mask_func=mask_func)
val_dataset   = BertFuncDataset.from_prot_func_dataset(val_dataset,   mask_func=mask_func)

train_loader = DataLoader(train_dataset, shuffle=True,  batch_size=10, collate_fn=prot_func_collate_bert)
val_loader   = DataLoader(val_dataset,   shuffle=False, batch_size=12, collate_fn=prot_func_collate_bert)

# ── Hyperparameters ───────────────────────────────────────────────────────────

hparams.label_counts    = train_dataset.labels.sum(axis=0).A1  # per-label frequency for adaptive filtering
hparams.num_train_steps = len(train_dataset) * 15               # total steps for cosine LR schedule
hparams.learning_rate   = 1e-6                                  # conservative LR for fine-tuning a 650M param model

# ── Model ─────────────────────────────────────────────────────────────────────

model = FuncCondESMFinetune(hparams)
print("freeze_func_encoder:", hparams.freeze_func_encoder)
print("func_emb requires_grad:", model.model.func_emb.requires_grad)

# ── Training ──────────────────────────────────────────────────────────────────

os.makedirs(hparams.output_dir, exist_ok=True)
log_dir = os.path.join(hparams.output_dir, "logs")

early_stop_callback = EarlyStopping(
    monitor="loss/val",
    min_delta=0.0,
    patience=2,
    verbose=True,
    mode="min",
)
checkpoint_callback = ModelCheckpoint(
    dirpath=hparams.output_dir,
    filename="func_cond_finetune",
    monitor="loss/val",
    mode="min",
    verbose=True,
)
logger = TensorBoardLogger(log_dir, name="func_cond_finetune", default_hp_metric=False)
logger.log_hyperparams(hparams)

trainer = pl.Trainer(
    devices=[hparams.gpu_id],
    max_epochs=10,                   # early stopping typically triggers before this
    callbacks=[early_stop_callback, checkpoint_callback],
    logger=logger,
    gradient_clip_val=1.0,
    accumulate_grad_batches=4,       # effective batch size = 10 * 4 = 40
    precision="bf16-mixed",
)
trainer.fit(model, train_loader, val_loader)
