import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from go_ml.models.bottleneck_dpg_conv import DPGModule
from go_ml.data_utils import *
from go_bench.metrics import calculate_ic, ic_mat
from argparse import ArgumentParser


"""
batch_size: 256
num_filters: 800
bottleneck_layers: 1.0
label_loss_weight: 10
sim_margin: 12
tmargin: 0.9
learning_rate: 5e-4

label_loss_decay: 
"""

parser = ArgumentParser()
parser = DPGModule.add_model_specific_args(parser)
model_hparams = parser.parse_known_args()[0]
hparams = parser.parse_args()

print("model hparams", model_hparams)

if __name__ == "__main__":
    train_path = "/home/andrew/cafa5_team/data/go_bench"
    with open("../data/cafa_dataset/go_terms.json", "r") as f:
        go_terms = json.load(f)
    with open("../data/cafa_dataset/prot_ids.json", "r") as f:
        prot_ids = json.load(f)
    with open("../data/cafa_dataset/rev_annot.pkl", "rb") as f:
        labels = pickle.load(f)
    print((np.asarray(labels.sum(axis=1)) > 0).sum() / labels.shape[0])
    prot_sequences, seq_ids = load_protein_sequences("../data/uniprot_sprot.fasta")
    # print(len(seq_ids), seq_ids[:100])
    # seq_ids = [s.split("|")[1] for s in seq_ids]
    assert all(s1 == s2 for s1, s2 in zip(prot_ids, seq_ids))
    gen = np.random.default_rng(seed=42)
    ind = gen.permuted(np.arange(len(prot_ids)))
    train_len = ind.shape[0] * 4 // 5
    train_ind = ind[:train_len]
    val_ind = ind[train_len:]
    np.sort(train_ind); np.sort(val_ind)
    train_ids = [prot_ids[i] for i in train_ind]; train_sequences = [prot_sequences[i] for i in train_ind]
    val_ids = [prot_ids[i] for i in val_ind]; val_sequences = [prot_sequences[i] for i in val_ind]
    train_labels = labels[train_ind, :]; val_labels = labels[val_ind, :]
    train_dataset = BertSeqDataset(train_ids, go_terms, train_sequences, train_labels)
    val_dataset = BertSeqDataset(val_ids, go_terms, val_sequences, val_labels)

    collate_seqs = get_bert_seq_collator(max_length=1024, add_special_tokens=True)
    dataloader_params = {"shuffle": True, "batch_size": 256, "collate_fn":collate_seqs}
    val_dataloader_params = {"shuffle": False, "batch_size": 256, "collate_fn":collate_seqs}

    train_loader = DataLoader(train_dataset, **dataloader_params, num_workers=6)
    val_loader = DataLoader(val_dataset, **val_dataloader_params)
    
    term_ic = torch.ones((1, train_dataset[0]['labels'].shape[0]))
    model_hparams.num_classes = train_dataset[0]['labels'].shape[0]
    model = DPGModule(**vars(model_hparams), term_ic=term_ic)
    early_stop_callback = EarlyStopping(monitor="loss/val", min_delta=0.00, patience=10, 
                                        verbose=True, mode='min', check_on_train_epoch_end=True)
    checkpoint_callback = ModelCheckpoint(
        filename="/home/andrew/go_metric/checkpoints/dpg-bottleneck",
        verbose=True,
        monitor="loss/val",
        save_on_train_epoch_end=True,
        mode='min'
    )

    trainer = pl.Trainer(devices=[0], max_epochs=150, callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(model, train_loader, val_loader)