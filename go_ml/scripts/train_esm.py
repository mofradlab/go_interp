import os, json, pickle
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from go_ml.models.esm_finetune import ESMFinetune
from go_ml.data_utils import *
from argparse import ArgumentParser
import transformers

parser = ArgumentParser()
parser = ESMFinetune.add_model_specific_args(parser)
hparams = parser.parse_args()
print("got hparams", hparams)

if __name__ == "__main__":
    train_path = "/home/andrew/cafa5_team/data/"
    with open(f"{train_path}/cafa_dataset/go_terms.json", "r") as f:
        go_terms = json.load(f)
    with open(f"{train_path}/cafa_dataset/prot_ids.json", "r") as f:
        prot_ids = json.load(f)
    with open(f"{train_path}/cafa_dataset/rev_annot.pkl", "rb") as f:
        labels = pickle.load(f)
    print((np.asarray(labels.sum(axis=1)) > 0).sum() / labels.shape[0])
    prot_sequences, seq_ids = load_protein_sequences(f"{train_path}/uniprot_sprot.fasta")
    assert all(s1 == s2 for s1, s2 in zip(prot_ids, seq_ids))
    labeled_id = (np.asarray(labels.sum(axis=1)) > 0).flatten()
    
    gen = np.random.default_rng(seed=42)
    ind = gen.permuted(np.arange(len(prot_ids))[labeled_id])
    train_len = ind.shape[0] * 4 // 5
    train_ind = ind[:train_len]
    val_ind = ind[train_len:]
    np.sort(train_ind); np.sort(val_ind)
    train_ids = [prot_ids[i] for i in train_ind]; train_sequences = [prot_sequences[i] for i in train_ind]
    val_ids = [prot_ids[i] for i in val_ind]; val_sequences = [prot_sequences[i] for i in val_ind]
    train_labels = labels[train_ind, :]; val_labels = labels[val_ind, :]
    train_dataset = BertSeqDataset(train_ids, go_terms, train_sequences, train_labels)
    val_dataset = BertSeqDataset(val_ids, go_terms, val_sequences, val_labels)
    print(f"train len {len(train_dataset)} val len {len(val_dataset)}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(hparams.model_name)
    collate_seqs = get_seq_collator(tokenizer, max_length=hparams.max_length, add_special_tokens=True)
    dataloader_params = {"shuffle": True, "batch_size": 6, "collate_fn":collate_seqs}
    val_dataloader_params = {"shuffle": False, "batch_size": 12, "collate_fn":collate_seqs}

    train_loader = DataLoader(train_dataset, **dataloader_params)
    val_loader = DataLoader(val_dataset, **val_dataloader_params)

    hparams.num_classes = train_dataset[0]['labels'].shape[0]
    hparams.num_train_steps = 10*len(train_dataset)

    model = ESMFinetune(hparams)
    
    early_stop_callback = EarlyStopping(monitor='F1/val', min_delta=0.00, patience=3, verbose=True, mode='max')
    checkpoint_callback = ModelCheckpoint(filename="/home/andrew/GO_interp/checkpoints/esm_finetune", 
                                          verbose=True, monitor='loss/val', mode='min')
    
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger("logs", name="esm_finetune")
    trainer = pl.Trainer(devices=[1], max_epochs=10, 
                         callbacks=[early_stop_callback, checkpoint_callback], logger=logger)
    trainer.fit(model, train_loader, val_loader)