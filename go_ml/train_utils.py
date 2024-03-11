import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import numpy as np
import json, pickle
from go_bench.load_tools import load_protein_sequences
from go_ml.data_utils import BertSeqDataset, get_seq_collator
import transformers

def get_dataloaders(model_name="facebook/esm2_t6_8M_UR50D", max_length=1024):
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

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    collate_seqs = get_seq_collator(tokenizer, max_length=max_length, add_special_tokens=True)
    dataloader_params = {"shuffle": True, "batch_size": 8, "collate_fn":collate_seqs}
    val_dataloader_params = {"shuffle": False, "batch_size": 24, "collate_fn":collate_seqs}

    train_loader = DataLoader(train_dataset, **dataloader_params, num_workers=6)
    val_loader = DataLoader(val_dataset, **val_dataloader_params)
    return train_loader, val_loader

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        Credit to https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            lr = self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            lr = [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            lr = [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]
        self._last_lr = lr
        return lr

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr