import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json, pickle
from go_bench.load_tools import load_protein_sequences
from go_ml.data_utils import BertSeqDataset, get_seq_collator
import transformers
from Bio import SeqIO
import pandas as pd

def get_elm_df(instance_url='/home/andrew/GO_interp/data/elm/elm_instances.tsv',
                cls_url='/home/andrew/GO_interp/data/elm/elm_classes.tsv',
                sequence_url='/home/andrew/GO_interp/data/elm/elm_sequences.fasta'):
    sequences = [str(s.seq).upper() for s in SeqIO.parse(sequence_url, 'fasta')]
    seq_ids = [str(s.id).split('|')[1] for s in SeqIO.parse(sequence_url, 'fasta')]
    seq_map = {si: s for si, s in zip(seq_ids, sequences)}
    elm_df = pd.read_csv(instance_url, sep='\t')
    elm_cls_df = pd.read_csv(cls_url, sep='\t')
    elm_df['Sequence'] = [seq_map[si] for si in elm_df['Primary_Acc']]
    elm_df = pd.merge(elm_df, elm_cls_df[['ELMIdentifier', 'Regex']], on='ELMIdentifier', how='left')
    elm_df = elm_df[pd.Series([len(seq) >= 50 and len(seq) <= 800 for seq in elm_df['Sequence']])]

    annot_ind = [list(range(si, ei+1)) for si, ei in zip(elm_df['Start'], elm_df['End'])]
    elm_df['MotifInd'] = annot_ind
    motif_str = [seq[si-1:ei] for seq, si, ei in zip(elm_df['Sequence'], elm_df['Start'], elm_df['End'])]
    elm_df['MotifStr'] = motif_str
    return elm_df

def get_enzyme_df(df_url="/home/andrew/GO_interp/data/enzyme_dataset_seq.csv", 
                  train_path="/home/andrew/cafa5_team/data/", max_len=800):
    enzyme_df= pd.read_csv(df_url)
    enzyme_df= enzyme_df[~enzyme_df['Sequence'].isna()]
    enzyme_go_terms = [gt.split("'")[1] for gt in enzyme_df['GOTerm']]
    with open(f"{train_path}/cafa_dataset/go_terms.json", "r") as f:
        go_terms = json.load(f)
    term_ind_map = {t:i for i, t in enumerate(go_terms)}
    enzyme_df['GOTerm'] = enzyme_go_terms
    enzyme_df= enzyme_df[enzyme_df['GOTerm'].isin(term_ind_map)]
    enzyme_term_index = [term_ind_map[t] for t in enzyme_df['GOTerm']]
    enzyme_df['GOTermIndex'] = enzyme_term_index
    annotated_indices = [list(filter(lambda x: x < min(1024, len(seq)), map(int, x[1:-1].split(',')))) for x, seq in zip(enzyme_df['AnnotatedIndices'], enzyme_df['Sequence'])]
    enzyme_df['AnnotatedIndices'] = annotated_indices
    enzyme_df = enzyme_df[[len(annot_ind) > 0 for annot_ind in annotated_indices]]
    enzyme_df = enzyme_df[[len(seq) <= max_len for seq in enzyme_df['Sequence']]]
    return enzyme_df

def cls_seq_encode(seq, tokenizer):
    inputs = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding='max_length',
                                             truncation=True, return_attention_mask=True, max_length=1024)
    return {'seq': seq, 'seq_ind': inputs['input_ids'], 'mask': inputs['attention_mask']}

def enzyme_iterator(enzyme_df, tokenizer):
     for i, pid, annot_ind, enzyme_cls, goterm, seq, go_ind in enzyme_df.itertuples():
          # print(pid, annot_ind, enzyme_cls, goterm, seq, go_ind)
          inputs = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding='max_length',
                                             truncation=True, return_attention_mask=True, max_length=1024)
          yield {'prot_id': pid, 'annot_ind': annot_ind, 'go_ind': go_ind, 'seq': seq, 'seq_ind': inputs['input_ids'], 'mask': inputs['attention_mask']}

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
                 max_lr_mul : float = 1e2,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        self.max_lr_mul = max_lr_mul
        self.base_max_lr_mul = max_lr_mul

        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            # param_group['lr'] = self.min_lr
            self.base_lrs.append(param_group['lr'])
    
    def get_lr(self):
        lr = []
        for base_lr in self.base_lrs:
            max_lr = self.max_lr_mul*base_lr # max learning rate in the current cycle
            if self.step_in_cycle == -1:
                lr.append(base_lr)
            elif self.step_in_cycle < self.warmup_steps:
                lr.append((max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr)
            else:
                lr.append(base_lr + (max_lr - base_lr) \
                        * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                        / (self.cur_cycle_steps - self.warmup_steps))) / 2)
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
                
        self.max_lr_mul = self.base_max_lr_mul * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def bert_mask(seq_batch, sequence_mask_token, base_tokens, mut_per=0.15):
    batch_size, seq_len = seq_batch.shape
    device = seq_batch.device
    # seq_len = torch.LongTensor([seq_len - 2]) #Discount SOS and EOS tokens at start and end
    seq_len -= 2
    mut_count = int(seq_len*mut_per)
    mut_inds = torch.stack([torch.randperm(seq_len) for _ in range(batch_size)])[:, :mut_count] + 1
    batch_inds = torch.tile(torch.arange(0, mut_inds.shape[0]).reshape((-1, 1)), (1, mut_count))
    mut_inds, batch_inds = mut_inds.to(device), batch_inds.to(device)
    update_batch = seq_batch.clone()
    replacement_tokens = seq_batch[batch_inds, mut_inds]
    mask_token_mask = torch.rand((replacement_tokens.shape)) > 0.2
    replacement_tokens[mask_token_mask] = sequence_mask_token
    mut_token_mask = mask_token_mask & (torch.rand((replacement_tokens.shape)) > 0.5)
    mut_index = torch.randint(0, base_tokens.shape[0], (mut_token_mask.sum(),))
    mut_tokens = base_tokens[mut_index]
    replacement_tokens[mut_token_mask] = mut_tokens
    update_batch[batch_inds, mut_inds] = replacement_tokens
    return update_batch
