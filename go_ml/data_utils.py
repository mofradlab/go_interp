"""
data_utils.py — Dataset classes, tokenization, and masking for training.

Dataset classes
---------------
  ProtFuncDataset     — protein sequences with sparse GO label vectors
  BertFuncDataset     — wraps ProtFuncDataset, applies a masking function
                        at item-fetch time for masked language modeling
  SequenceDataset     — sequences without functional labels (for inference)
  BertSeqDataset      — masked version of SequenceDataset

Masking strategies (applied per-sequence during training)
-----------------------------------------------------------
  bert_mask                  — random 15% token masking (standard BERT)
  bert_span_mask             — contiguous span masking
  bert_span_mask_parametrized — span masking with configurable context window
                                and span length (used for all paper models)

Key utilities
-------------
  gen_annot_mat       — build (N, L) annotation matrix from index lists
  prot_func_collate_bert — collate function for DataLoader (handles padding)

Label format
------------
  Labels are scipy sparse matrices with shape (N, n_go_terms). Each row is
  a binary vector indicating which GO terms are associated with that protein.
  Only GO terms with ≥ 50 occurrences in the training set are used during
  training (adaptive filtering applied inside FuncCondESMCFinetune).
"""

import gzip, json, os, pickle
from collections import Counter
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from scipy.sparse import csr_matrix, csc_matrix, dok_matrix, lil_matrix
from torch.utils import data
from transformers import BertTokenizer, AutoTokenizer
from go_bench.load_tools import load_GO_tsv_file, load_protein_sequences, convert_to_sparse_matrix
from torch.nn.utils.rnn import pad_sequence

def stable_hash(text:str):
  hash=0
  for ch in text:
    hash = ( hash*281  ^ ord(ch)*997) & 0xFFFFFFFF
  return 

def write_sparse(fn, preds, prot_rows, GO_cols, min_certainty):
    with open(fn, mode='w') as f:
        # f.write("g\tt\ts\n")
        for row, col in zip(*preds.nonzero()):
            prot_id = prot_rows[row]
            go_id = GO_cols[col]
            val = preds[row, col]
            if(val > min_certainty):
                f.write(f"{prot_id}\t{go_id}\t{val}\n")

def read_sparse(fn, prot_rows, GO_cols): 
    prm = {prot:i for i, prot in enumerate(prot_rows)}
    tcm = {term:i for i, term in enumerate(GO_cols)}
    sparse_probs = dok_matrix((len(prot_rows), len(GO_cols)))
    df = pd.read_csv(fn, sep='\t')
    for (i, prot, go_id, prob) in df.itertuples():
        if(prot in prm and go_id in tcm):
            sparse_probs[prm[prot], tcm[go_id]] = prob
    return csr_matrix(sparse_probs)

def gen_annot_mat(annot_col, seq_len, max_len=850):
    annot_mat = np.zeros((len(annot_col), max_len), dtype=bool)
    for i, annot in enumerate(annot_col):
        # print(annot)
        for chunk in annot:
            if isinstance(chunk, tuple):
                start, end = chunk
                start, end = int(start), int(end)
                annot_mat[i, start:end+1] = 1.0
            else:
                s = int(chunk)
                if(s <= seq_len[i]):
                    annot_mat[i, s] = 1.0
    return annot_mat

esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", do_lower_case=False)
class ProtFuncDataset(data.Dataset):
    def __init__(self, prot_ids, sequences, labels, tokenizer=None, data=None, tokenize=True):
        self.prot_ids = prot_ids
        self.sequences = sequences  # A list of strings representing proteins
        self.labels = labels
        self.tokenizer = tokenizer if tokenizer else esm_tokenizer
        # seq_tensors = []
        # seq_mask = []
        # for seq in sequences:
        #     inputs = esm_tokenizer(seq, add_special_tokens=True, padding='longest', truncation=True, max_length=1024, return_tensors='pt')
        #     seq_tensors.append(inputs['input_ids'][0])
        #     seq_mask.append(inputs['attention_mask'][0])
        self.tokenized = tokenize
        if(tokenize):
            tokenized_dataset = self.tokenizer.batch_encode_plus(
                sequences,
                add_special_tokens=True,
                padding='longest',
                truncation=True,
                max_length=1024,
                return_tensors='pt'
            )
            self.seq_tensors = tokenized_dataset['input_ids']
            self.seq_mask = tokenized_dataset['attention_mask']
        self.data = data if data is not None else [{} for _ in range(len(prot_ids))]

    def __len__(self):
        return len(self.prot_ids)
    def __getitem__(self, index):
        if(not self.tokenized):
            #throw error
            raise ValueError("Dataset not tokenized. Set tokenize=True in constructor.")
        d = self.data[index]
        d.update({
            "prot_id": self.prot_ids[index],
            "seq": self.sequences[index],
            "seq_tensor": self.seq_tensors[index],
            "seq_mask": self.seq_mask[index],
            "labels": torch.squeeze(torch.from_numpy(self.labels[index, :].toarray()), 0),
            "seq_len": len(self.sequences[index])
        })
        return d
    @classmethod
    def from_annot_df(cls, annot_df, go_terms, tokenizer=None):
        go_term_map = {term: i for i, term in enumerate(go_terms)}
        labels = np.zeros((len(annot_df), len(go_terms)), dtype=int)
        for i, (_, row) in enumerate(annot_df.iterrows()):
            term_ind = [go_term_map[term] for term in row['GOTerm'] if term in go_term_map]
            labels[i, term_ind] = 1
        labels = csr_matrix(labels)
        sequences = annot_df['Sequence'].tolist()
        prot_ids = annot_df['UniprotID'].tolist()
        annot_mat = torch.from_numpy(gen_annot_mat(annot_df['AnnotatedIndices'], 
                                                   [len(s) for s in annot_df['Sequence']]))
        assert len(prot_ids) == labels.shape[0] == len(sequences) == annot_mat.shape[0]
        return cls(prot_ids, sequences, labels, tokenizer=tokenizer, 
                   data=[{"GOTerm": row['GOTerm'], "annot_mask": annot_mat[i]} for i, (_, row) in enumerate(annot_df.iterrows())])
    
def dict_to_device(d, device):
    for k, v in d.items():
        if(type(v) is torch.Tensor):
            d[k] = v.to(device)
    return d

def prot_func_collate(batch, pad_token=1):
    prot_ids = [item['prot_id'] for item in batch]
    seq_tensors = torch.stack([item['seq_tensor'] for item in batch])
    seq_masks = torch.stack([item['seq_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    seq_len = [item['seq_len'] for item in batch]
    max_len = max(seq_len)

    d = {}
    for key in batch[0].keys():
        if key in ['prot_id', 'seq_tensor', 'seq_mask', 'labels', 'seq_len']:
            continue
        elif type(batch[0][key]) is torch.Tensor:
            d[key] = torch.stack([item[key] for item in batch])
        else:
            d[key] = [item[key] for item in batch]
    
    # # Pad sequences to the same length
    # padded_sequences = pad_sequence([torch.tensor(list(seq)) for seq in seq_tensors], batch_first=True, padding_value=pad_token)
    # padded_mask = pad_sequence([torch.tensor(list(mask)) for mask in seq_masks], batch_first=True, padding_value=False)
    padded_sequences = seq_tensors[:, :max_len+2] # +2 for special tokens
    padded_mask = seq_masks[:, :max_len+2] 

    d.update({
        'prot_id': prot_ids,
        'seq_ind': padded_sequences,
        'mask': padded_mask,
        'labels': labels
    })
    return d

mask_token_id = esm_tokenizer.convert_tokens_to_ids('<mask>')
aa_tokens = torch.arange(4, 24)
def bert_mask(seq_ind, attention_mask, mask_token_id, rand_token_id, mask_prob=0.15):
    masked_seq_ind = seq_ind.clone()
    labels = seq_ind.clone()
    special_tokens_mask = ~(attention_mask.bool())
    special_tokens_mask[:, 0] = 1  # [CLS]

    probability_matrix = torch.rand(*seq_ind.shape)
    probability_matrix[special_tokens_mask] = 1
    probability_matrix[attention_mask == 0] = 1

    # print(probability_matrix)

    masked_indices = probability_matrix < mask_prob
    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    labels[special_tokens_mask] = -100  # Do not compute loss on special tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (torch.rand(*labels.shape) < 0.8) & masked_indices
    masked_seq_ind[indices_replaced] = mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (torch.rand(*labels.shape) < 0.5) & masked_indices & ~indices_replaced
    random_words = torch.randint(0, len(rand_token_id), (indices_random.sum(),), dtype=torch.long)
    random_words = rand_token_id[random_words]
    masked_seq_ind[indices_random] = random_words
    return masked_seq_ind, labels

def bert_span_mask(seq_ind, attention_mask, mask_token_id, rand_token_id, mask_prob=0.15):
    masked_seq_ind = seq_ind.clone()
    labels = seq_ind.clone()
    special_tokens_mask = ~(attention_mask.bool())
    special_tokens_mask[:, 0] = 1  # [CLS]
    seq_len = attention_mask.sum(dim=1) - 2

    res_ind = torch.arange(seq_ind.shape[1]).unsqueeze(0).repeat(seq_ind.shape[0], 1)
    span_center = torch.rand(seq_ind.shape[0]) * seq_len.float()
    span_center = span_center.long() + 1
    span_center = span_center.unsqueeze(1)
    span_dist = torch.abs(res_ind - span_center)
    span_mask = (span_dist <= 55)

    kernel_size = 4
    probability_matrix = torch.rand(*seq_ind.shape)
    probability_matrix = (probability_matrix <= mask_prob / kernel_size).float()
    kernel = torch.ones((1, 1, kernel_size)) / kernel_size
    # padding = kernel_size // 2
    probability_matrix = probability_matrix.unsqueeze(1)  # add channel dimension
    probability_matrix = torch.nn.functional.conv1d(probability_matrix, kernel, padding='same')
    probability_matrix = (~(probability_matrix > 0)).float() # invert so that prob of masking is mask_prob
    probability_matrix = probability_matrix.squeeze(1)  # remove channel dimension

    probability_matrix[~span_mask] = 0 #Mask all outside span
    probability_matrix[special_tokens_mask] = 1 # Do not mask special tokens
    masked_indices = probability_matrix < mask_prob

    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    labels[~span_mask] = -100  # We only compute loss on masked tokens inside major span
    labels[special_tokens_mask] = -100  # Do not compute loss on special tokens
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = masked_indices
    masked_seq_ind[indices_replaced] = mask_token_id
    # # 10% of the time, we replace masked input tokens with random word
    # indices_random = (torch.rand(*labels.shape) < 0.5) & masked_indices & ~indices_replaced
    # random_words = torch.randint(0, len(rand_token_id), (indices_random.sum(),), dtype=torch.long)
    # random_words = rand_token_id[random_words]
    # masked_seq_ind[indices_random] = random_words
    return masked_seq_ind, labels

bert_mask_alias = lambda seq_ind, attention_mask: bert_mask(seq_ind, attention_mask, mask_token_id, aa_tokens, mask_prob=0.15)
bert_span_mask_alias = lambda seq_ind, attention_mask: bert_span_mask(seq_ind, attention_mask, mask_token_id, aa_tokens, mask_prob=0.35)

def bert_span_mask_parametrized(seq_ind, attention_mask, mask_token_id, rand_token_id, mask_prob=0.15, context_length=100, span_length=3):
    masked_seq_ind = seq_ind.clone()
    labels = seq_ind.clone()
    special_tokens_mask = ~(attention_mask.bool())
    special_tokens_mask[:, 0] = 1  # [CLS]
    seq_lens = attention_mask.sum(dim=1) - 2

    res_ind = torch.arange(seq_ind.shape[1]).unsqueeze(0).repeat(seq_ind.shape[0], 1)

    max_len = seq_ind.shape[1] - 2
    fill_total = max_len * mask_prob
    num_spans = int(fill_total / (span_length+1)) # +1 for minimum gap
    gap_total = max_len - (num_spans * (span_length+1)) # +1 for minimum gap

    ind = torch.randint(0, gap_total, (seq_ind.shape[0], num_spans))
    ind, _ = torch.sort(ind, dim=1)
    ind = ind + torch.arange(0, int(num_spans)).reshape(1, -1) * (span_length + 1)

    batch_ind = torch.arange(0, seq_ind.shape[0]).unsqueeze(1).repeat(1, int(num_spans))
    span_starts = ind
    span_runs = (span_starts.unsqueeze(2) + torch.arange(0, span_length).unsqueeze(0).unsqueeze(0)).flatten()
    batch_ind_runs = batch_ind.unsqueeze(2).repeat(1, 1, span_length).flatten()

    span_mask = torch.ones_like(seq_ind, dtype=torch.bool)
    span_mask[batch_ind_runs.long(), span_runs.long()] = 0

    context_start = torch.rand(seq_ind.shape[0]) * (seq_lens.float() - context_length)
    context_start = torch.clamp(context_start, min=0).long() + 1
    context_end = context_start + context_length

    context_mask = res_ind >= context_start.unsqueeze(1)
    context_mask = context_mask & (res_ind <= context_end.unsqueeze(1))

    batch_mask = span_mask & context_mask

    labels[batch_mask] = -100  # We only compute loss on masked tokens
    labels[~context_mask] = -100  # We only compute loss on masked tokens inside major span
    labels[special_tokens_mask] = -100  # Do not compute loss on special tokens

    indices_replaced = ~batch_mask
    masked_seq_ind[indices_replaced] = mask_token_id
    return masked_seq_ind, labels


#Inherit from ProtFuncDataset
class BertFuncDataset(ProtFuncDataset):
    def __init__(self, prot_ids, sequences, labels, tokenizer=None, 
                 data=None, tokenize=True, mask_func=bert_mask):
        if(tokenizer is None):
            tokenizer = esm_tokenizer
        super().__init__(prot_ids, sequences, labels, tokenizer=tokenizer, data=data, tokenize=tokenize)
        self.mask_func = mask_func

    @classmethod
    def from_prot_func_dataset(cls, prot_func_dataset, mask_func=bert_mask_alias):
        assert prot_func_dataset.tokenized, "Input ProtFuncDataset must be tokenized."
        bert_func_dataset = cls(prot_func_dataset.prot_ids, prot_func_dataset.sequences, 
                                prot_func_dataset.labels, tokenizer=prot_func_dataset.tokenizer, 
                                data=prot_func_dataset.data, tokenize=False, mask_func=mask_func)
        bert_func_dataset.seq_tensors = prot_func_dataset.seq_tensors
        bert_func_dataset.seq_mask = prot_func_dataset.seq_mask
        bert_func_dataset.tokenized = True
        return bert_func_dataset

    def __getitem__(self, index):
        d = self.data[index]
        d.update({
            "prot_id": self.prot_ids[index],
            "seq": self.sequences[index],
            "seq_tensor": self.seq_tensors[index],
            "seq_mask": self.seq_mask[index],
            "labels": torch.squeeze(torch.from_numpy(self.labels[index, :].toarray()), 0),
            "seq_len": len(self.sequences[index])
        })
        seq_ind = d['seq_tensor']
        attention_mask = d['seq_mask']
        masked_seq_ind, masked_seq_labels = self.mask_func(seq_ind.reshape(1, -1), attention_mask.reshape(1, -1))
        d['masked_seq_tensor'] = masked_seq_ind[0]
        d['masked_seq_labels'] = masked_seq_labels[0]
        return d
    
def truncated_stack(tensor_list, max_len): 
    stack = torch.stack(tensor_list)
    return stack[:, :max_len+2]

def prot_func_collate_bert(batch, pad_token=1):
    seq_len = [item['seq_len'] for item in batch]
    max_len = max(seq_len)

    d = {}
    for key in batch[0].keys():
        if key in ['seq_tensor', 'seq_mask', 'masked_seq_tensor', 'masked_seq_labels']:
            d[key] = truncated_stack([item[key] for item in batch], max_len)
        elif key == 'labels':
            d[key] = torch.stack([item[key] for item in batch])
        else:
            d[key] = [item[key] for item in batch]
    return d


class ProtDataset(data.Dataset):
    def __init__(self, prot_ids, sequences, prot_data=None):
        self.prot_ids = prot_ids
        self.sequences = sequences #A list of strings representing proteins
        if(prot_data is None):
            prot_data = [{} for _ in range(len(prot_ids))]
        self.prot_data = prot_data #A list of dictionaries representing data

    def __len__(self):
        return len(self.prot_ids)
    
    def __getitem__(self, index):
        dp = {"prot_id": self.prot_ids[index], "seq": self.sequences[index]}
        dp.update(self.prot_data[index])
        return dp

class SequenceDataset(data.Dataset):
    def __init__(self, prot_ids, go_terms, sequences, labels, mini=None):
        self.prot_ids = prot_ids
        self.go_terms = go_terms
        self.labels = labels #A csr matrix in which the ith row gives the classifications of the ith protein
        self.sequences = sequences #A list of strings representing proteins
        self.mini = mini

    @classmethod
    def from_pkl(cls, prot_ids, go_terms, sequence_path, labels_pkl, mini=None, 
                 prot_ids_subset=None, go_terms_subset=None):
        sequences, _ = load_protein_sequences(sequence_path, prot_ids)
        with open(labels_pkl, "rb") as f:
            labels = pickle.load(f)
        if(go_terms_subset is not None):
            term_col = {term: i for i, term in enumerate(go_terms)}
            index_subset = [term_col[term] for term in go_terms_subset]
            labels = labels[:, index_subset]
            go_terms = go_terms_subset
        if(prot_ids_subset is not None):
            prot_row = {prot_id: i for i, prot_id in enumerate(prot_ids)}
            index_subset = [prot_row[prot_id] for prot_id in prot_ids_subset]
            labels = labels[index_subset, :]
            prot_ids = prot_ids_subset
        ds = cls(prot_ids, go_terms, sequences, labels, mini=mini)
        return ds
    
    @classmethod
    def from_memory(cls, annotation_tsv_path, terms_list_path, sequence_path, cache_dir=None):
        if(cache_dir):
            cache_id = str(stable_hash(annotation_tsv_path+terms_list_path+sequence_path))
            cache_path = f"{cache_dir}/{cache_id}.pkl"
            if(os.path.isfile(cache_path)):
                with open(cache_path, 'rb') as f:
                    print("Loading from cache_id:", cache_id)
                    return pickle.load(f)
        with open(terms_list_path, "r") as f:
            term_list = json.load(f)
        protein_annotation_dict = load_GO_tsv_file(annotation_tsv_path)
        prot_id_whitelist = [prot_id for prot_id in protein_annotation_dict.keys()]
        sequences, prot_ids = load_protein_sequences(sequence_path, prot_id_whitelist)
        labels = convert_to_sparse_matrix(protein_annotation_dict, term_list, prot_ids)
        ds = cls(prot_ids, sequences, labels)
        if(cache_dir):
            with open(cache_path, 'wb') as f:
                print("Saving to cache_id:", cache_id)
                pickle.dump(ds, f)
        return ds
        
    def __len__(self):
        'Denotes the total number of samples'
        if(self.mini is not None):
            return self.mini #Good for debugging
        return len(self.sequences)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.sequences[index]
        y = torch.squeeze(torch.from_numpy(self.labels[index, :].toarray()), 0)
        return X, y

class BertSeqDataset(SequenceDataset):
    def __getitem__(self, index):
        X = " ".join(self.sequences[index].upper())
        y = torch.squeeze(torch.from_numpy(self.labels[index, :].toarray()), 0)
        prot_id = self.prot_ids[index]
        return {"seq": X, "labels": y, "prot_id": prot_id}
    
    def to_pickle(self, fn):
        import pickle
        with open(fn, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, fn, mini=None):
        import pickle
        with open(fn, 'rb') as f:
            s = pickle.load(f)
            s.mini = mini
            return s

def collate_dict(data_dict_l):
    keys = list(data_dict_l[0].keys())
    ex = data_dict_l[0]
    dd = {}
    for k, v in ex.items():
        if(type(v) is torch.Tensor):
            dd[k] = torch.stack([data_dict_l[i][k] for i in range(len(data_dict_l))])
        else:
            dd[k] = [data_dict_l[i][k] for i in range(len(data_dict_l))]                 
    return dd

# bert_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
def get_seq_collator(tokenizer, max_length=500, add_special_tokens=False):
    def seq_collator(data_dict_list):
        sample = collate_dict(data_dict_list)
        inputs = tokenizer.batch_encode_plus(sample["seq"],
                                                    add_special_tokens=add_special_tokens,
                                                    padding='longest',
                                                    truncation=True,
                                                    return_attention_mask=True,
                                                    max_length=max_length)
        sample['seq_ind'] = torch.tensor(inputs['input_ids'])
        sample['mask'] = torch.BoolTensor(inputs['attention_mask'])
        return sample
    return seq_collator

def write_sparse(fn, preds, prot_rows, GO_cols, min_certainty):
    with open(fn, mode='w') as f:
        f.write("g,t,s\n")
        for row, col in zip(*preds.nonzero()):
            prot_id = prot_rows[row]
            go_id = GO_cols[col]
            val = preds[row, col]
            if(val > min_certainty):
                f.write(f"{prot_id},{go_id},{val}\n")
                
def read_sparse(fn, prot_rows, GO_cols):
    prm = {prot:i for i, prot in enumerate(prot_rows)}
    tcm = {term:i for i, term in enumerate(GO_cols)}
    sparse_probs = dok_matrix((len(prot_rows), len(GO_cols)))
    df = pd.read_csv(fn, skiprows=1)
    for (i, prot, go_id, prob) in df.itertuples():
        if(prot in prm and go_id in tcm):
            sparse_probs[prm[prot], tcm[go_id]] = prob
    return csr_matrix(sparse_probs)

def map_embeddings(train_terms, emb_terms, emb):
    emb_mapping = {go_id: i for i, go_id in enumerate(emb_terms)}
    l = []
    for term in train_terms:
        if(term in emb_mapping):
            l.append(emb[emb_mapping[term], :])
        else:
            print("default zero")
            l.append(np.zeros(emb.shape[1]))
    term_embeddings = np.array(l)
    return term_embeddings