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
from transformers import BertTokenizer
from go_bench.load_tools import load_GO_tsv_file, load_protein_sequences, convert_to_sparse_matrix

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
                                                    padding='max_length',
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