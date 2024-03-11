import json, torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import go_bench
from go_bench.load_tools import load_GO_tsv_file, load_protein_sequences, convert_to_sparse_matrix
from go_ml.data_utils import *
from scipy.sparse import csr_matrix, csc_matrix, dok_matrix, vstack, hstack
from sklearn.metrics import precision_recall_fscore_support


# sequences, prot_ids = load_protein_sequences("/home/andrew/cafa5_team/Test/testsuperset.fasta")
# with open("/home/andrew/cafa5_team/data/cafa_dataset/go_terms.json") as f:
#     go_terms = json.load(f)
# test_dataset = BertSeqDataset(prot_ids, go_terms, sequences, csr_matrix((len(prot_ids), 123)))
# collate_seqs = get_bert_seq_collator(max_length=1024, add_special_tokens=False)
# val_dataloader_params = {"shuffle": False, "batch_size": 64, "collate_fn":collate_seqs}
# test_loader = DataLoader(test_dataset, **val_dataloader_params, num_workers=6)

# from go_ml.models.bert_emb import ProtBertBFDClassifier
# import pickle 
# with open("/home/andrew/cafa5_team/checkpoints/bert_emb_hparams.pkl", "rb") as f:
#     hparams = pickle.load(f)
#     hparams.num_classes = len(go_terms)

# model = ProtBertBFDClassifier.load_from_checkpoint("/home/andrew/cafa5_team/checkpoints/bert_emb_sample.ckpt", hparams=hparams)
# model.eval()
# device = torch.device('cuda:0')
# model.to(device)

# def get_sparse_probs_bert(model, dataloader, threshold=0.02):
#     prot_ids = []
#     probs_list = []
#     with torch.no_grad():
#         for i, d in enumerate(dataloader):
#             prot_id_l = d["prot_id"]
#             inputs, mask, y = d['seq'].to(device), d['mask'].to(device), d['labels'].to(device)
#             prot_ids.extend(prot_id_l)
#             m_probs = model.forward(inputs, None, mask)
#             torch.sigmoid(m_probs, out=m_probs)
#             m_probs = m_probs.cpu().numpy()
#             m_probs = np.where(m_probs > threshold, m_probs, 0) #Threshold unlikely predictions to keep output sparse. 
#             new_probs = csr_matrix(m_probs, dtype=np.float32)
#             probs_list.append(new_probs)
#             if(i % 10 == 0):
#                 print(100 * i / len(dataloader))
#     probs = vstack(probs_list)
#     return prot_ids, probs

# test_ids, test_probs = get_sparse_probs_bert(model, test_loader, threshold=0.03)
# import pickle
# with open("/home/andrew/cafa5_team/predictions/bert_preds.pkl", "wb") as f:
#     pickle.dump({"prot_ids": test_ids, "probs": test_probs}, f)

with open("/home/andrew/cafa5_team/predictions/bert_preds.pkl", "rb") as f:
    bert_preds = pickle.load(f)
    test_ids, test_probs = bert_preds['prot_ids'], bert_preds['probs']

with open("/home/andrew/cafa5_team/data/cafa_dataset/go_terms.json") as f:
    go_terms = json.load(f)

def write_sparse(fn, preds, prot_rows, GO_cols, min_certainty):
    with open(fn, mode='w') as f:
        # f.write("g\tt\ts\n")
        for row, col in zip(*preds.nonzero()):
            prot_id = prot_rows[row]
            go_id = GO_cols[col]
            val = preds[row, col]
            if(val > min_certainty):
                f.write(f"{prot_id}\t{go_id}\t{val}\n")

write_sparse("/home/andrew/cafa5_team/predictions/bert_preds.tsv", test_probs, test_ids, go_terms, 0.03)



# Code to generate embeddings
# def get_finetune_embeddings(model, dataset, device):
#     collate_seqs = get_bert_seq_collator(max_length=1024, add_special_tokens=True)
#     dataloader = DataLoader(dataset, collate_fn=collate_seqs, batch_size=128, shuffle=False)
#     prot_ids, emb_l = [], []
#     with torch.no_grad():
#         for inputs in dataloader:
#             prot_ids.extend(inputs['prot_id'])
#             tokenized_sequences = inputs["seq"].to(device)
#             attention_mask = inputs["mask"].to(device)

#             word_embeddings = model.ProtBertBFD(tokenized_sequences,
#                                            attention_mask)[0]
#             embedding = model.pool_strategy({"token_embeddings": word_embeddings,
#                                       "cls_token_embeddings": word_embeddings[:, 0],
#                                       "attention_mask": attention_mask,
#                                       }, pool_max=False, pool_mean_sqrt=False)
#             emb_l.append(embedding.cpu())
#             if(len(prot_ids) % 1024 == 0):
#                 print(f"{len(prot_ids)*100 / len(dataset)}%")
#     embeddings = torch.cat(emb_l, dim=0)
#     return prot_ids, embeddings

# test_ids, test_embeddings = get_finetune_embeddings(model, test_dataset, device)
# emb_dict = {"prot_id": test_ids, "embedding": test_embeddings}
# with open("emb/new_finetune_test_emb.pkl", "wb") as f:
#     pickle.dump(emb_dict, f)