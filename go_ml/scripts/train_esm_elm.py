import os, json, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from go_ml.data_utils import *
from argparse import ArgumentParser
import transformers
from transformers import AutoModelForMaskedLM

model_name = 'facebook/esm2_t33_650M_UR50D'

from go_ml.train_utils import get_elm_df
elm_df = get_elm_df(instance_url='/home/andrew/GO_interp/data/elm/elm_instances_all.tsv',
                cls_url='/home/andrew/GO_interp/data/elm/elm_classes.tsv',
                sequence_url='/home/andrew/GO_interp/data/elm/elm_sequences_all.fasta')
elm_id_counter = Counter(elm_df['ELMIdentifier'])
elm_df = elm_df[[elm_id_counter[eid] >= 10 for eid in elm_df['ELMIdentifier']]]
elm_cls_group = elm_df.groupby('Primary_Acc')[['ELMIdentifier']].agg(list).reset_index()
elm_df_cls = elm_cls_group.merge(elm_df[['Primary_Acc', 'Sequence']].drop_duplicates('Primary_Acc'), on='Primary_Acc', how='left')
cls_list = elm_df['ELMIdentifier'].unique(); cls_list.sort()
cls_ind = {cls_str:i for i, cls_str in enumerate(cls_list)}
elm_df_cls['ELMInd'] = elm_df_cls['ELMIdentifier'].apply(lambda l: [cls_ind[c] for c in l])
label_series = [np.zeros(len(cls_list)) for _ in range(len(elm_df_cls))]
for t_m, elm_ind in zip(label_series, elm_df_cls['ELMInd']):
    t_m[elm_ind] = 1
elm_df_cls['ELMLabel'] = label_series

from go_ml.train_utils import bert_mask
from go_ml.data_utils import collate_dict
from torch.utils.data import DataLoader

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
prot_ids = elm_df_cls['Primary_Acc']
prot_sequences = elm_df_cls['Sequence']
prot_labels = [{'elm_label': x} for x in elm_df_cls['ELMLabel']] 

dataset = ProtDataset(prot_ids, prot_sequences, prot_data=prot_labels)
SEQUENCE_MASK_TOKEN = tokenizer.get_vocab()['<mask>']
i2t = {i:t for t, i in tokenizer.get_vocab().items()}
base_tokens = torch.tensor([range(4, 24)]).long().flatten()
aa_list = [i2t[i] for i in range(4, 24)]

def joint_mask_seq_collator(data_dict_list):
    sample = collate_dict(data_dict_list)
    inputs = tokenizer.batch_encode_plus(sample["seq"],
                                                add_special_tokens=True,
                                                return_attention_mask=True, 
                                                padding=True)
    sample['seq_ind'] = torch.tensor(inputs['input_ids'])
    sample['masked_ind'] = bert_mask(sample['seq_ind'], sequence_mask_token=SEQUENCE_MASK_TOKEN, base_tokens=base_tokens, mut_per=0.15)
    sample['mask'] = torch.BoolTensor(inputs['attention_mask'])
    sample['elm_label'] = torch.LongTensor(np.stack(sample['elm_label']))
    return sample

from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

class ClassCondBert(pl.LightningModule):
    def __init__(self, model_args) -> None:
        super(ClassCondBert, self).__init__()
        self.model_name = model_args['model_name']
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.classification_head = nn.Linear(2*model_args['hidden_dim'], model_args['num_cls'])

    def forward(self, input_ids, attention_mask):
        logits, hidden_states = self.model(input_ids, attention_mask, output_hidden_states=True).values()
        word_embeddings = hidden_states[-1]
        pooling = self.pool_strategy({"token_embeddings": word_embeddings,
                                      "cls_token_embeddings": word_embeddings[:, 0],
                                      "attention_mask": attention_mask,
                                      })
        return logits, self.classification_head(pooling)
    
    def pool_strategy(self, features, pool_cls=True, pool_mean=True):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        if pool_cls:
            output_vectors.append(cls_token)
        if pool_mean:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)

        # output_vector = torch.stack(output_vectors, -1).sum(dim=-1)
        output_vector = torch.cat(output_vectors, 1)
        return output_vector

    def training_step(self, batch, batch_idx):
        seq_logits, cls_logits = self.forward(batch['masked_ind'], batch['mask'])
        seq_ind = batch['seq_ind']
        seq_ce = F.cross_entropy(seq_logits.reshape(-1, seq_logits.shape[-1]), seq_ind.flatten(), reduce=False).reshape(seq_ind.shape)
        mask = (batch['masked_ind'] != batch['seq_ind'])
        mask_loss = (seq_ce*mask).sum() / mask.sum()

        labels = batch['elm_label'].float()
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, labels)
        loss = mask_loss + cls_loss
        self.log('train_mask_loss', mask_loss)
        self.log('train_cls_loss', cls_loss) 
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer
    
if __name__ == "__main__":
    train_loader = DataLoader(dataset, collate_fn=joint_mask_seq_collator, batch_size=6, shuffle=True)
    checkpoint_callback = ModelCheckpoint(filename="/home/andrew/GO_interp/checkpoints/esm_elm_finetune", 
                                          verbose=True, monitor='train_loss', mode='min')
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger("logs", name="esm_elm_finetune")
    trainer = pl.Trainer(devices=[1], max_epochs=3, 
                         callbacks=[checkpoint_callback], logger=logger)
    model_params = {'model_name': model_name, "hidden_dim": 1280, "num_cls": len(elm_df_cls['ELMLabel'][0])}
    model = ClassCondBert(model_params)
    logger.log_hyperparams(model_params)
    trainer.fit(model, train_loader)