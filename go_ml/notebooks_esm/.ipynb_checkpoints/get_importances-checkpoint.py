import pandas as pd
enzyme_df= pd.read_csv('../../data/enzyme_dataset_seq.csv')
enzyme_df= enzyme_df[~enzyme_df['Sequence'].isna()]
enzyme_go_terms = [gt.split("'")[1] for gt in enzyme_df['GOTerm']]
import json
train_path = "/home/andrew/cafa5_team/data/"
with open(f"{train_path}/cafa_dataset/go_terms.json", "r") as f:
    go_terms = json.load(f)
term_ind_map = {t:i for i, t in enumerate(go_terms)}
enzyme_df['GOTerm'] = enzyme_go_terms
enzyme_df= enzyme_df[enzyme_df['GOTerm'].isin(term_ind_map)]
enzyme_term_index = [term_ind_map[t] for t in enzyme_df['GOTerm']]
enzyme_df['GOTermIndex'] = enzyme_term_index
annotated_indices = [list(filter(lambda x: x < min(1024, len(seq)), map(int, x[1:-1].split(',')))) for x, seq in zip(enzyme_df['AnnotatedIndices'], enzyme_df['Sequence'])]
enzyme_df['AnnotatedIndices'] = annotated_indices

import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
def enzyme_iterator():
     for i, pid, annot_ind, enzyme_cls, goterm, seq, go_ind in enzyme_df.itertuples():
        #   print(pid, annot_ind, enzyme_cls, goterm, seq, go_ind)
          inputs = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding='max_length',
                                             truncation=True, return_attention_mask=True, max_length=1024)
          yield {'prot_id': pid, 'annot_ind': annot_ind, 'go_ind': go_ind, 'seq': seq, 'seq_ind': inputs['input_ids'], 'mask': inputs['attention_mask']}

import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from go_ml.models.bert_finetune import BERTFinetune
device = torch.device('cpu')

checkpoint_dir = "/home/andrew/GO_interp/checkpoints"
with open(f"{checkpoint_dir}/esm_finetune_hparams.pkl", "rb") as f:
    hparams = pickle.load(f)
model = BERTFinetune.load_from_checkpoint(f"{checkpoint_dir}/esm_finetune.ckpt", model_args=hparams, 
                                          map_location=device)
class SingleGOModel(nn.Module):
    def __init__(self, model, go_idx):
        super().__init__()
        self.model = model
        self.go_idx = go_idx

    def forward(self, seq, mask, go_idx):
        return torch.sigmoid(self.model(seq, mask)[:, go_idx])
    
goind_model = SingleGOModel(model, 0)
goind_model.eval()
print("Model ready")

def get_preds(model, iter):
    with torch.no_grad():
        pred_l = []
        for i, r in enumerate(iter):
            seq_ind, mask =  torch.tensor(r['seq_ind']).to(device), torch.BoolTensor(r['mask']).to(device)
            pred = model(seq_ind, mask, r['go_ind'])
            pred_l.append(pred.cpu())
        return pred_l
pred_l = get_preds(goind_model, enzyme_iterator())
preds = torch.stack(pred_l)

import numpy as np

def mutate(seq, lambda_):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    mut_seq = np.array(list(seq))
    num_changes = np.random.poisson(lambda_)
    indices_to_change = np.random.choice(len(seq), num_changes, replace=False)
    mut_seq[indices_to_change] = np.random.choice(list(amino_acids), num_changes)
    mut_seq = ''.join(mut_seq)
    return mut_seq, num_changes

def mut_iterator(mut_df, go_ind):
     for i, seq in mut_df.itertuples():
        inputs = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding='max_length',
                                             truncation=True, return_attention_mask=True, max_length=1024)
        # note the go_ind is not actually associated but added here to make the pred function
        # able to locate the relevant go_ind
        yield {'seq': seq, 'seq_ind': inputs['input_ids'], 'mask': inputs['attention_mask'], 'go_ind': go_ind}

def gen_mut_sequences(seq, lambda_=20, min_mutations=2):
    # num refers to number of mutated sequences to generate
    mutated_sequences = []
    mutations_counter = {}

    np.random.seed(0)
    
    count = 0
    
    mutated_sequences_num_changes  = [mutate(seq, lambda_) for i in range(len(seq))]
    mutated_sequences, num_changes_array = zip(*mutated_sequences_num_changes)
        
    mutations_counter = {}
    for mutated_seq in mutated_sequences:
        for j in range(len(seq)):
            if seq[j] != mutated_seq[j]:
                if j not in mutations_counter:
                    mutations_counter[j] = 1
                else:
                    mutations_counter[j] += 1
        
        # check if each amino acid is mutated at least min_mutations times
        if all(count >= min_mutations for count in mutations_counter.values()):
            mutated_sequences = [seq] + list(mutated_sequences)
            num_changes_array = [0] + list(num_changes_array)


    
    df = pd.DataFrame(mutated_sequences, columns=['Sequences'])
    return df, num_changes_array

from embedding import embedding

def gen_X_pred_mut(seq, go_ind, lambda_=20, min_mutations=2):
    # generate mutated sequences
    sequences, num_mut = gen_mut_sequences(seq)
    # this includes the true and the mutated sequences
    X = sequences["Sequences"].values
    X = torch.tensor(np.array([embedding(seq) for seq in X]))
    X = X.type(torch.FloatTensor)
    # generate the prediction 
    pred_l = get_preds(goind_model, mut_iterator(sequences, go_ind))
    pred = torch.stack(pred_l)
    return X, pred, num_mut


class LIMELogisticRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.flatted_input_size = input_size[1] * input_size[2]
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatted_input_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
    
    def get_weights(self):
        params = self.state_dict()
        weights = params['layers.1.weight']
        weights_unflattened = torch.unflatten(weights, 1, (self.input_size[1], self.input_size[2]))
        return weights_unflattened
        
    def get_importance_abs(self):
        weights = self.get_weights()
        absolute_weights = torch.abs(weights)
        feature_importance = absolute_weights.sum(dim=2)
        return feature_importance
        
        
    def get_importance_sum(self):
        weights = self.get_weights()
        feature_importance = weights.sum(dim=2)
        return feature_importance
    

import torch.optim as optim

def train(model, X, y, num_mut, epochs=1000, lambda_=0.01):
    # is this best way to determine proximity?
    proximity = [1 / (mut + 1) for mut in num_mut]
    proximity = torch.tensor(proximity).unsqueeze(1)
    # change this potentially, 
    criterion = nn.BCELoss(weight=proximity)
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        output = model(X)
        loss = criterion(output, y)
        
        l1_regularization = torch.tensor(0., requires_grad=False)
        for param in model.parameters():
            l1_regularization += torch.norm(param, p=1)
            
        loss += lambda_ * l1_regularization
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

np.random.seed(0)
torch.manual_seed(0)


def get_LIME_importances(iter):
    importances_l = []
    sums_l = []
    abs_l = []
    go_index_l = []
    annotated_indices_l = []
    for i, r in tqdm(enumerate(iter)):
        try:
            if i == 5:
                break
            
            go_index = r["go_ind"]
            annotated_indices = r["annot_ind"]
                
            X, pred, num_mut = gen_X_pred_mut(r['seq'], r['go_ind'])
            model = LIMELogisticRegression(X.size())
            train(model, X, pred, num_mut)
            prot_importance = model.get_weights()
            importance_sum = model.get_importance_sum()
            importance_abs = model.get_importance_abs()

            # padding the sequences to 1024
            padding_needed = 1024 - prot_importance.shape[1]
            prot_importance = torch.nn.functional.pad(prot_importance, (0, 0, 0, padding_needed))
            importance_sum = torch.nn.functional.pad(importance_sum, (0, padding_needed))
            importance_abs = torch.nn.functional.pad(importance_abs, (0, padding_needed))

            go_index_l.append(go_index)
            annotated_indices_l.append(annotated_indices)
            sums_l.append(importance_sum)
            abs_l.append(importance_abs)
            importances_l.append(prot_importance)
   

        except Exception as e:
            print(f"An error occurred: {e}")

    importances = torch.stack(importances_l).squeeze().cpu()
    return importances, sums_l, abs_l, go_index_l, annotated_indices_l, r["prot_id"]
    
importances_LIME = get_LIME_importances(enzyme_iterator())

import pickle
importances = {'importances_LIME': importances_LIME}
with open('../../data/enzyme_importances_weighted.pkl', 'wb') as f:
    pickle.dump(importances, f)
