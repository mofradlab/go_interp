from argparse import ArgumentParser
from go_ml.models.bert_finetune import BERTFinetune
from go_ml.train_utils import get_dataloaders
train_loader, val_loader = get_dataloaders()
train_dataset = train_loader.dataset

parser = ArgumentParser()
parser = BERTFinetune.add_model_specific_args(parser)
hparams = parser.parse_args()
print("got hparams", hparams)
hparams.num_classes = train_dataset[0]['labels'].shape[0]
hparams.num_train_steps = 10*len(train_dataset)
hparams.encoder_features = 320

import pickle
checkpoint_dir = "/home/andrew/GO_interp/checkpoints"
with open(f"{checkpoint_dir}/esm_finetune_hparams.pkl", "wb") as f:
    pickle.dump(hparams, f)