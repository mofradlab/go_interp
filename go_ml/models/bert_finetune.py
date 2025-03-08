from typing import Tuple
import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import f1_score
import pytorch_lightning as pl

from transformers import AutoTokenizer, AutoModel, AutoConfig, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    MultiStepLR,
    ReduceLROnPlateau,
)

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from go_ml.train_utils import CosineAnnealingWarmupRestarts
from transformers import DataCollatorWithPadding
from torchmetrics.classification import MultilabelF1Score

import pandas as pd
import os
import re
from collections import OrderedDict
import logging as log
import numpy as np

class BERTFinetune(pl.LightningModule):
    """
    # https://github.com/minimalist-nlp/lightning-text-classification.git
    
    Sample model to show how to use BERT to classify sentences.
    
    :param hparams: ArgumentParser containing the hyperparameters.
    hparams: {
        nr_frozen_epochs,
        batch_size,
        num_classes,
        gradient_checkpointing,
    }
    """
    def __init__(self, model_args) -> None:
        super(BERTFinetune, self).__init__()
        self.save_hyperparameters()
        self.h = model_args
        self.model_name = model_args.model_name
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False)
        self.h.encoder_features = self.model.encoder.config.hidden_size*2
        self.classification_head = nn.Sequential(
            nn.Linear(self.h.encoder_features, self.h.cls_bottleneck),
            nn.Linear(self.h.cls_bottleneck, self.h.num_classes),
        )
        self.loss = nn.BCEWithLogitsLoss()

    def pool_strategy(self, features,
                      pool_cls=True, pool_mean=True):
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

            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)
            
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            if pool_mean:
                output_vectors.append(sum_embeddings / sum_mask)

        # output_vector = torch.stack(output_vectors, -1).sum(dim=-1)
        output_vector = torch.cat(output_vectors, 1)
        return output_vector
    
    def forward(self, input_ids, attention_mask):
        word_embeddings = self.model(input_ids,
                                           attention_mask)[0]

        pooling = self.pool_strategy({"token_embeddings": word_embeddings,
                                      "cls_token_embeddings": word_embeddings[:, 0],
                                      "attention_mask": attention_mask,
                                      })
        return self.classification_head(pooling)

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ 
        Runs one training step. This usually consists in the forward function followed
            by the loss function.
        
        :param batch: The output of your dataloader. 
        :param batch_nb: Integer displaying which batch this is
        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs, mask, y = batch['seq'], batch['mask'], batch['labels']
        y_hat = self.forward(inputs, mask)
        loss_val = self.loss(y_hat, y.float())

        current_lr = self.scheduler.get_last_lr()[0]
        self.log("lr", current_lr, prog_bar=True, on_step=True)

        tqdm_dict = {"train_loss": loss_val, "lr": current_lr}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict})
        return output

    def on_validation_epoch_start(self) -> None:
        self.val_outputs = []
        return super().on_validation_epoch_start()

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, mask, y = batch['seq'], batch['mask'], batch['labels']
        y_hat = self.forward(inputs, mask)
        loss_val = self.loss(y_hat, y.float())
        self.log("loss/val", loss_val)
        output = OrderedDict({'logits':y_hat.detach().cpu(), 'labels':y.detach().cpu()})
        self.val_outputs.append(output)
        return output
        
    def on_validation_epoch_end(self) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        
        Returns:
            - Dictionary with metrics to be added to the lightning Qger.  
        """
        outputs = self.val_outputs
        labels = torch.cat([x['labels'] for x in outputs])
        logits = torch.cat([x['logits'] for x in outputs])
        preds = logits > 0
        f1 = f1_score(labels, preds, average='micro')
        self.log('F1/val', f1, prog_bar=True)

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.classification_head.parameters(), "lr": self.h.learning_rate},
            {"params": self.model.parameters(), "lr": self.h.encoder_learning_rate},
        ]
        optimizer = optim.Adam(parameters, lr=self.h.learning_rate, weight_decay=self.h.weight_decay)

        scheduler_config = dict(
            scheduler_name="cosine_restarts",
            first_cycle_steps=int(self.h.num_train_steps * scheduler_config["first_cycle_steps_ratio"]), 
            first_cycle_steps_ratio=0.5,
            cycle_mult=1.0,
            max_lr=2e-5,
            min_lr=1e-7,
            warmup_steps=100,
            gamma=0.8,
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer, **scheduler_config)
        
        self.scheduler = lr_scheduler
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": False,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def on_epoch_end(self):
        pass

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument(
            "--model_name",
            default="facebook/esm2_t33_650M_UR50D",
            type=str,
            help="Model name",
        )
        parser.add_argument(
            "--max_length",
            default=1024,
            type=int,
            help="Maximum sequence length.",
        )
        parser.add_argument(
            "--cls_bottleneck",
            default=512,
            type=int,
            help="Rank of cls matrix",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-05,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=9e-05,
            type=float,
            help="Classification head learning rate.",
        )
        parser.add_argument(
            "--nr_frozen_epochs",
            default=0,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen."
        )
        parser.add_argument("--gradient_checkpointing", default=True, type=bool, 
            help="Enable or disable gradient checkpointing which use the cpu memory \
                with the gpu memory to store the model.",
        )
        parser.add_argument(
            "--gradient_clipping", default=1.0, type=float, help="Global norm gradient clipping"
        )
        parser.add_argument(
            "--weight_decay", default=0.00, type=float, help="Weight decay per train step."
        )
        return parser