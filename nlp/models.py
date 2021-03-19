import os, sys, gc
import pickle
import subprocess
import numpy as np
import pandas as pd
from typing import List
from functools import lru_cache
from argparse import Namespace
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA
from transformers import (
  AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup,
  T5EncoderModel, MT5EncoderModel
)

from .activation import Mish
from .utils import evaluation, is_blackbone, Printer, WorkplaceManager, Timer
from .data import BaseDataset, FastTokCollateFn
from .sampler import SortishSampler

class BaseTransformer(nn.Module):
  def __init__(self, global_config, **kwargs):
    super(BaseTransformer, self).__init__()

    self.global_config = global_config

    self._setup_model()

    self.low_dropout = nn.Dropout(self.global_config.low_dropout)
    self.high_dropout = nn.Dropout(self.global_config.high_dropout)

    self.l0 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
    self.classifier = nn.Linear(self.config.hidden_size, self.global_config.n_classes)

    self._init_weights(self.l0)
    self._init_weights(self.classifier)

  def _setup_model(self):
    try:
      model_name = self.global_config.model_path
    except AttributeError:
      model_name = self.global_config.model_name

    self.config = AutoConfig.from_pretrained(self.global_config.config_name)

    if 't5' in model_name:
      _class = T5EncoderModel
    elif 'mt5' in model_name:
      _class = MT5EncoderModel
    else:
      _class = AutoModel

    if self.global_config.pretrained:
      self.model = _class.from_pretrained(model_name)
    else:
      self.model = _class.from_config(self.config)

  def _init_weights(self, layer):
    layer.weight.data.normal_(mean=0.0, std=0.02)
    if layer.bias is not None:
      layer.bias.data.zero_()

  def freeze(self):
    for child in self.model.children():
      for param in child.parameters():
        param.requires_grad = False

  def unfreeze(self):
    for child in self.model.children():
      for param in child.parameters():
        param.requires_grad = True

  def forward(self, inputs):
    outputs = self.model(**inputs)
    x = outputs[0][:, 0, :]

    x = self.l0(self.low_dropout(x))
    x = torch.tanh(x)

    try:
      x = torch.mean(
          torch.stack(
              [self.classifier(self.high_dropout(x)) for _ in range(self.global_config.multi_drop_nb)],
              dim=0,
          ),
          dim=0,
      )
    except:
      x = self.classifier(self.high_dropout(x))

    return x

class LightTrainingModule(nn.Module):
    def __init__(self, global_config, model=None):
        super().__init__()
	
        self.model = model or BaseTransformer(global_config)
        self.loss = global_config.loss
        self.loss_name = global_config.loss_name
        self.activation = global_config.activation
        self.global_config = global_config
        self.losses = {'loss': [], 'val_loss': []}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)

    def move_to_device(self, x, device):
        return {key:val.to(device) for key,val in x.items()}

    def freeze(self):
      self.model.freeze()

    def unfreeze(self):
      self.model.unfreeze()

    def step(self, batch, step_name="train", epoch=-1):
        x, y = batch
        x, y = self.move_to_device(x, self.device), y.to(self.device)
        y_probs = self.forward(x)

        loss = self.loss(y_probs, y, epoch)

        try:
        	y_probs = self.activation(y_probs, dim=1) #softmax
        except:
        	y_probs = self.activation(y_probs) #sigmoid

        loss_key = f"{step_name}_loss"

        return { ("loss" if step_name == "train" else loss_key): loss.cpu()}, y_probs.cpu()

    def forward(self, X, *args):
        return self.model(X, *args)

    def training_step(self, batch, batch_idx, epoch):
        return self.step(batch, "train", epoch)
    
    def validation_step(self, batch, batch_idx, epoch):
        return self.step(batch, "val", epoch)

    def training_epoch_end(self, outputs: List[dict]):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.losses['loss'].append(loss.item())

        return {"train_loss": loss}

    def validation_epoch_end(self, outputs: List[dict]):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.losses['val_loss'].append(loss.item())

        return {"val_loss": loss}
        
    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")
    
    def train_dataloader(self):
        return self.create_data_loader(self.global_config.train_df, shuffle=True)

    def val_dataloader(self):
        return self.create_data_loader(self.global_config.val_df)

    def test_dataloader(self):
        return self.create_data_loader(self.global_config.test_df, 'test')
                
    def create_data_loader(self, df: pd.DataFrame, task='train', shuffle=False):
        sampler = None
        batch_size=self.global_config.batch_size

        if hasattr(self.global_config, 'use_bucketing') and task!='test':
          sampler = SortishSampler(df['length'].values.tolist(), batch_size, shuffle)
          shuffle = False

        return DataLoader(
            BaseDataset(df, task, self.loss_name, c=self.global_config.n_classes),
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=FastTokCollateFn(self.model.config, self.global_config.config_name, self.global_config.max_tokens, self.global_config.on_batch),
    		    num_workers=4,
    		    pin_memory=True
        )
        
    def total_steps(self, epochs):
        return len(self.train_dataloader()) // self.global_config.accumulate_grad_batches * epochs

    def configure_optimizers(self, lr=None, epochs=None):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = self.model.named_parameters()
        optimizer_grouped_parameters = [
             {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
             {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr or self.global_config.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.global_config.warmup_steps,
                    num_training_steps=self.total_steps(epochs or self.global_config.epochs),
        )
        if self.global_config.swa: optimizer = SWA(optimizer, self.global_config.swa_start, self.global_config.swa_freq, self.global_config.swa_lr)
        return [optimizer], [lr_scheduler]

class Trainer:
  def __init__(self, global_config, **kwargs):

    self.metric_name = global_config.metric_name
    self.global_config = global_config
    self.fold = global_config.fold

    self._reset()
    self._set_module(kwargs)
    self._set_loaders()
    self._set_optimizers()
    self._set_logger()

  def _reset(self):
    self.probs = None
    self.best_log = np.inf
    self.best_metric = 0
    self.best_eval = []
    self.scores = []

  def _set_loaders(self):
    self.train_dl = self.module.train_dataloader()
    self.val_dl = self.module.val_dataloader()
    self.test_dl = self.module.test_dataloader()

  def _set_module(self, kwargs):
    try:
      self.module = kwargs['module']
    except:
      self.module = LightTrainingModule(self.global_config)

  def _set_optimizers(self, lr=None, epochs=None):
    self.opts, scheds = self.module.configure_optimizers(lr, epochs)
    self.scheduler = scheds[0]

  def _change_lr(self, lr=None):
    lr = lr or self.global_config.lr

    for opt in self.opts:
      for param_group in opt.param_groups:
        param_group['lr'] = lr

  def _set_logger(self):
    self.printer = Printer(self.global_config.fold)


  def train(self, epoch):
    self.module.train()
    self.module.zero_grad()
    outputs = []

    for i, batch in enumerate(tqdm(self.train_dl, desc='Training')):
      output, _ = self.module.training_step(batch, i, epoch)
      outputs.append(output)

      output['loss'].backward()

      if (i+1) % self.module.global_config.accumulate_grad_batches == 0:
        if self.global_config.clip_grad: 
          nn.utils.clip_grad_norm_(self.module.model.parameters(), self.global_config.max_grad_norm)

        self.opts[0].step()

        if self.global_config.scheduler and epoch >= self.global_config.finetune_epochs: self.scheduler.step()
      self.module.zero_grad()

      self.printer.pprint(**output)
    
    self.module.training_epoch_end(outputs)

  def evaluate(self, epoch):
    self.module.eval()

    with torch.no_grad():
      score = []
      outputs = []
      eval_probs = []

      for i, batch in enumerate(tqdm(self.val_dl, desc='Eval')):
        output, y_probs = self.module.validation_step(batch, i, epoch)
        y_probs = y_probs.detach().cpu().numpy()      
        score += [ self.get_score(batch, y_probs) ]
        eval_probs.append(y_probs.reshape(-1, self.global_config.n_classes))
        outputs.append(output)

        self.printer.pprint(**output)

      score = self.get_mean_score(score)
      self.scores.append(score)
      self.module.validation_epoch_end(outputs)
      self._check_evaluation_score(score[self.metric_name], score['Logloss'], eval_probs)
    
  def predict(self):
    if self.probs is None:
      self.module.eval()
      self.probs = []

      with torch.no_grad():
        for i, batch in enumerate(self.test_dl):
          _, y_probs = self.module.test_step(batch, i)
          self.probs += y_probs.detach().cpu().numpy().tolist()
    else:
      print('[WARNINGS] Already predicted. Use "trainer.get_preds()" to obtain the preds.')

  def fit_one_epoch(self, epoch):
    timer = Timer()

    self.train(epoch)
    if self.global_config.swa and (self.global_config.epochs-1) == epoch:
      self.opts[0].swap_swa_sgd()
    self.evaluate(epoch)
    
    self.printer.update_and_show(epoch, self.module.losses, self.scores[epoch], timer.to_string())

  def finetune_head_one_epoch(self, epoch):
      self.module.freeze()
      self.fit_one_epoch(epoch)
      self.module.unfreeze()

  def fit(self, epochs=None, lr=None, reset_lr=True):
    epochs = epochs or self.global_config.epochs
    add = len(self.scores)

    if reset_lr: self._change_lr(lr)

    for epoch in range(epochs):
      self.fit_one_epoch(epoch + add)

  def finetune(self):
    self.module.freeze()
    self.fit(self.global_config.finetune_epochs, lr=self.global_config.head_lr)
    self.module.unfreeze()

  def get_preds(self):
    return self.probs

  def get_score(self, batch, y_probs):
    return evaluation(batch[-1].cpu().numpy(), y_probs, labels=list(range(self.global_config.n_classes)))

  def get_mean_score(self, scores):
    keys = scores[0].keys()
    return {key:np.mean([score[key] for score in scores]) for key in keys}

  def _save_weights(self, half_precision=False, path='models/'):
    print('Saving weights ...')
    if half_precision: self.module.half() #for fast inference
    torch.save(self.module.state_dict(), f'{path}model_{self.fold}.bin')
    gc.collect()

  def _check_evaluation_score(self, metric, log_score, best_eval=None):
    if metric > self.best_metric:
      self.best_metric = metric
      self.best_log = log_score
      self.best_eval = best_eval
      self._save_weights()

  def save_best_eval(self, path='evals/{}/fold_{}'):
    if self.global_config.task=='train':
      np.save(path.format(self.global_config.model_name, self.global_config.fold)+'_best_eval.npy', np.vstack(self.best_eval))
