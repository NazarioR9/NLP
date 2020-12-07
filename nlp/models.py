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
        AutoConfig, AutoModel, AdamW, Adafactor,
        get_linear_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup
  )

from .activation import Mish
from .utils import evaluation, is_blackbone, Printer, WorkplaceManager
from .data import BaseDataset, FastTokCollateFn

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

    if self.global_config.pretrained:
      self.model = AutoModel.from_pretrained(model_name)
    else:
      self.model = AutoModel.from_config(self.config)

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
        return DataLoader(
                    BaseDataset(df, task, self.loss_name, c=self.global_config.n_classes),
                    batch_size=self.global_config.batch_size if task=='train' else int(0.25*self.global_config.batch_size),
                    shuffle=shuffle,
                    collate_fn=FastTokCollateFn(self.model.config, self.global_config.model_name, self.global_config.max_tokens, self.global_config.on_batch)
        )
        
    @lru_cache()
    def total_steps(self):
        return len(self.train_dataloader()) // self.global_config.accumulate_grad_batches * self.global_config.epochs

    def _get_optimizer(self, params):
      opt_name = self.global_config.opt_name
      opt_config = self.global_config.opt_config

      if opt_name == 'adafactor':
        return Adafactor(params, **opt_config)

      return AdamW(params, **opt_config)

    def _get_scheduler(self, opt):
      sch_name = self.global_config.sch_name
      sch_config = self.global_config.sch_config

      if sch_name == 'cosine':
        return get_cosine_schedule_with_warmup(opt, num_training_steps=self.total_steps(), **sch_config)
      elif sch_name == 'cosine_hard':
        return get_cosine_with_hard_restarts_schedule_with_warmup(opt, num_training_steps=self.total_steps(), **sch_config)

      return get_linear_schedule_with_warmup(opt, num_training_steps=self.total_steps(), **sch_config)

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = self.model.named_parameters()
        optimizer_grouped_parameters = [
             {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
             {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = self._get_optimizer(optimizer_grouped_parameters)
        lr_scheduler = self._get_scheduler(optimizer)

        if self.global_config.swa: optimizer = SWA(optimizer, self.global_config.swa_start, self.global_config.swa_freq, self.global_config.swa_lr)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

class Trainer:
  def __init__(self, global_config, **kwargs):

    if global_config.task=='train':
      self.best_eval = []
      self.scores = []
      self.best_log = np.inf
      self.best_metric = 0
      self.metric_name = global_config.metric_name
      self.global_config = global_config
      self.fold = global_config.fold
      self.printer = Printer(global_config.fold)
      self.module = LightTrainingModule(global_config)
      self.opts, scheds = self.module.configure_optimizers()
      self.scheduler = scheds[0]['scheduler']
      self.train_dl = self.module.train_dataloader()
      self.val_dl = self.module.val_dataloader()
    else:
      self.probs = None
      self.module = kwargs['module']
      self.test_dl = self.module.test_dataloader()

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

        if self.global_config.scheduler: self.scheduler.step()
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
    self.train(epoch)
    if self.global_config.swa and (self.global_config.epochs-1) == epoch:
    	self.opts[0].swap_swa_sgd()
    self.evaluate(epoch) 
    self.printer.update_and_show(epoch, self.module.losses, self.scores[epoch])

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
