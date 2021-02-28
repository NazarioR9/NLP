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
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA
from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup
from transformers import is_torch_tpu_available

from .activation import Mish
from .utils import evaluation, is_blackbone, Printer, WorkplaceManager, Timer, import_xla_if_available
from .data import BaseDataset, FastTokCollateFn

USE_TPU = is_torch_tpu_available()

if USE_TPU:
  import torch_xla
  import torch_xla.distributed.data_parallel as dp
  import torch_xla.distributed.parallel_loader as pl
  import torch_xla.core.xla_model as xm
  import torch_xla.distributed.xla_multiprocessing as xmp

class BaseTransformer(nn.Module):
  def __init__(self, global_config, **kwargs):
    super(BaseTransformer, self).__init__()

    self.args = global_config
    self.do_md = hasattr(global_config, 'multi_drop_nb')

    self._setup_model()

    self.low_dropout = nn.Dropout(self.args.low_dropout)
    self.high_dropout = nn.Dropout(self.args.high_dropout)

    self.l0 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
    self.classifier = nn.Linear(self.config.hidden_size, self.args.n_classes)

    self._init_weights(self.l0)
    self._init_weights(self.classifier)

  def _setup_model(self):
    try:
      model_name = self.args.model_path
    except AttributeError:
      model_name = self.args.model_name

    self.config = AutoConfig.from_pretrained(self.args.config_name)

    if self.args.pretrained:
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

    if self.do_md:
      x = torch.mean(
          torch.stack(
              [self.classifier(self.high_dropout(x)) for _ in range(self.args.multi_drop_nb)],
              dim=0,
          ),
          dim=0,
      )
    else:
      x = self.classifier(self.high_dropout(x))

    return x

class LightTrainingModule(nn.Module):
    def __init__(self, global_config, model=None):
        super().__init__()
	
        self.model = model or BaseTransformer(global_config)
        self.loss = global_config.loss
        self.loss_name = global_config.loss_name
        self.activation = global_config.activation
        self.args = global_config
        self.losses = {'loss': [], 'val_loss': []}

    def setup_device(self, device):
      self.device = device

    def move_to_device(self, x, device):
      if isinstance(x, dict):
        return {key:val.to(device) for key,val in x.items()}
      return x.to(device)

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

        return { ("loss" if step_name == "train" else loss_key): loss}, y_probs.detach()

    def forward(self, X, *args):
        return self.model(X, *args)

    def training_step(self, batch, batch_idx, epoch):
        return self.step(batch, "train", epoch)
    
    def validation_step(self, batch, batch_idx, epoch):
        return self.step(batch, "val", epoch)

    def training_epoch_end(self, outputs: List[dict]):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        if USE_TPU:
          loss = xm.mesh_reduce('loss', loss, np.mean)
        self.losses['loss'].append(loss.item())

        return {"train_loss": loss}

    def validation_epoch_end(self, outputs: List[dict]):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        if USE_TPU:
          loss = xm.mesh_reduce('val_loss', loss, np.mean)
        self.losses['val_loss'].append(loss.item())

        return {"val_loss": loss}
        
    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")
    
    def train_dataloader(self):
        return self.create_data_loader(self.args.train_df, shuffle=True)

    def val_dataloader(self):
        return self.create_data_loader(self.args.val_df)

    def test_dataloader(self):
        return self.create_data_loader(self.args.test_df, 'test')

    def _get_sampler(self, ds, shuffle):
        if USE_TPU:
          return DistributedSampler(
              ds,
              num_replicas=xm.xrt_world_size(),
              rank=xm.get_ordinal(),
              shuffle=shuffle
            )
        else:
          return None
                
    def create_data_loader(self, df: pd.DataFrame, task='train', shuffle=False):
        ds = BaseDataset(df, task, self.loss_name, c=self.args.n_classes)
        sampler = self._get_sampler(ds, shuffle)
        shuffle = (shuffle and sampler is None)
        return DataLoader(
            ds,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            sampler = sampler,
            collate_fn=FastTokCollateFn(self.model.config, self.args.model_name, self.args.max_tokens, self.args.on_batch),
    		    num_workers=8 if USE_TPU else 4,
    		    pin_memory=True
        )
        
    def total_steps(self, epochs):
        return len(self.train_dataloader()) // self.args.accumulate_grad_batches * epochs

    def configure_optimizers(self, lr=None, epochs=None):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = self.model.named_parameters()
        optimizer_grouped_parameters = [
             {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
             {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr or self.args.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.args.warmup_steps,
                    num_training_steps=self.total_steps(epochs or self.args.epochs),
        )
        if self.args.swa: optimizer = SWA(optimizer, self.args.swa_start, self.args.swa_freq, self.args.swa_lr)
        return optimizer, lr_scheduler

class Trainer:
  def __init__(self, global_config, **kwargs):

    self.metric_name = global_config.metric_name
    self.args = global_config
    self.fold = global_config.fold
    self.replicas = 1

    self._reset()
    self._setup_device()
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

  def _setup_device(self):
    if USE_TPU:
      self.device = xm.xla_device()
      self.replicas = xm.xrt_world_size()
    elif torch.cuda.is_available():
      self.device = torch.device('cuda:0')
    else:
      self.device = 'cpu'

  def _set_loaders(self):
    self.train_dl = self.module.train_dataloader()
    self.val_dl = self.module.val_dataloader()
    self.test_dl = self.module.test_dataloader()

    self.train_steps = len(self.args.train_df) // (self.args.batch_size * self.replicas)
    self.val_steps = len(self.args.val_df) // (self.args.batch_size * self.replicas)
    self.test_steps = len(self.args.test_df) // (self.args.batch_size * self.replicas)

  def _set_module(self, kwargs):
    try:
      self.module = kwargs['module']
    except:
      self.module = LightTrainingModule(self.args)
    self.module.setup_device(self.device)
    self.module.to(self.device)

  def _set_optimizers(self, lr=None, epochs=None):
    self.opt, self.scheduler = self.module.configure_optimizers(lr, epochs)

  def _change_lr(self, lr=None):
    lr = lr or self.args.lr

    if USE_TPU:
      lr = lr *xm.xrt_world_size()

    for param_group in self.opt.param_groups:
      param_group['lr'] = lr

  def _set_logger(self):
    self.printer = Printer(self.args.fold)

  def _optimizer_step(self, step):
    if (step+1) % self.args.accumulate_grad_batches == 0:
        if self.args.clip_grad: 
          nn.utils.clip_grad_norm_(self.module.model.parameters(), self.args.max_grad_norm)

        if USE_TPU:
          xm.optimizer_step(self.opt)
        else:
          self.opt.step()

        if self.args.scheduler and epoch >= self.args.finetune_epochs: self.scheduler.step()
    self.module.zero_grad()

  def _swa(epoch):
    if self.args.swa and (self.args.epochs-1) == epoch:
      self.opt.swap_swa_sgd()

  def _get_iterator(self, dl):
    if USE_TPU:
      return  pl.ParallelLoader(dl, [self.device]).per_device_loader(self.device)

    return dl

  def train(self, epoch):
    self.module.train()
    self.module.zero_grad()
    outputs = []

    for i, batch in enumerate(tqdm(self._get_iterator(self.train_dl), total=self.train_steps, desc='Training')):
      output, _ = self.module.training_step(batch, i, epoch)
      outputs.append(output)

      output['loss'].backward()

      self._optimizer_step(i)

      self.printer.pprint(**output)
    
    self.module.training_epoch_end(outputs)
    self._swa(epoch)

  def evaluate(self, epoch):
    self.module.eval()

    with torch.no_grad():
      score = []
      outputs = []
      eval_probs = []

      for i, batch in enumerate(tqdm(self._get_iterator(self.val_dl), total=self.val_steps, desc='Eval')):
        output, y_probs = self.module.validation_step(batch, i, epoch)
        y_probs = y_probs.cpu().numpy()
        score += [ self.get_score(batch, y_probs) ]
        eval_probs.append(y_probs.reshape(-1, self.args.n_classes))
        outputs.append(output)

        self.printer.pprint(**output)

      score = self.xm_reduce(self.get_mean_score(score))
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
    self.evaluate(epoch)
    
    self.printer.update_and_show(epoch, self.module.losses, self.scores[epoch], timer.to_string())

  def finetune_head_one_epoch(self, epoch):
      self.module.freeze()
      self.fit_one_epoch(epoch)
      self.module.unfreeze()

  def fit(self, epochs=None, lr=None, reset_lr=True):
    epochs = epochs if epochs is not None else self.args.epochs
    add = len(self.scores)

    if reset_lr: self._change_lr(lr)

    for epoch in range(epochs):
      self.fit_one_epoch(epoch + add)

  def finetune(self):
    self.module.freeze()
    self.fit(self.args.finetune_epochs, lr=self.args.head_lr)
    self.module.unfreeze()

  def get_preds(self):
    return self.probs

  def get_score(self, batch, y_probs):
    return evaluation(batch[-1].cpu().numpy(), y_probs, labels=list(range(self.args.n_classes)))

  def get_mean_score(self, scores):
    keys = scores[0].keys()
    return {key:np.mean([score[key] for score in scores]) for key in keys}

  def xm_reduce(self, scores, suffixe='eval'):
    if USE_TPU:
      return {key: xm.mesh_reduce(suffixe+key, val, np.mean) for key, val in scores.items()}
    else:
      return scores

  def _save_weights(self, path='models/'):
    print('Saving weights ...')
    if USE_TPU:
      xm.save(self.module.state_dict(), f'{path}model_{self.fold}.bin')
    else:
      torch.save(self.module.state_dict(), f'{path}model_{self.fold}.bin')
    gc.collect()

  def _check_evaluation_score(self, metric, log_score, best_eval=None):
    if metric > self.best_metric:
      self.best_metric = metric
      self.best_log = log_score
      self.best_eval = best_eval
      self._save_weights()

  def save_best_eval(self, path='evals/{}/fold_{}'):
    if self.args.task=='train':
      np.save(path.format(self.args.model_name, self.args.fold)+'_best_eval.npy', np.vstack(self.best_eval))
