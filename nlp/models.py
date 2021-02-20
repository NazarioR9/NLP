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
    AutoConfig, AutoModel, AdamW, Adafactor, get_linear_schedule_with_warmup,
    AutoModelForSequenceClassification, AutoModelForSeq2SeqLM,
)

from .activation import Mish
from .utils import *
from .data import *
from .sampler import SortishSampler

class Transformer(nn.Module):
  _task = 'pretraining'
  _model_name = ''
  _auto_loader = AutoModel

  def __init__(self, global_config, **kwargs):
    super(Transformer, self).__init__()

    self._setup_model(global_config)

  def _setup_model(self, gconfig):
    try:
      self._model_name = gconfig.model_path
    except AttributeError:
      self._model_name = gconfig.model_name

    config_args = dict(pretrained_model_name_or_path=gconfig.config_name)
    if self._task == 'seqClassification':
      config_args['num_labels'] = gconfig.num_labels

    self.config = AutoConfig.from_pretrained(**config_args)

    if gconfig.pretrained:
      self.model = self._auto_loader.from_pretrained(self._model_name, config=self.config)
    else:
      self.model = self._auto_loader.from_config(self.config)

  def base_model(self):
    return self.model.base_model

  def freeze(self):
    for child in self.base_model().children():
      for param in child.parameters():
        param.requires_grad = False

  def unfreeze(self):
    for child in self.base_model().children():
      for param in child.parameters():
        param.requires_grad = True

  def forward(self, inputs):
    outputs = self.model(**inputs)
    return outputs

  def use_task_specific_params(self, pars):
    self.model.config.update(pars)

class SeqClassificationTransformer(Transformer):
  _task = 'seqClassification'
  _auto_loader = AutoModelForSequenceClassification


class Seq2SeqTransformer(Transformer):
  _task = 'seq2Seq'
  _auto_loader = AutoModelForSeq2SeqLM

  def generate(self, batch):
    return self.model.generate(**batch)


class LightTrainingModule(nn.Module):
    _dataset_class = None
    _collator_class = None
    _tranformer_class = None
    _sampler = None

    def __init__(self, global_config, model=None, augs={}):
        super().__init__()
	
        self.augs = augs
        self.model = model or self._tranformer_class(global_config)
        self.activation = global_config.activation
        self.global_config = global_config
        self.losses = {'loss': [], 'val_loss': []}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._setup_tok(global_config.config_name)

        self.model.to(self.device)

    def _setup_tok(self, tok_name):
      self.tokenizer = getTokenizer(self.model.config, tok_name)

    def use_task_specific_params(self, pars):
      self.model.use_task_specific_params(pars)

    def move_to_device(self, x):
      if isinstance(x, dict):
        return {key:val.to(self.device) for key,val in x.items()}
      return x.to(self.device)

    def freeze(self):
      self.model.freeze()

    def unfreeze(self):
      self.model.unfreeze()

    def step(self, batch, phase="train", epoch=-1):
        pass

    def forward(self, x, *args):
        return self.model(x, *args)

    def training_step(self, batch, batch_idx, epoch):
        return self.step(batch, "train", epoch)
    
    def validation_step(self, batch, batch_idx, epoch):
        return self.step(batch, "val", epoch)

    def training_epoch_end(self, outputs: List[dict]):
        loss = torch.stack([x["loss"].item() for x in outputs]).mean()
        self.losses['loss'].append(loss)

        return {"train_loss": loss}

    def validation_epoch_end(self, outputs: List[dict]):
        loss = torch.stack([x["val_loss"].item() for x in outputs]).mean()
        self.losses['val_loss'].append(loss)

        return {"val_loss": loss}
        
    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")
    
    def train_dataloader(self):
        return self.create_data_loader(self.global_config.train_df, shuffle=True)

    def val_dataloader(self):
        return self.create_data_loader(self.global_config.val_df, 'val')

    def test_dataloader(self):
        return self.create_data_loader(self.global_config.test_df, 'test')
                
    def create_data_loader(self, df: pd.DataFrame, phase='train', shuffle=False):
        sampler = None
        batch_size=self.global_config.batch_size

        if self._sampler:
          sampler = self._sampler(df['length'].values.tolist(), batch_size, shuffle)
          shuffle = False

        return DataLoader(
                    self._dataset_class(df, phase, aug=self.augs.get(phase, None), c=self.global_config.num_labels),
                    batch_size=batch_size,
                    shuffle=shuffle,
                    sampler = sampler,
                    collate_fn=self._collator_class(self.model.config, self.global_config.model_name, self.global_config.max_tokens, self.global_config.on_batch, phase),
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

        optimizer_kwargs = {'lr': lr or self.global_config.lr}

        if self.global_config.use_adafactor:
          optimizer_cls = Adafactor
          optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        else:
          optimizer_cls = AdamW

        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.global_config.warmup_steps,
                    num_training_steps=self.total_steps(epochs or self.global_config.epochs),
        )
        if self.global_config.swa: optimizer = SWA(optimizer, self.global_config.swa_start, self.global_config.swa_freq, self.global_config.swa_lr)
        return [optimizer], [lr_scheduler]

class SeqClassificationModule(LightTrainingModule):
    _dataset_class = SeqClassificationDataset
    _collator_class = SeqClassificationCollator
    _tranformer_class = SeqClassificationTransformer

    def step(self, batch, phase="train", epoch=-1):
        batch = self.move_to_device(batch)
        outputs = self.forward(batch)

        loss = outputs['loss']
        y_probs = outputs['logits']

        try:
          y_probs = self.activation(y_probs, dim=1) #softmax
        except:
          y_probs = self.activation(y_probs) #sigmoid

        return { ("loss" if phase == "train" else f"{phase}_loss"): loss}, y_probs.cpu()

class Seq2SeqModule(LightTrainingModule):
    _dataset_class = Seq2SeqDataset
    _collator_class = Seq2SeqCollator
    _tranformer_class = Seq2SeqTransformer
    _sampler = SortishSampler

    def generate(self, batch):
      batch.pop('labels', None)
      batch.update({'num_beams': self.global_config.num_beams})

      return self.model.generate(batch)

    def decode(self, generated):
      return self.tokenizer.batch_decode(
          generated,
          skip_special_tokens=True,
          clean_up_tokenization_spaces=False
        )

    def step(self, batch, phase="train", epoch=-1):
      x = self.move_to_device(batch[0])

      loss = self.forward(x)['loss']

      return { ("loss" if phase == "train" else f"{phase}_loss"): loss}, x
        
    def test_step(self, batch, batch_idx):
      batch = self.move_to_device(batch)
      return {}, self.decode(self.generate(batch))

############## Trainer

class Trainer:
  _module_class = None

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
      self.module = self._module_class(self.global_config)

    if hasattr(self.global_config, 'task_specific_params'):
      self.module.use_task_specific_params(self.global_config.task_specific_params)

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

  def _detach(self, x):
    if isinstance(x, torch.Tensor):
      return x.detach().cpu().numpy()
    return x

  def train(self, epoch):
    self.module.train()
    self.module.zero_grad(True)
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
      self.module.zero_grad(True)

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
        y_probs = self._detach(y_probs)
        score += [ self.get_score(batch, y_probs) ]
        try:
          eval_probs.append(y_probs.reshape(-1, self.global_config.num_labels))
        except:
          eval_probs.append(y_probs)
        outputs.append(output)

        self.printer.pprint(**output)

      score = self.get_mean_score(score)
      self.scores.append(score)
      self.module.validation_epoch_end(outputs)
      self._check_evaluation_score(score[self.metric_name], eval_probs)
    
  def predict(self):
    if self.probs is None:
      self.module.eval()
      self.probs = []

      with torch.no_grad():
        for i, batch in enumerate(tqdm(self.test_dl, desc='Inference')):
          _, y_probs = self.module.test_step(batch, i)
          self.probs += [self._detach(y_probs)]
    else:
      print('[WARNINGS] Already predicted. Use "trainer.get_preds()" to obtain the preds.')

  def fit_one_epoch(self, epoch):
    timer = Timer()

    self.train(epoch)

    if self.global_config.swa and (self.global_config.epochs-1) == epoch:
      self.opts[0].swap_swa_sgd()
    
    if self.global_config.evaluate:
      self.evaluate(epoch)
      self.printer.update_and_show(epoch, self.module.losses, self.scores[epoch], timer.to_string())
    else:
      self._save_weights(path=f'checkpoints/checkpoint_{self.global_config.fold}-{epoch}.bin')

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
    return evaluation(batch[-1].cpu().numpy(), y_probs, labels=list(range(self.global_config.num_labels)))

  def get_mean_score(self, scores):
    keys = scores[0].keys()
    return {key:np.mean([score[key] for score in scores]) for key in keys}

  def _save_weights(self, half_precision=False, path='models/'):
    if os.path.isdir(path): path = os.path.join(path, f'model_{self.fold}.bin')

    print('Saving weights ...', end='')
    if half_precision: self.module.half() #for fast inference
    torch.save(self.module.state_dict(), path)
    gc.collect()
    print('done')

  def load(self, path='models/'):
    if os.path.isdir(path): path = os.path.join(path, f'model_{self.fold}.bin')

    print('Loading weights ... ', end='')
    self.module.load_state_dict(torch.load(path))
    gc.collect()
    print('done')

  def _check_evaluation_score(self, metric, best_eval=None):
    if metric > self.best_metric:
      self.best_metric = metric
      self.best_eval = best_eval
      self._save_weights()

class TrainerForSeqClassification(Trainer):
  _module_class = SeqClassificationModule

  def save_best_eval(self, path='evals/{}/fold_{}_best_eval.npy'):
    if self.global_config.phase=='train':
      np.save(path.format(self.global_config.model_name, self.global_config.fold), np.vstack(self.best_eval))

class TrainerForSeq2Seq(Trainer):
  _module_class = Seq2SeqModule

  def save_best_eval(self, path='evals/{}/fold_{}_best_eval.txt'):
    if self.global_config.phase=='train':
      file = path.format(self.global_config.model_name, self.global_config.fold)
      with open(file, 'w') as f:
        for batch in self.best_eval:
          for s in batch:
            f.write(s+'\n')

  def get_score(self, batch, decoded):
    _, raw_texts = batch
    return {
      'bleu': calculate_bleu(raw_texts['trg'], decoded)
    }

  def evaluate_generation(self):
    score = []

    self.load()

    with torch.no_grad():
      for i, batch in enumerate(tqdm(self.val_dl, desc='Eval')):
        _, decoded = self.module.test_step(batch[0], i)
        score += [ self.get_score(batch, decoded) ]

        self.printer.pprint(**score[-1])
      score = self.get_mean_score(score)
    
    return score