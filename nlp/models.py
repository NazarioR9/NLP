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
from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup

from .activation import Mish
from .utils import evaluation, is_blackbone, Printer, WorkplaceManager, Timer
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

    if hasattr(self.global_config, 'multi_drop_nb'):
      x = torch.mean(
          torch.stack(
              [self.classifier(self.high_dropout(x)) for _ in range(self.global_config.multi_drop_nb)],
              dim=0,
          ),
          dim=0,
      )
    else:
      x = self.classifier(self.high_dropout(x))

    return x