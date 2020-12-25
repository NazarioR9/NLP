import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from keras.utils import to_categorical
from .utils import getTokenizer
from .augment import *

class BaseDataset(Dataset):
  def __init__(self, df, task='train', aug=None):
    super(BaseDataset, self).__init__()

    self.length_col = 'length'
    self.aug = aug
    self.task = task
    self.df = df.reset_index(drop=True)
    
  def __len__(self):
    return self.df.shape[0]

class SeqClassificationDatatset(BaseDataset):
  def __init__(self, df, task='train', aug=None, c=3):
    super(SeqClassificationDatatset, self).__init__(df, task, aug)

    self.text_col = 'text'
    self.target_col = 'label'
    self.c = c

  def __getitem__(self, idx):
    text = self.df.loc[idx, self.text_col]
    length = self.df.loc[idx, self.length_col]
    y = self.df.loc[idx, self.target_col] if self.task!='test' else 0

    if self.aug:
        text = aug([text])

    return [text, length, y]

class Seq2SeqDataset(BaseDataset):
    def __init__(self, df, task='train', aug=None):
        super(Seq2SeqDataset, self).__init__(df, task, aug)

        self.src_text = 'src_text'
        self.trg_text = 'trg_text'
        self.src_lang = 'src_lang'
        self.trg_lang = 'trg_lang'

        self.aug = aug

    def __getitem__(self, idx):
        row = self.df.loc[idx, [self.src_text, self.trg_text, self.src_lang, self.trg_lang, self.length_col]]
        src_text, trg_text, src_lang, trg_lang, length = row.to_numpy().tolist()

        if aug:
            src_text, trg_text = aug([src_text, trg_text])

        return [src_text, trg_text, src_lang, trg_lang, length]

#########################   TokCollate

class BaseFastCollator:
    def __init__(self, model_config, tok_name, max_tokens=100, on_batch=False):
        self.tokenizer = getTokenizer(model_config, tok_name)
        self.max_tokens = max_tokens
        self.on_batch = on_batch

    def __call__(self, batch):
        return np.array(batch)

    def _map_to_int(self, x):
        return list(map(int, x))

    def encode(self, texts, max_pad):
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_pad,
            return_attention_mask=True,
            return_tensors='pt'
        )


class SeqClassificationCollator(BaseFastCollator):
    def __init__(self, model_config, tok_name, max_tokens=100, on_batch=False):
        super(SeqClassificationCollator, self).__init__(model_config, tok_name, max_tokens, on_batch)

    def __call__(self, batch):
        batch = super().__call__(batch)

        labels = torch.tensor(self._map_to_int(batch[:,-1]))
        max_pad = self.max_tokens

        if self.on_batch:
            max_pad = min(max(self._map_to_int(batch[:,1])), max_pad)
        
        encoded = self.encode(batch[:,0].tolist(), max_pad)
        
        return encoded, labels

class Seq2SeqCollator(BaseFastCollator):
    def __init__(self, model_config, tok_name, max_tokens=100, on_batch=False):
        super(Seq2SeqCollator, self).__init__(model_config, tok_name, max_tokens, on_batch)

    def __call__(self, batch):
        batch = super().__call__(batch)

        max_pad = self.max_tokens

        if self.on_batch:
            max_pad = min(max(self._map_to_int(batch[:,-1])), max_pad)
        
        src_encoded = self.encode(batch[:,0].tolist(), max_pad)
        trg_encoded = self.encode(batch[:,1].tolist(), max_pad)
        
        return src_encoded, trg_encoded