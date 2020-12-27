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
  def __init__(self, df, phase='train', aug=None, c=3):
    super(BaseDataset, self).__init__()

    self.length_col = 'length'
    self.aug = aug
    self.phase = phase
    self.c = c
    self.df = df.reset_index(drop=True)
    
  def __len__(self):
    return self.df.shape[0]

class SeqClassificationDataset(BaseDataset):
  def __init__(self, df, phase='train', aug=None, c=3):
    super(SeqClassificationDataset, self).__init__(df, phase, aug, c)

    self.text_col = 'text'
    self.target_col = 'label'

  def __getitem__(self, idx):
    text = self.df.loc[idx, self.text_col]
    length = self.df.loc[idx, self.length_col]
    y = self.df.loc[idx, self.target_col] if self.phase!='test' else 0

    if self.aug:
        text = self.aug([text])

    return [text, length, y]

class Seq2SeqDataset(BaseDataset):
    def __init__(self, df, phase='train', aug=None, c=-1):
        super(Seq2SeqDataset, self).__init__(df, phase, aug, c)

        self.src_text = 'src_text'
        self.trg_text = 'trg_text'
        self.src_lang = 'src_lang'
        self.trg_lang = 'trg_lang'

        self.aug = aug

    def __getitem__(self, idx):
        row = self.df.loc[idx, [self.src_text, self.trg_text, self.src_lang, self.trg_lang]]
        src_text, trg_text, src_lang, trg_lang = row.to_numpy().tolist()

        if self.aug:
            src_text, trg_text = self.aug([src_text, trg_text])

        return [src_text, trg_text, src_lang, trg_lang]

#########################   TokCollate

class BaseFastCollator:
    def __init__(self, model_config, tok_name, max_tokens=100, on_batch=False, phase='train'):
        self.tokenizer = getTokenizer(model_config, tok_name)
        self.max_tokens = max_tokens
        self.on_batch = on_batch
        self.phase = phase

    def __call__(self, batch):
        return np.array(batch)

    def _map_to_int(self, x):
        return list(map(int, x))

    def _encode(self, texts, max_pad=None, padding=True, return_attention_mask=True):
        return self.tokenizer(
            texts,
            truncation=max_pad is not None,
            padding=padding,
            max_length=max_pad,
            return_attention_mask=return_attention_mask,
            return_tensors='pt'
        )


class SeqClassificationCollator(BaseFastCollator):
    def __call__(self, batch):
        batch = super().__call__(batch)

        max_pad = self.max_tokens

        if self.on_batch:
            max_pad = min(max(self._map_to_int(batch[:,1])), max_pad)
        
        encoded = self._encode(batch[:,0].tolist(), max_pad)
        encode.update({'labels': torch.tensor(self._map_to_int(batch[:,-1]))})
        
        return encoded


class Seq2SeqCollator(BaseFastCollator):
    def __call__(self, batch):
        batch = super().__call__(batch)

        src = batch[:,0].tolist()

        if self.phase != 'test':

            trg = batch[:,1].tolist()
            raw_texts = {'src': src, 'trg': trg}

            if hasattr(self.tokenizer, "prepare_seq2seq_batch"):
                batch =  self.encode(src, trg)
            else:
                batch = self._encode(src, max_pad=self.max_tokens)
                trg_encoded = self._encode(trg, max_pad=max_tokens)

                batch.update({"labels": trg_encoded['input_ids']})
            
            return batch, raw_texts
        else:
            return self._encode(src, max_pad=None, truncation=False)

    def encode(self, src_texts, trg_texts):
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            src_texts,
            tgt_texts=trg_texts,
            padding=True,
            truncation= self.max_tokens is not None,
            max_length = self.max_tokens,
            return_tensors="pt"
        )

        return batch_encoding.data