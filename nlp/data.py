import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from keras.utils import to_categorical
from .utils import getTokenizer

class BaseDataset(Dataset):
  def __init__(self, df, task='train', loss_name='ce', c=3):
    super(BaseDataset, self).__init__()

    self.text_col = 'text'
    self.target_col = 'label'
    self.length_col = 'length'
    self.c = c
    self.loss_name = loss_name

    self.task = task
    self.df = df.reset_index(drop=True)
    
  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self, idx):
    text = self.df.loc[idx, self.text_col]
    length = self.df.loc[idx, self.length_col]
    y = self.df.loc[idx, self.target_col] if self.task=='train' else 0

    if self.loss_name == 'bce':
        y = to_categorical(y, self.c)

    return [text, length, y]
    

class FastTokCollateFn:
    def __init__(self, model_config, tok_name, max_tokens=100, on_batch=False):
        self.tokenizer = getTokenizer(model_config, tok_name)
        self.max_tokens = max_tokens
        self.on_batch = on_batch

    def __call__(self, batch):
        batch = np.array(batch)

        labels = torch.tensor(self._map_to_int(batch[:,-1]))
        max_pad = self.max_tokens

        if self.on_batch:
            max_pad = min(max(self._map_to_int(batch[:,1])), max_pad)
        
        encoded = self.tokenizer(
            batch[:,0].tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_pad,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return encoded, labels

    def _map_to_int(self, x):
        return list(map(int, x))