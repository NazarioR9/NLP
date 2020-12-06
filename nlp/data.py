import torch
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from keras.utils import to_categorical
from .utils import getTokenizer

class BaseDataset(Dataset):
  def __init__(self, df, task='train', loss_name='ce'):
    super(BaseDataset, self).__init__()

    self.text_col = 'text'
    self.target_col = 'label'
    self.length_col = 'length'
    self.c = 3

    self.task = task
    self.df = df.reset_index(drop=True)
    
  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self, idx):
    text = self.df.loc[idx, self.text_col]
    length = self.df.loc[idx, self.length_col]
    y = self.df.loc[idx, self.target_col] if self.task=='train' else -1

    if self.loss_name == 'bce':
        y = to_categorical(y, self.c)

    return text, length, y
    

class FastTokCollateFn:
    def __init__(self, model_config_name, max_tokens=100, on_batch=False):
        self.tokenizer = getTokenizer(model_config_name)
        self.max_tokens = max_tokens
        self.on_batch = on_batch

    def __call__(self, batch):
        texts = [x[0] for x in batch]
        labels = torch.tensor([x[-1] for x in batch])
        max_pad = self.max_tokens

        if self.on_batch:
            lengths = [x[1] for x in batch]
            max_pad = min(max(lengths), max_pad)
        
        encoded = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=max_pad, 
            return_special_tokens_mask=True, 
            return_attention_mask=True
        )

        outputs = {}
        
        for key, val in encoded.items():
            outputs[key] = torch.tensor(val)
        
        return outputs, labels