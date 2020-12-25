import os, gc, random
from time import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, f1_score, accuracy_score
from collections import Counter
from IPython.display import clear_output
import torch
from transformers import (
    AutoTokenizer, RobertaTokenizerFast, 
    BertTokenizerFast, ElectraTokenizerFast
)

def seed_everything(seed):
  print(f'Set seed to {seed}.')
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available(): 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_blackbone(n):
  return n.startswith('model')
  
def evaluation(ytrue, y_pred, labels=[0,1,2,3]):
  log = log_loss(ytrue, y_pred, labels=labels)
  f1 = f1_score(ytrue, y_pred.argmax(1), average='weighted')
  acc = accuracy_score(ytrue, y_pred.argmax(1))

  return {'Logloss': log, 'F1': f1, 'Acc': acc}

def getTokenizer(model_config, tok_name):
  return AutoTokenizer.from_pretrained(tok_name, config=model_config, add_prefix_space=False)

class EarlyStopping:
  def __init__(self, patience=5, mode='max'):
    self.step = 0
    self.stop = False
    self.score = 0
    self.patience = patience
    self.mode = mode
    self.mult = 1 if mode=='max' else -1

  def update(self, score):
    if self.mult*(self.score-score) > 0:
      self.step += 1
    else: 
      self.step = 0
      self.score = score
    
    if self.step == self.patience: 
      self.stop = True

class Timer:
  def __init__(self):
    self._time = 0
    self.is_stopped = False
    self._start()

  def _start(self):
    self._time = time()

  def _stop(self):
    if not self.is_stopped:
      self.is_stopped = True
      self._time = time()-self._time

  @property
  def time(self):
    self._stop()
    return self._time

  def to_string(self):
    return "{:02d}:{:02d}".format(*self.m_s())

  def m_s(self):
    t = round(self.time)
    s = t%60
    m = t//60

    return m,s


class Printer:
  def __init__(self, fold=0):
    self._print = []
    self.fold = fold

  def pprint(self, **kwargs):
    str_log = "\r"
    for key in kwargs.keys():
      str_log += "{}: {} - ".format(key, kwargs[key])
  
    print(str_log, end='')

  def update(self, epoch, losses, scores, time = None):
    str_log = f"⏰ {time} | " if time else ""
    str_log += "Epoch: {} - Loss: {:.5f} - ValLoss: {:.5f}".format(epoch, losses['loss'][epoch], losses['val_loss'][epoch])
    for metric_name, value in scores.items():
      str_log += ' - {}: {:.5f}'.format(metric_name, value)

    self._print.append(str_log)

  def show(self):
    clear_output()

    print("_"*100, "\nFold ", self.fold)
    for p in self._print:
      print("_" * 100)
      print('| '+ p)

  def update_and_show(self, epoch, losses, score, time=None):
    self.update(epoch, losses, score, time)
    self.show()


class WorkplaceManager:
  def __init__(self, seed, dirs, exts, n_fols=10):
    self.seed = seed
    self.dirs = dirs
    self.exts = exts
    self.n_folds = n_fols

    self._set_workplace()

  @staticmethod
  def create_dir(dir):
    os.makedirs(dir, exist_ok=True)
  
  def _create_dirs(self):
    print('Created {}'.format(' '.join(self.dirs)))
    for d in self.dirs:
      self.create_dir(d)
  
  def _clear_dirs(self):
    print('Deleted {}'.format(' '.join(self.dirs)))
    self.clear([f'{d}*' for d in self.dirs])

  def _clear_files(self):
    print('Deleted {}'.format(' '.join(self.exts)))
    self.clear([f'*{ext}' for ext in self.exts])

  def clear(self, objs_name):
    os.system('rm -r {}'.format(' '.join(objs_name)))

  def _set_workplace(self):
    seed_everything(self.seed)
    if os.path.exists('models') and len(os.listdir('models/')) == self.n_folds:
      self._clear_dirs()
      self._clear_files()    
    self._create_dirs()


class CrossValLogger:
  def __init__(self, df, metric_name, n_folds=10, oof_cv = 'cv_score.pkl', path='evals/roberta-base/'):
    assert df.fold.nunique()==n_folds, "Unconsistency between df.n_folds and n_folds"

    self.df = df.copy()
    self.metric_name = metric_name
    self.path = path
    self.n_folds = n_folds
    self.oof_cv = oof_cv
    self.score1, self.score2 = None, None

  def _retrieve_eval_preds(self):
    ph = self.path+'fold_{}_best_eval.npy'
    shape = ( self.df.shape[0], self.df.label.nunique() )
    preds = np.empty(shape, dtype=np.float32)
    for i in self.df.fold.unique():
      index = self.df[self.df.fold==i].index.values
      fold_pred = np.load(ph.format(i))
      preds[index] = fold_pred[:, :]
    return preds

  def _load_oof_cv_score(self):
    score = 0
    with open(self.oof_cv, 'rb') as f:
      score = pickle.load(f)
      f.close()
    return score

  def show_results(self, return_score=False):
    if self.score1 is None:
      eval_preds = self._retrieve_eval_preds()
      self.score1 = self._load_oof_cv_score() / self.n_folds #oof_cv_scores
      self.score2 = evaluation(self.df.label.values, eval_preds, labels=self.df.label.unique())[self.metric_name] #ovr_score

    print('OOF_CV_SCORE: {:.5f} | OVR_SCORE: {:.5f}'.format(self.score1, self.score2))
    
    if return_score: return self.score1, self.score2
