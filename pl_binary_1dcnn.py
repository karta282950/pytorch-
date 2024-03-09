'''
[v1]
* table classification
Ref1: https://github.com/JiaBinBin233/CNN1D.git
Ref2: https://lyb592.blog.csdn.net/article/details/133556204?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7EPayColumn-1-133556204-blog-125709932.235%5Ev40%5Epc_relevant_3m_sort_dl_base4&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7EPayColumn-1-133556204-blog-125709932.235%5Ev40%5Epc_relevant_3m_sort_dl_base4&utm_relevant_index=1&ydreferer=aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNTI5MTUxL2FydGljbGUvZGV0YWlscy8xMjU3MDk5MzI%3D
'''

from torch import nn
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import Accuracy
import pytorch_lightning as pl
from pytorch_lightning.loggers.csv_logs import CSVLogger
import pandas as pd

batch_szie = 128
lr = 1e-3
num_classes = 2
num_workers = 4

train = pd.read_csv('')
test = pd.read_csv('')

class SmartDataset(Dataset):
  def __init__(self, data: pd.core.frame.DataFrame, is_train: bool=True):
    self.inputs = data.drop(['failure'], axis=1).values
    self.labels = data['failure'].values

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, idx):
    target = self.labels[idx]
    data = self.inputs[idx]
    return torch.tensor(data, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

class SmartDataModule(pl.LightningDataModule):
  def __init__(self, batch_szie=128, num_workers=4):
    super().__init__()
    self.batch_szie = batch_szie
    self.num_workers = num_workers
  
  #def prepare_data(self):
  #  self.train_df = self.setup(pd.read_csv('/content/CNN1D/train.csv'))
  #  self.test_df = self.setup(pd.read_csv('/content/CNN1D/test.csv'))

  def setup(self, stage=None):
    self.full_df = SmartDataset(pd.read_csv('train.csv'))
    self.train_df, self.val_df = random_split(self.full_df, [0.7, 0.3])
    self.test_df = SmartDataset(pd.read_csv('test.csv'))

    #encoder = LabelEncoder()
    #Y_train = encoder.fit_transform(Y_train.ravel())

  def train_dataloader(self):
    return DataLoader(self.train_df, batch_size=self.batch_szie, shuffle=True, pin_memory=True, num_workers=self.num_workers)
  
  def val_dataloader(self):
    return DataLoader(self.val_df, batch_size=self.batch_szie, shuffle=True, pin_memory=True, num_workers=self.num_workers)

  def test_dataloader(self):
    return DataLoader(self.test_df, batch_size=self.batch_szie, shuffle=False, pin_memory=True, num_workers=self.num_workers)
  
class CNN1D(pl.LightningModule):
  def __init__(self, lr=1e-3, num_epochs=100):
    super(CNN1D, self).__init__()
    self.lr = lr
    self.num_epochs = num_epochs
    self.acc = Accuracy(task='binary', num_classes=2)
    self.net = nn.Sequential(
        nn.Conv1d(1, 16, 2),
        nn.Sigmoid(),
        nn.MaxPool1d(2),
        nn.Conv1d(16, 32, 2),
        nn.Sigmoid(),
        nn.MaxPool1d(4),
        nn.Flatten(),
        nn.Linear(32, 2, bias=True),
        nn.Sigmoid()
    )
  
  def forward(self, x):
    x = x.reshape(-1,1,11)
    x = self.net(x)
    return x
  
  def training_step(self, batch, batch_idx):
    x, y = batch
    preds = self(x)
    loss = F.cross_entropy(preds, y)
    preds = preds.argmax(axis=1)
    acc = self.acc(preds, y)
    self.log('train_loss', torch.tensor([loss]), prog_bar=True)
    self.log('train_acc', torch.tensor([acc]), prog_bar=True)
    return {"loss":loss,"preds":preds.detach(),"y":y.detach()}

  def validation_step(self, batch, batch_idx):
    x, y = batch
    preds = self(x)
    loss = F.cross_entropy(preds, y)
    preds = preds.argmax(axis=1)
    acc = self.acc(preds, y)
    self.log('val_loss', torch.tensor([loss]), prog_bar=True)
    self.log('val_acc', torch.tensor([acc]), prog_bar=True)
    return {"loss":loss, "preds":preds.detach(), "y":y.detach()}

  def test_step(self, batch, batch_idx):
    x, y = batch
    preds = self(x)
    loss = F.cross_entropy(preds, y)
    preds = preds.argmax(axis=1)
    acc = self.acc(preds, y)
    self.log('test_loss', torch.tensor([loss]), prog_bar=True)
    self.log('test_acc', torch.tensor([acc]), prog_bar=True)
    return {"loss":loss,"preds":preds.detach(),"y":y.detach()}
  
  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=self.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                    T_max=self.num_epochs,
                                                    eta_min=self.lr/1e2,
                                                    )
    return [optimizer], [scheduler]

pl.seed_everything(42)
ckpt_callback = pl.callbacks.ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max')
pbar = pl.callbacks.TQDMProgressBar(refresh_rate=1)
early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
csv_logger = CSVLogger('./', name='linear', version='0'),

data_smart = SmartDataModule()
data_smart.setup()
model = CNN1D()
trainer = pl.Trainer(max_epochs=100,
                  callbacks=[ckpt_callback, pbar, early_stopping],
                  logger=csv_logger,
                  #logger=pl_logger,
                  enable_model_summary=True,
                  accelerator='auto',
                  devices=1,
                  num_sanity_val_steps=1,
                  benchmark=True,
                  fast_dev_run=False)

trainer.fit(model, data_smart)
trainer.test(model, data_smart)