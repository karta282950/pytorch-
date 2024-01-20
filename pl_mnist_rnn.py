from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import Accuracy
from torchvision.datasets import MNIST
import torchvision.transforms as T 

import pytorch_lightning as pl 
from pytorch_lightning.loggers.csv_logs import CSVLogger

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#batch_size = 100
#num_workers = 1
input_size = 28
seq_len = 28
num_classes = 10
hidden_size = 128
num_layers = 2

class MNISTDataMudle(pl.LightningDataModule):
    def __init__(self, data_dir: str='./data/mnist', batch_size: int=100, num_workers: int=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(0.1307, 0.3081)
        ])
        self.full_df = MNIST(self.data_dir, train=True, transform=transform, download=False)
        self.train_df, self.val_df = random_split(self.full_df, [0.7, 0.3])
        self.test_df = MNIST(self.data_dir, train=False, transform=transform, download=False)

    
    def train_dataloader(self):
        return DataLoader(self.train_df, batch_size=self.batch_size, num_workers=0, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_df, batch_size=self.batch_size, num_workers=0, shuffle=True, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_df, batch_size=self.batch_size, num_workers=0, shuffle=True, pin_memory=True)

data_mnist = MNISTDataMudle()
data_mnist.setup()

for inputs, labels in data_mnist.train_dataloader():
    print(inputs.shape)
    print(labels.shape)
    break

for inputs, labels in data_mnist.val_dataloader():
    print(inputs.shape)
    print(labels.shape)
    break

class Model(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, learning_rate=1e-3):
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.train_acc = Accuracy(task='multiclass', num_classes=10)
        self.val_acc = Accuracy(task='multiclass', num_classes=10)
        self.test_acc = Accuracy(task='multiclass', num_classes=10)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # x -> (batch_size, seq, input_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        # out: batch_size, seq_lenght, hidden_size
        # out: (N, 28, 128)
        out = out[:,-1,:]
        out = self.fc(out)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(-1, seq_len, input_size).to(device)
        y = y.to(device)
        preds = self(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        acc = self.train_acc(preds, y)
        self.log('train_loss', torch.tensor([loss]))
        self.log('train_acc', torch.tensor([acc]))
        return {"loss":loss,"preds":preds.detach(),"y":y.detach()}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(-1, seq_len, input_size).to(device)
        y = y.to(device)
        preds = self(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        acc = self.val_acc(preds, y)
        self.log('val_loss', torch.tensor([loss]))
        self.log('val_acc', torch.tensor([acc]))
        return {"loss":loss,"preds":preds.detach(),"y":y.detach()}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(-1, seq_len, input_size).to(device)
        y = y.to(device)
        preds = self(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        acc = self.test_acc(preds, y)
        self.log('test_loss', torch.tensor([loss]))
        self.log('test_acc', torch.tensor([acc]))
        return {"loss":loss,"preds":preds.detach(),"y":y.detach()}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [scheduler]
    
model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
summary = pl.utilities.model_summary.ModelSummary(model, max_depth=-1)
print(summary)

pl.seed_everything(42)

ckpt_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')

csv_logger = CSVLogger('./', name='rnn', version='0'),

trainer = pl.Trainer(max_epochs=20,
    accelerator="auto",
    callbacks = [ckpt_callback, early_stopping],
    profiler="simple",
    logger=csv_logger)

#训练模型
trainer.fit(model, data_mnist)
trainer.test(model, datamodule=data_mnist)

#驗證測試集
#result = trainer.test(model, data_mnist.test_dataloader(), ckpt_path='lightning_logs/version_2/checkpoints/epoch=19-step=26260.ckpt')
#print(result)

#
metrics = pd.read_csv('./rnn/0/metrics.csv')
train_loss = metrics[['train_loss', 'step', 'epoch']][~np.isnan(metrics['train_loss'])]
val_loss = metrics[['val_loss', 'epoch']][~np.isnan(metrics['val_loss'])]
test_loss = metrics['test_loss'].iloc[-1]

fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=100)
axes[0].set_title('Train loss per batch')
axes[0].plot(train_loss['step'], train_loss['train_loss'])
axes[1].set_title('Validation loss per epoch')
axes[1].plot(val_loss['epoch'], val_loss['val_loss'], color='orange')
plt.show(block = True)

print('Loss:')
print(f"Train loss: {train_loss['train_loss'].iloc[-1]:.3f}")
print(f"Val loss:   {val_loss['val_loss'].iloc[-1]:.3f}")
print(f'Test loss:  {test_loss:.3f}')