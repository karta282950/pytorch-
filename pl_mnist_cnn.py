#from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
#from typing import Any
#from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from torchmetrics import Accuracy
import pytorch_lightning as pl
from torchvision import transforms as T 
from torchvision.datasets import MNIST
from pytorch_lightning.loggers import WandbLogger

pl_logger = WandbLogger(name='exp2', project="pl_mnist")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MNISTDataMudle(pl.LightningDataModule):
    def __init__(self, data_dir: str='./data/mnist', batch_size: int=32, num_workers: int=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        transform = T.Compose([T.ToTensor()])
        self.ds_full = MNIST(self.data_dir, train=True, transform=transform, download=False)
        self.ds_test = MNIST(self.data_dir, train=False, transform=transform, download=False)
        self.ds_train, self.ds_val = random_split(self.ds_full, [0.7, 0.3])
    
    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=True, pin_memory=True)
    

data_mnist = MNISTDataMudle()
data_mnist.setup()

for features, labels in data_mnist.train_dataloader():
    print(features.shape)
    print(labels.shape)
    break

net = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout2d(p=0.1),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

class Model(pl.LightningModule):
    def __init__(self, net, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = net
        self.train_acc = Accuracy(task='multiclass', num_classes=10)
        self.val_acc = Accuracy(task='multiclass', num_classes=10)
        self.test_acc = Accuracy(task='multiclass', num_classes=10)

    def forward(self, x):
        x = self.net(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        acc = self.val_acc(preds, y)
        self.log('train_loss', torch.tensor([loss]))
        self.log('train_acc', torch.tensor([acc]))
        return {"loss":loss,"preds":preds.detach(),"y":y.detach()}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        acc = self.test_acc(preds, y)
        self.log('val_loss', torch.tensor([loss]))
        self.log('val_acc', torch.tensor([acc]))
        return {"loss":loss,"preds":preds.detach(),"y":y.detach()}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
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
    
model = Model(net)
summary = pl.utilities.model_summary.ModelSummary(model, max_depth=-1)
print(summary)

pl.seed_everything(42)

ckpt_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')

trainer = pl.Trainer(max_epochs=20,
    accelerator="auto",
    #gpus=0, #单GPU模式
    #num_processes=4, strategy="ddp_find_unused_parameters_false", #多CPU(进程)模式
    #gpus=[0,1,2,3],strategy="dp", #多GPU的DataParallel(速度提升效果一般)
    #gpus=[0,1,2,3],strategy=“ddp_find_unused_parameters_false" #多GPU的DistributedDataParallel(速度提升效果好)
    callbacks = [ckpt_callback, early_stopping],
    logger=pl_logger,
    profiler="simple")

#断点续训
#trainer = pl.Trainer(resume_from_checkpoint='./lightning_logs/version_31/checkpoints/epoch=02-val_loss=0.05.ckpt')

#训练模型
trainer.fit(model, data_mnist)
trainer.test(model, datamodule=data_mnist)

#result = trainer.test(model, data_mnist.test_dataloader(), ckpt_path='lightning_logs/version_2/checkpoints/epoch=19-step=26260.ckpt')
#print(result)