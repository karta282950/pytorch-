import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from torchmetrics import Accuracy
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from omegaconf import DictConfig
import hydra

from models.model1 import LinearModel, CNNBaseModel, GoogLeNet, ResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_dir = self.cfg.data_dir
        self.batch_size = self.cfg.batch_size
        self.num_workers = self.cfg.num_workers

    def setup(self, stage=None):
        transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
        self.full_df = CIFAR10(root=self.data_dir, train=True, transform=transform, download=False)
        self.trn_df, self.val_df = random_split(self.full_df, [0.7, 0.3])
        self.test_df = CIFAR10(root=self.data_dir, train=False, transform=transform, download=False)
    
    def train_dataloader(self):
        return DataLoader(self.trn_df, batch_size=self.batch_size, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_df, batch_size=self.batch_size, shuffle=True, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_df, batch_size=self.batch_size, shuffle=True, pin_memory=True)

class Model(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(Model, self).__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.num_epochs = self.cfg.num_epochs
        self.learning_rate = self.cfg.learning_rate
        self.acc = Accuracy(task='multiclass', num_classes=10)
        self.net = LinearModel()
    def forward(self, x):
        x = self.net(x)
        return x 
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        acc = self.acc(preds, y)
        self.log('train_loss', torch.tensor([loss]), prog_bar=True)
        self.log('train_acc', torch.tensor([acc]), prog_bar=True)
        return {"loss":loss,"preds":preds.detach(),"y":y.detach()}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        acc = self.acc(preds, y)
        self.log('val_loss', torch.tensor([loss]), prog_bar=True)
        self.log('val_acc', torch.tensor([acc]), prog_bar=True)
        return {"loss":loss,"preds":preds.detach(),"y":y.detach()}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        acc = self.acc(preds, y)
        self.log('test_loss', torch.tensor([loss]), prog_bar=True)
        self.log('test_acc', torch.tensor([acc]), prog_bar=True)
        return {"loss":loss,"preds":preds.detach(),"y":y.detach()}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                        T_max=self.num_epochs,
                                                        eta_min=self.learning_rate/1e2)
        return [optimizer], [scheduler]

@hydra.main(config_path="./", config_name="cifar10", version_base="1.1")
def main(cfg: DictConfig):
    data_cifar10 = CIFAR10DataModule(cfg)
    data_cifar10.setup()

    model = Model(cfg)
    summary = pl.utilities.model_summary.ModelSummary(model, max_depth=-1)
    print(summary)

    pl.seed_everything(42)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss', save_top_k=1, mode='min')
    early_stopping = pl.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, mode='min')

    trainer = pl.Trainer(max_epochs=5,
        accelerator="auto",
        callbacks = [ckpt_callback, early_stopping],
        profiler="pytorch",
        )

    #训练模型
    #trainer.fit(model, data_cifar10)
    #trainer.test(model, datamodule=data_cifar10)
    model = model.load_from_checkpoint(
        ckpt_callback.best_model_path,
    )
    model.eval()
    idx = 0
    for inputs, lables in data_cifar10.test_dataloader():
        if idx%100==0:
            outputs = model(inputs)
            outputs = torch.argmax(outputs, 1)
            acc = torch.eq(outputs, lables)
            print('Accuracy: {:.2%}'.format((torch.sum(acc) / acc.shape[0]).item()))
        idx+=1
if __name__== '__main__':
    main()