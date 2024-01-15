import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import MNIST

class MNISTDataMudle(Dataset):
    def __init__(self, data_dir: str='../mnist/', batch_size=128, num_workers=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        transform = T.Compose([
            T.Resize((32, 32)), #DenseNet121: 32*32
            T.ToTensor(),
            T.Normalize(0.1307, 0.3081)
        ])
        self.ds_train = MNIST('../mnist', train=True, transform=transform, download=False)
        self.ds_test = MNIST('../mnist', train=False, transform=transform, download=False)
    
    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=True, pin_memory=True)

data_mnist = MNISTDataMudle()
data_mnist.setup()

for features, labels in data_mnist.train_dataloader():
    print(features.shape)
    print(labels.shape)
    break

for features, labels in data_mnist.test_dataloader():
    print(features.shape)
    print(labels.shape)
    break

# GoogLeNet中的Inception
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1) #[batch, channel, length, width]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5) #24+24+24+16=88

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)
        
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10) #88*4*4=1408
    
    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        #print(x.shape)->[128, 1408]
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # 用于统一x和y的通道数，最好不要复用conv2，参数单独训练
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activate = torch.nn.ReLU()

    def forward(self, x):
        y = self.activate(self.conv1(x))
        y = self.conv2(y)
        x = self.conv(x)
        return self.activate(x+y)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.mp = nn.MaxPool2d(2)

        self.rblock1 = ResidualBlock(16, 16)
        self.rblock2 = ResidualBlock(32, 32)
        
        self.fc = nn.Linear(512, 10) #32*4*4
    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)

        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x

#model = Net()
from models.densenet import DenseNet121
model = DenseNet121(num_classes=10, grayscale=True)
print(model)

#构建损失器
criterion = nn.CrossEntropyLoss()
 
#构建优化器
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)

#训练网络
def train(epoch):
    runing_loss = 0
 
    for batchix, datas in enumerate(data_mnist.train_dataloader(),0):
        inputs, label = datas
 
        optimizer.zero_grad()
        output, _ = model(inputs)
        '''
        #DenseNet output
        def forward(self, x):
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            logits = self.classifier(out)
            probas = F.softmax(logits, dim=1)
            return logits, probas
        '''
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
 
        runing_loss +=loss.item()
 
        if batchix%300 == 299:
            print('[%d %3d] %.3f'%(epoch+1,batchix+1, runing_loss/300))
            runing_loss = 0
 
#测试网络
def test():
    total = 0
    correct = 0
    with torch.no_grad():
        for data in data_mnist.test_dataloader():
            inputs,label = data
            outputs, _ = model(inputs)
            _,pre = torch.max(outputs, dim=1)
            total += label.size(0)
            correct += (pre==label).sum().item()
    print('准确率为%.3f%%' % (correct / total * 100))
 
if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()