import torch
from torch.nn import Sequential, Linear, MSELoss
from torch_pso import ParticleSwarmOptimizer
from torch import nn
from torch.utils.data import Dataset

batch_size = 128
lr = 1e-3

class BostonDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        target = self.labels[idx]
        data = self.inputs[idx]
        return torch.tensor(data), torch.tensor(target)

class SimpleConvNet(nn.Module):
  '''
    Simple Convolutional Neural Network
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(1, 10, kernel_size=3),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(26 * 26 * 10, 50),
      nn.ReLU(),
      nn.Linear(50, 20),
      nn.ReLU(),
      nn.Linear(20, 10)
    )

  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)

#net = Sequential(Linear(10,100), Linear(100,100), Linear(100,10))
optim = ParticleSwarmOptimizer(net.parameters(),
                               inertial_weight=0.5,
                               num_particles=100,
                               max_param_value=1,
                               min_param_value=-1)
criterion = MSELoss()
target = torch.rand((10,)).round()

x = torch.rand((10,))
for _ in range(100):
    
    def closure():
        # Clear any grads from before the optimization step, since we will be changing the parameters
        optim.zero_grad()  
        return criterion(net(x), target)
    
    optim.step(closure)
    print('Prediciton', net(x))
    print('Target    ', target)