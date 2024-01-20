import torch
from torch import nn 
from einops import rearrange

class LinearModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 10)
        )
    
    def forward(self, x):
        x = rearrange(x, 'b 1 x y -> b (x y)', x=28, y=28)
        x = self.net(x)
        return x

if __name__=='__main__':
    model = LinearModel(hidden_dim=100)
    inputs = torch.ones((64, 1, 28, 28))
    outputs = model(inputs)
    print(outputs)