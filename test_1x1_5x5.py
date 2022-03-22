from ast import Pass
import torch
import torch.nn as nn
import torchvision
import numpy 
from torchsummary import summary



class solo(nn.Module):
    def __init__(self):
        super(solo, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, padding='same',)
    
    def forward(self, x):
        x = self.conv1(x)
        return x
    
class double(nn.Module):
    def __init__(self):
        super(double, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, padding='same',)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding='same',)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x    

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn((1, 128, 28, 28)).to(device)
model = solo().to(device)
output = model(x)
print('output size:', output.size())


summary(model, (128, 28, 28), device=device.type)
    
        