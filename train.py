import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('cifar10')

path2data = 'cifar10/'
if not os.path.exists(path2data):
    os.mkdir(path2data)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(224)])

train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# make dataloader
trainloader= DataLoader(train_ds, batch_size=32, shuffle=True)
testloader = DataLoader(val_ds, batch_size=32, shuffle=True)

print(len(train_ds))
print(len(val_ds))