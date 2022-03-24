import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import *
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# data
path2data = 'cifar10/'
if not os.path.exists(path2data):
    os.mkdir(path2data)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224)
                                ])

train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# make dataloader
trainloader= DataLoader(train_ds, batch_size=32, shuffle=True)
testloader = DataLoader(val_ds, batch_size=32, shuffle=True)




 # train
CEE = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mean_loss = []

for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    inputs, labels = inputs.cuda(), labels.cuda()
    inputs, labels = inputs.to(device), labels.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    loss = CEE(outputs, labels)
    print(f'{i} loss is {loss.item()}')
    mean_loss.append(loss.item())
    loss.backward()
    optimizer.step()
        

print(f'mean loss was {sum(mean_loss) / len(mean_loss)}')