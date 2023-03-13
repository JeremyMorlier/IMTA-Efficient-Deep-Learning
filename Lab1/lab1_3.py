from torchvision.datasets import CIFAR100
import numpy as np 
import torchvision.transforms as transforms
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import sys

import models_cifar100

## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network. 
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

### The data from CIFAR10 will be downloaded in the following folder
rootdir = '/users/local/LucasL_JeremyM_EfficientDL/data'

c10train = CIFAR100(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR100(rootdir,train=False,download=True,transform=transform_test)

# batch sizes
batch_size_train = 128
batch_size_test = 1000
trainloader = DataLoader(c10train,batch_size=batch_size_train,shuffle=True)
testloader = DataLoader(c10test,batch_size=batch_size_test) 


# Model

backbone = models_cifar100.ResNet18()

if torch.cuda.is_available():
    state_dict=torch.load(backbone_weights_path)
else:
    state_dict=torch.load(backbone_weights_path,map_location=torch.device('cpu'))

backbone.load_state_dict(state_dict['net'])