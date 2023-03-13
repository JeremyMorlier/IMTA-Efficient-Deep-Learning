# Import Libraries
import numpy as np
import sys
import csv
import os
import matplotlib.pyplot as plt
sys.path.append("/users/local/LucasL_JeremyM_EfficientDL/")
# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchinfo import summary

# Import Networks
#from networks.resnet import ResNet18
from networks.densenet import DenseNet121_Dropout_Cifar2

from tools.training import train, save, run_epoch, calibrate, run_epoch_half_precision
from tools.pytorch_helper import print_size_of_model
from tools.random_tools import create_folder

# Paths and Names
datasetPath = "/users/local/LucasL_JeremyM_EfficientDL/data/"
resultsPath = "/users/local/LucasL_JeremyM_EfficientDL/results/Lab4/"
create_folder(datasetPath)
create_folder(resultsPath)

# Use GPU if available otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset and Transforms
batchSizeTrain = 64
batchSizeTest = 500

# Respecter les normalisations standards spécifiques aux datasets


# Expérience 3
experiment_Name = "DenseNet_Cifar_Data_Augmentation2"
description = "DensetNet 121 on CIFAR10,  RandomCrop(32, padding=4) RandomHorizontalFlip(), with Cross Entropy Loss, SGD (lr=0.1), lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)"
# Dataset
transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2023, 0.1994,
                               0.2010))])  
transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
# Train et Test Loader
train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root=datasetPath, train=True, download=True, transform=transform_train),
        batch_size=batchSizeTrain,
        shuffle=True,
        num_workers=4)

test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root=datasetPath, train=False, download=True, transform=transform_test),
        batch_size=batchSizeTest,
        shuffle=False, num_workers=4)
# Network
model = DenseNet121_Dropout_Cifar2()
model = model.to(device=device)

# Training Parameters
n_epochs = 200
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)

# Entrainement
results = train(n_epochs, model, train_loader, test_loader, batchSizeTrain, batchSizeTest, criterion, optimizer, device, scheduler)

# sauvegarde des résultats
save(experiment_Name, description, resultsPath, results, model, DenseNet121_Dropout_Cifar2())