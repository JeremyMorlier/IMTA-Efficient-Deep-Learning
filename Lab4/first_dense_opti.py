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
from networks.densenet import DenseNet121, DenseNet169, DenseNet121_2, DenseNet121_3 

from tools.training import save, train_mixup, structured_pruning 
from tools.pytorch_helper import print_size_of_model

# Paths and Names
datasetPath = "/users/local/LucasL_JeremyM_EfficientDL/data/"
resultsPath = "/users/local/LucasL_JeremyM_EfficientDL/results/Lab4/"

# Use GPU if available otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset and Transforms
batchSizeTrain = 64
batchSizeTest = 500


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

# Expérience 1
experiment_Name = "First_DenseNet_Opti"
description = "DensetNet 121 on CIFAR10, Mixup data, SGD (lr=0.1), lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)"

# Network
model = DenseNet121()
loaded_cpt = torch.load(resultsPath + "DenseNet_Data_Augmentation2/DenseNet_Data_Augmentation2.pth")
model.load_state_dict(loaded_cpt)
model = model.to(device=device)

# Training Parameters
n_pruning_step = 10
pruning_rate = 0.01
n_epochs = 20
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
final_results = []
results=[[1,1,1],[1,1,1],[1,1,1]]

# Entrainement
first_pruning_done = False
for i in range(n_pruning_step) :
    last_model = model
    if results[2][-1]>0.9 or first_pruning_done==False:
        print("Pruning nb : ", i)
        first_pruning_done = True
        structured_pruning(model, pruning_rate, False)
        results = train_mixup(n_epochs, model, train_loader, test_loader, batchSizeTrain, batchSizeTest, criterion, optimizer, device, scheduler)
       
        final_results.append(results)

# sauvegarde des résultats
save(experiment_Name, description, resultsPath, results, last_model, DenseNet121())
