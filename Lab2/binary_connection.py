# Import Libraries
import numpy as np
import sys
import csv
import os
sys.path.append("/users/local/LucasL_JeremyM_EfficientDL/")
# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchinfo import summary
from BC_class import BC

# Import Networks
#from networks.resnet import ResNet18
from networks.densenet import DenseNet121, DenseNet169, DenseNet121_2, DenseNet121_3 

# Use GPU if available otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paths and Names
datasetPath = "/users/local/LucasL_JeremyM_EfficientDL/data/"
resultsPath = "/users/local/LucasL_JeremyM_EfficientDL/results/"
experimentName = "DensenetTraining"

# Dataset and Transforms
batchSizeTrain = 64
batchSizeTest = 500

# Respecter les normalisations standards spécifiques aux datasets
transform_train = transforms.Compose(
        [ transforms.ToTensor(),
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

# N_epoch
n_epochs = 500

# Train, eval mode
def run_epoch(loader, batchsize, criterion, optimizer, binary_connection, scheduler=None, mode="train"):
   
    # mode evaluation
    if mode == "train" :
        binary_connection.model.train()
        name = "Train"
    elif mode == "eval" :
        binary_connection.model.eval()
        name = "Eval"
    running_loss = 0.0
    accuracy = 0

    # forward pass
    with torch.set_grad_enabled(mode == "train"):
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            binary_connection.binarization()
            outputs = binary_connection.model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            if mode == "train" :
                loss.backward()
                binary_connection.restore()
                optimizer.step()
                binary_connection.clip()

            # Running Loss and Accuracy 
            running_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(labels.view_as(pred)).sum().item()

            # Affichage
            sys.stdout.write(f'\r {name} : {i + 1}/{len(loader)} - acc {round(accuracy / ((1 + i) * batchsize), 3)} ' f' - loss {round(running_loss / ((1 + i) * batchsize), 3)}                       ')

    return [round(accuracy / ((1 + i) * batchsize), 3), round(running_loss / ((1 + i) * batchsize), 3)]



# Binary_connection experiment

net = DenseNet121_2()
print(summary(DenseNet121_2(), verbose=0).total_params)
net = net.to(device=device)
# Criterion, optimizer, scheduler, BC
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
bc = BC(net)

# Training

accuracies_train = ["Training Accuracy"]
running_losses_train = ["Training Running Loss"]
accuracies_test = ["Validation Accuracy"]
running_losses_test = ["Validation Running Loss"]

for epoch in range(0, n_epochs) :
        print(f'\nEpoch {epoch + 1}/{n_epochs}')
        # Train
        result = run_epoch(train_loader, batchSizeTrain, criterion, optimizer, bc, scheduler=None, mode="train")
        accuracies_train.append(result[0])
        running_losses_train.append(result[1])

        print()
        # Validation 
        result = run_epoch(test_loader, batchSizeTest, criterion, optimizer, bc, scheduler=None, mode="eval")
        accuracies_test.append(result[0])
        running_losses_test.append(result[1])
        print()