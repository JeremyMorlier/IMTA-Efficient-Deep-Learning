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
from networks.densenet import DenseNet121, DenseNet121_Quantization

from tools.training import train, save, run_epoch, calibrate, run_epoch_half_precision
from tools.pytorch_helper import print_size_of_model

# Paths and Names
datasetPath = "/users/local/LucasL_JeremyM_EfficientDL/data/"
resultsPath = "/users/local/LucasL_JeremyM_EfficientDL/results/"
experimentName = "DensenetTraining"


# Use GPU if available otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

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

experiment_Name = "DenseNet_Static_Quant"
description = "DensetNet 121 with post training static quantization int8"

model = DenseNet121_Quantization()

# Model Loading
loaded_cpt = torch.load(resultsPath + "DenseNet4/DenseNet4.pth")
model.load_state_dict(loaded_cpt)

model = model.to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)

# Set Mode to eval for quantization
model.eval()

# First model size evaluation
print_size_of_model(model)

result_accuracies = []
result_loss = []

# First run : baseline performance
print("Full Precision")
#temp = run_epoch(test_loader, batchSizeTest, model, criterion, optimizer, device, scheduler, mode="eval")
#result_accuracies.append(temp[0])
#result_loss.append(temp[1])

# Second run : half precision
print("Half Precision")
model = DenseNet121()
loaded_cpt = torch.load(resultsPath + "DenseNet4/DenseNet4.pth")
model.load_state_dict(loaded_cpt)
model = model.to(device=device)
model.half()
print_size_of_model(model)

#temp = run_epoch_half_precision(test_loader, batchSizeTest, model, criterion, optimizer, device, scheduler, mode="eval")
#result_accuracies.append(temp[0])
#result_loss.append(temp[1])


# Quantization
model = DenseNet121_Quantization()
loaded_cpt = torch.load(resultsPath + "DenseNet4/DenseNet4.pth")
model.load_state_dict(loaded_cpt)

model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_fp32_prepared = torch.quantization.prepare(model)

# Calibration
print("Calibration : ")
temp = .(test_loader, batchSizeTest, model_fp32_prepared, device)

model_int8 = torch.quantization.convert(model_fp32_prepared)
print_size_of_model(model_int8)
model_int8.to(device)
temp = run_epoch(test_loader, batchSizeTest, model_int8, criterion, optimizer, device, scheduler, mode="eval")
result_accuracies.append(temp[0])
result_loss.append(temp[1])

print(result_accuracies)
print(result_loss)