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

# Use GPU if available otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


