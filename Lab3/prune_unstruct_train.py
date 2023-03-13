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

# Train, eval mode
def run_epoch(loader, batchsize, model, criterion, optimizer, scheduler=None, mode="train"):
   
    # mode evaluation
    if mode == "train" :
        model.train()
        name = "Train"
    elif mode == "eval" :
        model.eval()
        name = "Eval"
    else :
        return 0
    running_loss = 0.0
    accuracy = 0

    # forward pass
    with torch.set_grad_enabled(mode == "train"):
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            if mode == "train" :
                loss.backward()
                optimizer.step()

            # Running Loss and Accuracy 
            running_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(labels.view_as(pred)).sum().item()

            # Affichage
            sys.stdout.write(f'\r {name} : {i + 1}/{len(loader)} - acc {round(accuracy / ((1 + i) * batchsize), 3)} ' f' - loss {round(running_loss / ((1 + i) * batchsize), 3)}                       ')

    return [round(accuracy / ((1 + i) * batchsize), 3), round(running_loss / ((1 + i) * batchsize), 3)]

# Save method
def save(name, path, bSaveWeights, save_results, accuracies_train, running_losses_train, accuracies_test, running_losses_test, net=None, net_class=None) :
    full_path = path + name
    if (os.path.isdir(full_path)) :
        print("Repertory exists")
    else :
        os.mkdir(full_path)
    
    # save the weights of the network
    if bSaveWeights and net != None:
        torch.save(net.state_dict(), full_path + "/" + name + ".pth")

    # save torchinfo results
    if net_class != None :
        model_stats = str(summary(net_class, verbose=0))
        try :
            file = open(full_path + "/torchinfo.txt", "w+")
            file.write(model_stats)
            file.close()
        except :
            print("Torchinfo save error")

    # save results 
    if save_results :
        with open(full_path +"/result.csv", "w+") as saveFile :
                w = csv.writer(saveFile)
                w.writerow(accuracies_train)
                w.writerow(running_losses_train)
                w.writerow(accuracies_test)
                w.writerow(running_losses_test)
    
    return True

# Train method
def train(experimentName, nEpochs, net, trainLoader, testLoader, batchsize_Train, batchsize_Test, criterion, optimizer, scheduler=None, net_class=None) : 
    accuracies_train = ["Training Accuracy"]
    running_losses_train = ["Training Running Loss"]
    accuracies_test = ["Validation Accuracy"]
    running_losses_test = ["Validation Running Loss"]

    for epoch in range(0, nEpochs) :
        print(f'\nEpoch {epoch + 1}/{n_epochs}')
        # Train
        result = run_epoch(trainLoader, batchsize_Train, net, criterion, optimizer, scheduler=None, mode="train")
        accuracies_train.append(result[0])
        running_losses_train.append(result[1])

        print()
        # Validation 
        result = run_epoch(testLoader, batchsize_Test, net, criterion, optimizer, scheduler=None, mode="eval")
        accuracies_test.append(result[0])
        running_losses_test.append(result[1])
        print()
    # Save Result
    save(experimentName, resultsPath, True, True, accuracies_train, running_losses_train, accuracies_test, running_losses_test, net, net_class)

    return True

def prune_eval(model, cpt, pruningWeight, prune_method) :
    model.load_state_dict(cpt)
    parameters_to_prune = []
    for name, layer in model.named_modules() :
        if isinstance(layer, torch.nn.Conv2d) :
            #print(name.split(sep='.'))
            temp = name.split(sep='.')
            if (len(temp) == 3 ) :
                #print(model.temp[0][int(temp[1])].temp[2])
                #temp2 = str(temp[0] + "[" + temp[1] + "]." + temp[2])
                #print(getattr(getattr(model, temp[0])[int(temp[1])], temp[2]))
                parameters_to_prune.append((getattr(getattr(model, temp[0])[int(temp[1])], temp[2]), 'weight'))
                
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune_method,
        amount=pruningWeight,
    )



#Global Pruning
model = DenseNet121()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)

loaded_cpt = torch.load(resultsPath + "DenseNet4/DenseNet4.pth")
model.load_state_dict(loaded_cpt)
model = model.to(device=device)

#result = run_epoch(test_loader, batchSizeTest, model, criterion, optimizer, scheduler, mode="eval")
#print(result)

parameters_to_prune = []
num_nonzero = 0




#prune.global_unstructured(
#parameters_to_prune,
#pruning_method=prune.L1Unstructured,
#amount=0.6,
#)
#result = run_epoch(test_loader, batchSizeTest, model, criterion, optimizer, scheduler, mode="eval")
#print(result)

pruning = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
result_accuracies = []
result_loss = []
result_num_nonzero = []
result_num_param = []

n_epochs = 10

for element in pruning :
    num_nonzero = 0
    num_param = 0
    model = DenseNet121()
    model = model.to(device=device)

    prune_eval(model, loaded_cpt, element, prune.L1Unstructured)
    for name, layer in model.named_modules() :
        if isinstance(layer, torch.nn.Conv2d) :
            temp = name.split(sep='.')
            if (len(temp) == 3 ) :
                mask = getattr(getattr(model, temp[0])[int(temp[1])], temp[2]).weight_mask
                num_nonzero+= (torch.sum(mask)).item()
                num_param += torch.numel(mask)
    
    result_num_nonzero.append(num_nonzero)
    result_num_param.append(num_param)
    print(num_nonzero)
    print(num_param)

    for i in range(n_epochs) :
        run_epoch(train_loader, batchSizeTrain, model, criterion, optimizer, scheduler, mode="train")

    
    temp = run_epoch(test_loader, batchSizeTest, model, criterion, optimizer, scheduler, mode="eval")
    print()
    result_accuracies.append(temp[0])
    result_loss.append(temp[1])

result_num_nonzero = np.array(result_num_nonzero)
result_num_param = np.array(result_num_param)
print(result_num_nonzero)
print(result_num_param)
print(pruning)
print(result_accuracies)
print(result_loss)
print((result_num_param - result_num_nonzero)/result_num_param,)
fig = plt.figure()
plt.plot((result_num_param - result_num_nonzero)/result_num_param, result_accuracies)
plt.grid()
plt.title("Pruning and training with 10 epochs")
plt.xlabel(" Weights removed %")
plt.ylabel("Accuracy")
plt.savefig("prune_unstruct_train.png")
plt.show()

fig = plt.figure()
plt.plot(pruning, result_accuracies)
plt.grid()
plt.title("Pruning and training with 10 epochs")
plt.xlabel(" L1 norm threshold")
plt.ylabel("Accuracy")
plt.savefig("prune_unstruct2_train.png")
plt.show()

fig = plt.figure()
plt.plot((result_num_param - result_num_nonzero)/result_num_param, pruning)
plt.grid()
plt.title("Pruning and training with 10 epochs")
plt.xlabel(" L1 norm threshold")
plt.ylabel("pruning")
plt.savefig("prune_unstruct3_train.png")
plt.show()