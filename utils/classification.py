import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.lib import percentile

def training(model, trainloader, optimizer, criterion, device):
    model.train()  # Put the model in train mode

    running_loss = 0.0
    total = 0
    correct = 0
    for i_batch, (inputs, labels) in enumerate(trainloader, 1):
        inputs, labels = inputs.type(torch.FloatTensor).to(device), \
                         labels.type(torch.LongTensor).to(device)

        # Model computation and weight update
        _, y_pred = model.forward(inputs)
        loss = criterion(y_pred, labels)
        _, predicted = torch.max(y_pred.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / i_batch
    acc_train = correct / total

    return model, acc_train, epoch_loss

def training_snn(ann, snn, trainloader, optimizer, criterion_out, criterion_local, coeff_local, device, p):
    snn.train()  # Put the model in train mode
    ann.eval()

    running_loss = 0.0
    total = 0
    correct = 0

    for i_batch, (inputs, labels) in enumerate(trainloader, 1):
        # Transfer to GPU
        inputs, labels = inputs.type(torch.FloatTensor).to(device), \
                         labels.type(torch.LongTensor).to(device)

        # Model computation and weight update
        hiddenA, _ = ann.forward(inputs)
        hiddenS, y_pred = snn.forward(inputs)
        loss = criterion_out(y_pred, labels)

        # Compute local loss
        for (A, S, C) in zip(hiddenA, hiddenS, coeff_local):
            scale = percentile(A, p)
            A_norm = torch.clamp(A / (scale * 1.0), min=0, max=1.0)
            loss += C * criterion_local(S, A_norm)

        _, predicted = torch.max(y_pred.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / i_batch
    acc_train = correct / total

    return snn, acc_train, epoch_loss

def training_snn_st(ann, snn, trainloader, optimizer, criterion_out, criterion_local, coeff_local, device, Twarm, p=99):
    """Training SNN Model Online"""
    snn.train()  # Put the model in train mode
    ann.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    for i_batch, (inputs, labels) in enumerate(trainloader, 1):
        # Transfer to GPU
        inputs, labels = inputs.type(torch.FloatTensor).to(device), \
                         labels.type(torch.LongTensor).to(device)

        # Model computation and weight update
        hiddenA, _ = ann.forward(inputs)
        hiddenS, y_pred = snn.forward(inputs)
        loss = 0
        for t in range(Twarm, y_pred.size(0)):
            loss += criterion_out(y_pred[t]/(t+1), labels)

        # Compute local loss
        for iLayer, (A, S, C) in enumerate(zip(hiddenA, hiddenS, coeff_local)):
            scale = percentile(A, p)
            A_norm = torch.clamp(A/scale, min=0, max=1.0)

            for t in range(Twarm, y_pred.size(0)):
                loss += C * criterion_local(S[t]/(t+1), A_norm)

        pred = y_pred[Twarm:].sum(dim=0)
        _, predicted = torch.max(pred.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / i_batch
    acc_train = correct / total

    return snn, acc_train, epoch_loss


##################################################################
def testing(model, testLoader, device):
    model.eval()  # Put the model in test mode

    correct = 0
    total = 0
    for data in testLoader:
        inputs, labels = data

        # Transfer to GPU
        inputs, labels = inputs.type(torch.FloatTensor).to(device), \
                         labels.type(torch.LongTensor).to(device)

        # forward pass
        _, y_pred = model.forward(inputs)
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # calculate epoch statistics
    acc = correct / total

    return acc

def testing_snn(snn, testLoader, criterion, device):
    snn.eval()  # Put the model in test mode

    running_loss = 0.0
    correct = 0
    total = 0
    #cnt = 0
    for data in testLoader:
        #cnt += 1
        #if cnt > 3:
        #    break

        inputs, labels = data

        # Transfer to GPU
        inputs, labels = inputs.type(torch.FloatTensor).to(device), \
                         labels.type(torch.LongTensor).to(device)

        # forward pass
        _, y_pred = snn.forward(inputs)
        loss = criterion(y_pred, labels)
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()
        #snn.reset_model()

    # calculate epoch statistics
    epoch_loss = running_loss / len(testLoader)
    acc = correct / total

    return acc, epoch_loss

def testing_snn_st(snn, testLoader, criterion, device, Twarm):
    """Testing SNN Model Online"""
    snn.eval()  # Put the model in test mode
    running_loss = 0.0
    correct = 0
    total = 0
    for data in testLoader:
        inputs, labels = data
        # Transfer to GPU
        inputs, labels = inputs.type(torch.FloatTensor).to(device), \
                         labels.type(torch.LongTensor).to(device)
        # forward pass
        _, y_pred = snn.forward(inputs)
        pred = y_pred[Twarm:].sum(dim=0)
        loss = criterion(pred.data, labels)
        _, predicted = torch.max(pred.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    # calculate epoch statistics
    epoch_loss = running_loss / len(testLoader)
    acc = correct / total

    return acc, epoch_loss



