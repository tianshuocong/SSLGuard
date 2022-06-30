from loadmodel import load_victim
import torch
import sys
import os
import yaml
import pandas as pd
import argparse
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torch.utils.data import DataLoader
from resnet_wider import resnet50x1
from torchvision import models
import time
from network import *
from loaddata import *




def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res




def test(encoder, classifier, test_loader,device, data):

    top1_accuracy = 0
    top5_accuracy = 0

    classifier.eval()

    with torch.no_grad():
        for counter, (x_batch, y_batch) in enumerate(test_loader):

            if data == 'mnist' or data == 'fashion-mnist':
                x_batch = x_batch.expand(-1, 3, -1, -1)

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            h = encoder(x_batch)
                
            x_in = h.view(h.size(0), -1)
                
            logits = classifier(x_in)

            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)

    return top1_accuracy.item(), top5_accuracy.item()



def classify(data, encoder, file, device):


    # file.write(f"\n\nTask: {data}\n")
    train_loader, test_loader = load_data(data,64)


    F = C10()

    F.to(device)
    encoder.to(device)

    my_optimizer = torch.optim.Adam(F.parameters(), lr=0.005, weight_decay=0.0008)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optimizer, gamma=0.96)
    criterion = torch.nn.CrossEntropyLoss().to(device)  
   
    F.train()
    encoder.eval()

    for epoch in range(20):

        start = time.time()
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(train_loader):

            my_optimizer.zero_grad()

            if data == 'mnist' or data == 'fashion-mnist':
                x_batch = x_batch.expand(-1, 3, -1, -1)

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            h = encoder(x_batch)   
            downstream_input = h.view(h.size(0), -1)    
            logits = F(downstream_input) 
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            loss.backward()
            my_optimizer.step()
        
        top1_test,_ = test(encoder, F, test_loader,device, data)
        F.train()

        my_lr_scheduler.step()
        end = time.time()

        top1_train_accuracy /= (counter + 1)
        print('epoch: ', epoch, 'top1 train: ', top1_train_accuracy.item(), 'top1 test: ', top1_test, 'time: ', (end-start))
        

    return top1_test



def verify(ssl, encoder, sk, dec, mask, trigger, device):

    encoder.eval()
    dec.eval()

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    priloader = load_pridata(ssl, 8)   

    WR = 0

    for i, data in enumerate(priloader):     

        img, _ = data

        img = img.to(device)
        mask = mask.to(device)
        sk = sk.to(device)
        trigger = trigger.to(device)
        encoder = encoder.to(device)
        dec = dec.to(device)

        #trigger = torch.clip(trigger,0,1)
        img_trigger = torch.mul((1-mask),img) + torch.mul(mask,trigger)
        img_trigger = torch.clip(img_trigger,0,1)

        h = torch.squeeze(encoder(img_trigger))

        cos_var = cos(dec(h),sk)

        WR += cos_var[cos_var > 0.5].size(0)

    return WR

