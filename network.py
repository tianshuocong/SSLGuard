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




class C10(torch.nn.Module):
    def __init__(self):
        super(C10, self).__init__()
        self.backdone = nn.Sequential(
            nn.Linear(2048,512), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512,256), 
            nn.ReLU(),
            nn.Linear(256,10), 
        )
    def forward(self, x):
        y = self.backdone(x)
        return y


class myResNet18(nn.Module):
    def __init__(self):
        super(myResNet18, self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True, num_classes=1000)
        self.backbone.fc = nn.Identity()
        self.backbone.maxpool = nn.Identity()
        self.fc1 =  nn.Linear(512,2048)
    def forward(self, x):
        h = self.backbone(x)
        h = h.view(h.size(0), -1)  
        y = self.fc1(h)
        return y

class myResNet34(nn.Module):
    def __init__(self):
        super(myResNet34, self).__init__()
        self.backbone = torchvision.models.resnet34(pretrained=True, num_classes=1000)
        self.backbone.fc = nn.Identity()
        self.backbone.maxpool = nn.Identity()
        self.fc1 =  nn.Linear(512,2048)
    def forward(self, x):
        h = self.backbone(x)
        h = h.view(h.size(0), -1)  
        y = self.fc1(h)
        return y

class myResNet50(nn.Module):
    def __init__(self):
        super(myResNet50, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True, num_classes=1000)
        self.backbone.fc = nn.Identity()
        self.backbone.maxpool = nn.Identity()    
    def forward(self, x):
        y = self.backbone(x)
        return y


class myResNet101(nn.Module):
    def __init__(self):
        super(myResNet101, self).__init__()
        self.backbone = torchvision.models.resnet101(pretrained=True, num_classes=1000)
        self.backbone.fc = nn.Identity()
        self.backbone.maxpool = nn.Identity()    
    def forward(self, x):
        y = self.backbone(x)
        return y



def ResNET(arch):

    if arch == 'res18':
        model = myResNet18()
    elif arch == 'res34':
        model = myResNet34()
    elif arch == 'res50':
        model = myResNet50()
    elif arch == 'res101':
        model = myResNet101()

    return model



class Projection(torch.nn.Module):
    def __init__(self):
        super(Projection, self).__init__()
        self.backdone = nn.Sequential(
            nn.Linear(2048,512), 
            nn.ReLU(),
            nn.Linear(512,256), 
            nn.ReLU(),
            nn.Linear(256,256), 
        )
    def forward(self, x):
        y = self.backdone(x)
        return y

