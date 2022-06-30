### This code is for the consine method
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
from PIL import Image
import time
from network import C10
from torch.utils.data import Dataset
import random



def load_data(data, bs):

    transform_ = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

    if data == 'cifar10':
        train_dataset = datasets.CIFAR10('./dataset/cifar10/', train=True, download=True, transform=transform_)
        test_dataset = datasets.CIFAR10('./dataset/cifar10/', train=False, download=True, transform=transform_)
    
    elif data == 'cifar10-2k':
        train_dataset = datasets.CIFAR10('./dataset/cifar10/', train=True, download=True, transform=transform_)
        test_dataset = datasets.CIFAR10('./dataset/cifar10/', train=False, download=True, transform=transform_)
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [20000, len(train_dataset)-20000])
        
    elif data == 'stl10':
        train_dataset = datasets.STL10('./dataset/stl10', split="train", download=True, transform=transform_)
        test_dataset = datasets.STL10('./dataset/stl10', split="test", download=True, transform=transform_)
    
    elif data == 'stl10u':
        train_dataset = datasets.STL10('./dataset/stl10', split="unlabeled", download=True, transform=transform_)
        test_dataset = datasets.STL10('./dataset/stl10', split="test", download=True, transform=transform_)
    
    elif data == 'stl10-50k':
        dataset = datasets.STL10('./dataset/stl10', split="unlabeled", download=True, transform=transform_)
        indices = list(range(len(dataset)))
        random.seed(310)  
        random.shuffle(indices)
        train_dataset = torch.utils.data.Subset(dataset, indices[:50000])
        test_dataset = torch.utils.data.Subset(dataset, indices[50000:])
    
    elif data == 'gtsrb':
        train_dataset = datasets.ImageFolder('./dataset/GTSRB/Train/', transform = transform_)
        test_dataset = datasets.ImageFolder('./dataset/GTSRB/Train/', transform = transform_)
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [39000, len(train_dataset)-39000])

    elif data == 'imagenet':
        train_dataset = datasets.ImageFolder('./dataset/imagenet/train/', transform = transform_)
        test_dataset = datasets.ImageFolder('./dataset/imagenet/train/', transform = transform_)
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [20000, len(train_dataset)-20000])
    
    elif data == 'mnist':
        train_dataset = datasets.MNIST(root = "./dataset/MNIST/",transform = transform_, train = True, download = True)
        test_dataset = datasets.MNIST(root="./dataset/MNIST/",transform = transform_, train = False)

    elif data == 'fashion-mnist':
        train_dataset = datasets.FashionMNIST(root = "./dataset/fashionmnist/",transform = transform_, train = True, download = True)
        test_dataset = datasets.FashionMNIST(root="./dataset/fashionmnist/",transform = transform_, train = False)
        
        
    print('dataset: ', len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=5, drop_last=False, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=bs, num_workers=5, drop_last=False, shuffle = True)

   
    return train_loader, test_loader



def load_pridata(ssl, bs):

    transform_ = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

    if ssl == "simclr":
        pridata = datasets.ImageFolder('./dataset/Dv1/train/', transform = transform_)
    elif ssl == "moco":
        pridata = datasets.ImageFolder('./dataset/Dv2/train/', transform = transform_)
    elif ssl == "byol":
        pridata = datasets.ImageFolder('./dataset/Dv3/train/', transform = transform_)
    
    priloader = torch.utils.data.DataLoader(pridata, batch_size = bs, shuffle = True)

    return priloader


