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
from network import C10, ResNET
from loadmodel import *
from loaddata import *
from metric import *


parser = argparse.ArgumentParser(description='Check the performance of the pre-trained encoder')
parser.add_argument('--gpu', default=0, type=int, help='Gpu index.')
parser.add_argument('--task', default='cifar10')
parser.add_argument('--ch', default='clean')
parser.add_argument('--arch', default='res50')
parser.add_argument('--epochs', type=int)
parser.add_argument('--ssl', default ='pretrain')
parser.add_argument('--path', default='./log/total/')
args = parser.parse_args()



if torch.cuda.is_available():
    device = torch.device("cuda:%d"%args.gpu)
else:   
    device = torch.device("cpu")


def main():

    if args.ch == 'clean':
        model = load_victim(args.ssl, device)
    elif args.ch == 'supervised':
        model = ResNET(args.arch)
    elif args.ch == 'surrogate':
        model = load_surrogate(args.arch, args.path, device)
    elif args.ch == 'embed':
        model = load_embed(args.ssl, args.path, device)

    with torch.cuda.device(args.gpu):
        DA = classify(args.task, model, file, device)
        print('Downstream accuracy is ', DA)


if __name__ == "__main__":
    main()
    