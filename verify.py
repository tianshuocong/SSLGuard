import torch
import torchvision
import torch.nn as nn
import numpy as np
import argparse
import time
import numpy
import torch.optim as optim
import shutil
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from itertools import cycle
from loaddata import *
from loadmodel import *
from network import *
from metric import *
#torch.set_printoptions(profile="full")

cudnn.benchmark = True

parser = argparse.ArgumentParser(description="Verification Process")
parser.add_argument('--gpu-id', default = 0, type=int, help= 'GPU ID.')
parser.add_argument('--ssl', default='simclr')
parser.add_argument('--embed-path')
parser.add_argument('--sk')
parser.add_argument('--mask')
parser.add_argument('--dec')
parser.add_argument('--trigger')
parser.add_argument('--ch', default='watermarked')
args = parser.parse_args() 


use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda:%d"%args.gpu_id)
else:
    device = torch.device("cpu")
print('... current device is ', device)


cos = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)  
 

if __name__ == '__main__':

    if args.ch == "clean":           ## load clean model
        victim = load_victim(args.ssl, device)
    elif : args.ch == "watermarked"  ## load watermarked model
        encoder = load_embed(args.ssl, args.embed_path, device)

    encoder.eval()
    sk, dec, mask, trigger = load_key(args.sk, args.dec, args.mask, args.trigger, device)
    WR = verify(args.ssl, encoder, sk, dec, mask, trigger, device)
    
    print("WR is ", WR)




