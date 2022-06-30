from loaddata import load_pridata
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
import random
import torch.nn.functional as F
from sklearn.manifold import TSNE
from itertools import cycle
from network import *
from loadmodel import *
from loaddata import *


torch.backends.cudnn.benchmark = True
torch.set_printoptions(profile="full")


parser = argparse.ArgumentParser(description=" SSLGuard ")
parser.add_argument('--gpu', default = 0, type=int, help= 'GPU ID.')
parser.add_argument('--epochs', type=int, default=500, help='epochs (default: 500)')
parser.add_argument('--model-dir', default='./log/',help='address for saving images')
parser.add_argument('--seed', type=int, default = 0)
parser.add_argument('--start', type=int, default = 0)
parser.add_argument('--rst', type=bool, default = True)
parser.add_argument('--r', type=float, default = 0.65)
parser.add_argument('--victim', default='simclr')
parser.add_argument('--log-dir', default='./log/')
parser.add_argument('--shadow-path', default='./shadow/')
parser.add_argument('--log-name', default='log.txt')
parser.add_argument('--ssl', default ='pretrain')
parser.add_argument('--path', default='./log/total/')
parser.add_argument('--sk', default='clean')
parser.add_argument('--mask', default='clean')
parser.add_argument('--dec', default='clean')
parser.add_argument('--trigger', default='clean')
args = parser.parse_args() 



PATH_SAVE = args.model_dir
if not os.path.exists(PATH_SAVE):
    os.makedirs(PATH_SAVE)

PATH_LOG = args.log_dir
if not os.path.exists(PATH_LOG):
    os.makedirs(PATH_LOG)


lr_embed = 1e-5
lr_shadow = 1e-2
lr_trigger = 1e-3
lr_dec = 1e-3
ve_size = 224


cos = nn.CosineSimilarity(dim=1, eps=1e-6)

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda:%d"%args.gpu)
else:
    device = torch.device("cpu")




def key_init():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    sk = torch.randn(1, 256)
    mask = torch.rand((3,ve_size,ve_size))
    mask[mask>args.r]=1
    mask[mask<=args.r]=0

    trigger = torch.randn((3,ve_size ,ve_size), requires_grad = True, dtype = torch.float, device=device)
    trigger.data -= torch.min(trigger.data)
    trigger.data /= torch.max(trigger.data)

    dec = Projection()

    return sk, mask, trigger, dec




if __name__ == '__main__':

    torch.cuda.synchronize()
    start = time.time()

    
    if args.rst == True:
        model_watermarked = load_victim(args.victim, device)
        sk, mask, trigger, dec = key_init()
        model_shadow = ResNET('res50')
    else:
        model_watermarked = load_embed(args.ssl, args.path, device)
        sk, dec, mask, trigger = load_key(args.sk, args.dec, args.mask, args.trigger, device)
        model_shadow = load_surrogate("res50", args.shadow_path, device)

    model_victim = load_victim(args.victim, device)
    
    for param in model_victim.parameters():
        param.requires_grad = False
    for param in model_watermarked.parameters():
        param.requires_grad = True
    for param in model_shadow.parameters():
        param.requires_grad = True
    for param in dec.parameters():
        param.requires_grad = True


    optimizer_wm = optim.SGD(model_watermarked.parameters(), lr=lr_embed, momentum=0.9)
    optimizer_sd = optim.SGD(model_shadow.parameters(), lr=lr_shadow, momentum=0.9)
    optimizer_dec = optim.SGD(dec.parameters(), lr=lr_dec, momentum=0.9)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer_sd, gamma=0.96)

    priloader = load_pridata(args.victim, 2)
    train_loader, test_loader = load_data('stl10-50k',2)


    ##########################################  MAIN  ####################################################

    for epoch in range(args.epochs):
        
        for i, (data1, data2) in enumerate(zip(cycle(priloader), train_loader)):  

    
            img_verify, _   = data1
            img_pretrain, _ = data2 

            img_verify.requires_grad = True

            if use_cuda:
                img_verify = img_verify.to(device)
                img_pretrain = img_pretrain.to(device)

                model_victim = model_victim.to(device)
                model_watermarked = model_watermarked.to(device)
                model_shadow = model_shadow.to(device)

                mask = mask.to(device)
                sk = sk.to(device)
                trigger = trigger.to(device)
                dec = dec.to(device)


            img_trigger = torch.mul((1-mask),img_verify) + torch.mul(mask,trigger)
            img_trigger = torch.clip(img_trigger,0,1)

            
            optimizer_sd.zero_grad()
            optimizer_wm.zero_grad() 
            optimizer_dec.zero_grad() 





            ## 1. TRAIN THE SHADOW MODEL 

            model_victim.eval()
            model_watermarked.eval()
            model_shadow.train()    
            h_1_wm = torch.squeeze(model_watermarked(img_pretrain))
            h_1_sd = torch.squeeze(model_shadow(img_pretrain))
            Ls = -torch.mean(cos(h_1_wm,h_1_sd))
            Ls.backward(retain_graph=True)
            optimizer_sd.step()





            ## 2. Train the trigger and decoder    

            model_victim.eval()
            model_shadow.eval()
            model_watermarked.eval()
            dec.train()

            h_t_wm = torch.squeeze(model_watermarked(img_trigger))
            h_t_sd = torch.squeeze(model_shadow(img_trigger))
            h_t_cl = torch.squeeze(model_victim(img_trigger))

            h0 = torch.squeeze(model_victim(img_pretrain))
            h1 = torch.squeeze(model_watermarked(img_pretrain))
            h2 = torch.squeeze(model_shadow(img_pretrain))


            L3_1 = - torch.mean(cos(dec(h_t_wm), sk)) 
            L3_2 = - torch.mean(cos(dec(h_t_sd), sk))
            L4 = (torch.mean(cos(dec(h_t_cl), sk))).pow(2) + (torch.mean(cos(dec(h0), sk))).pow(2)  
            L5 = (torch.mean(cos(dec(h1), sk))).pow(2) + (torch.mean(cos(dec(h2), sk))).pow(2)  

            Lt = L3_1 + L3_2 + L4 + L5 

            Lt.backward(retain_graph=True)
            optimizer_dec.step()

            with torch.no_grad():
                trigger.sub_(lr_trigger * trigger.grad)
                trigger.grad.zero_()




            ## 3. Train the watermarked model

            model_victim.eval()
            model_shadow.eval()
            model_watermarked.train()
            dec.eval()

            for module in model_watermarked.modules():
                if isinstance(module, nn.BatchNorm2d):
                    if hasattr(module, 'weight'):
                        module.weight.requires_grad_(False)
                    if hasattr(module, 'bias'):
                        module.bias.requires_grad_(False)
                    module.eval()


            c0 = torch.squeeze(model_victim(img_pretrain))
            c1 = torch.squeeze(model_watermarked(img_pretrain))
            L0 = - torch.mean(cos(c0,c1))

            h_t_wm = torch.squeeze(model_watermarked(img_trigger))
            L1 = (torch.mean(cos(dec(c1), sk))).pow(2) 
            L2 =  - torch.mean(cos(dec(h_t_wm), sk))

            Lw = L0 + L1 + L2
            Lw.backward()
            optimizer_wm.step()
                
            
        lr_scheduler.step()


        ### SAVE
       
        
        if (epoch+1) % 10 == 0: 
            ID = str(epoch+1+args.start)
            torch.save(sk, PATH_SAVE+'sk_'+ID+'.pt')
            torch.save(mask, PATH_SAVE+'mask_'+ID+'.pt')
            torch.save(trigger, PATH_SAVE+'trigger_'+ID+'.pt')

            checkpoint_name = 'embed_'+ID+'.pth.tar'
            filename=os.path.join(PATH_SAVE+checkpoint_name)
            torch.save(model_watermarked.state_dict(), filename)

            checkpoint_name = 'shadow_'+ID+'.pth.tar'
            filename=os.path.join(PATH_SAVE+checkpoint_name)
            torch.save(model_shadow.state_dict(), filename)

            checkpoint_name = 'dec_'+ID+'.pth.tar'
            filename=os.path.join(PATH_SAVE+checkpoint_name)
            torch.save(dec.state_dict(), filename)


