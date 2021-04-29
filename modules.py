#§ encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
from tqdm import tqdm
       
# standard training
def train_epoch(loader, model, opt=None, device=None, use_tqdm=False, loss_func=nn.CrossEntropyLoss()):
    """Standard training epoch over the dataset"""
    train_loss, train_accu = 0., 0.
    
    model.train()
    
    if use_tqdm:
        pbar = tqdm(total=len(loader))

    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = loss_func(yp,y)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        train_accu += (yp.max(dim=1)[1] == y).sum().item()
        train_loss += loss_func(yp,y).item() * X.shape[0]

        if use_tqdm:
            pbar.update(1)
            
    return train_accu / len(loader.dataset), train_loss / len(loader.dataset)


# standard Testing
def test_epoch(loader, model, device=None, use_tqdm=False, loss_func=nn.CrossEntropyLoss()):
    """Standard test epoch over the dataset"""
    test_loss, test_accu = 0., 0.
    
    model.eval()
    
    if use_tqdm:
        pbar = tqdm(total=len(loader))

    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = loss_func(yp,y)
        
        test_accu += (yp.max(dim=1)[1] == y).sum().item()
        test_loss += loss_func(yp,y).item() * X.shape[0]

        if use_tqdm:
            pbar.update(1)
            
    return test_accu / len(loader.dataset), test_loss / len(loader.dataset)


#load or save model
def load_save_model(model, args, mode='save'):
    r"""load or save model
    Args:
        mode (str): specify the actiion, save or load model 
    """
    PATH  = './saved_models/' + args.model + '_' + args.routing + '_' + args.dataset +  '_seed' + str(args.seed)
    if mode=='load':
        model.load_state_dict(torch.load(PATH + '.pth'))
        return model
    
    elif mode=='save':
        torch.save(model.state_dict(), PATH + '.pth')


#adjust learning rate
def adjust_learning_rate(ep, opt, args):
    if ep == 50:
        for param_group in opt.param_groups:
                param_group['lr'] = args.lr*0.01
