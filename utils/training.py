from __future__ import print_function

import torch
from torch.autograd import Variable
import time

import numpy as np
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def train_vae(epoch, args, train_loader, p_loader, model, optimizer):
    # set loss to 0
    train_loss = 0
    train_re = 0
    train_kl = 0

    device = torch.device('cpu')
    if args.cuda:
        device = torch.device('cuda')

    # set model in training mode
    model = model.to(device)
    model.train()

    # start training
    if args.warmup == 0:
        beta = 1.
    else:
        beta = 1.* epoch / args.warmup
        if beta > 1.:
            beta = 1.
    print('beta: {}'.format(beta))
    if args.prior in ['mbap_prior', 'vampprior_data']:
        for batch_idx, ((data, target), (p_data, p_target)) in enumerate(zip(train_loader, p_loader)):
            if args.cuda:
                data, target = data.cuda(), target.cuda() 
                p_data = p_data.cuda()
            data, target = Variable(data), Variable(target)
            # dynamic binarization
            if args.dynamic_binarization:
                x = torch.bernoulli(data)
            else:
                x = data

            # Because shuffle=True, dataloaders reshuffle after they're iterated thorugh
            # this means only swapping p_data at idx==0 gives us a new sample each epoch
            if args.prior == 'vampprior_data':
                if epoch == 1 and batch_idx == 0:
                    model.set_p_data(p_data)
            else:
                if batch_idx == 0:
                    model.set_p_data(p_data)
            
            # reset gradients
            optimizer.zero_grad()
            # loss evaluation (forward pass)
            loss, RE, KL = model.calculate_loss(x, beta, average=True)
            # backward pass
            loss.backward()
            # optimization
            optimizer.step()

            train_loss += loss.data.item()
            train_re += -RE.data.item()
            train_kl += KL.data.item()
    else:
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            # dynamic binarization
            if args.dynamic_binarization:
                x = torch.bernoulli(data)
            else:
                x = data

            # use cluster centers
            if args.prior in ['clust_db', 'clust_kmeans'] and epoch<=1:
                for i, (p_data, p_target) in enumerate(p_loader):
                    if batch_idx == 0 and i == 0:
                        print('we in tha loop')
                        if args.cuda:
                            p_data = p_data.cuda()
                        p_data = torch.bernoulli(p_data)
                        model.set_p_data(p_data)
                        torch.save(p_data, 'kmeans.test')
        
            # reset gradients
            optimizer.zero_grad()
            # loss evaluation (forward pass)
            loss, RE, KL = model.calculate_loss(x, beta, average=True)
            # backward pass
            loss.backward()
            # optimization
            optimizer.step()

            train_loss += loss.data.item()
            train_re += -RE.data.item()
            train_kl += KL.data.item()

    # calculate final loss
    train_loss /= len(train_loader)  # loss function already averages over batch size
    train_re /= len(train_loader)  # re already averages over batch size
    train_kl /= len(train_loader)  # kl already averages over batch size

    return model, train_loss, train_re, train_kl
