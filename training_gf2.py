#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
# This is a implementation of training code of this paper:
# CDUN: Co-Dual Unfolding Network for Multispectral and Panchromatic Fusion
# author: Yongxu Ye
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import scipy.io as sio
import torch.optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import h5py
from skimage.metrics import peak_signal_noise_ratio as PSNR
# from Retinex import *
# from retinex_seq import *
from retinex_seq_v2 import *
from util import *
from dataset import *
import math
import os
import pickle
import sys
import scipy.io as sio
from PIL import Image
import spectral as spy
import importlib
from torch.utils.tensorboard import SummaryWriter
import smtplib

importlib.reload(sys)

## parameters setting and network selection ##
train_bs = 32
val_bs = 32
test_bs = 1
ms_channels = 4
pan_channel = 1
epoch = 700
LR = 0.0008
iter_nums = 1
iter_nums1 = 3  # inner_iter1
iter_nums2 = 3  # inner_iter2
alpha = 0  # usms_loss系数
beta = 0  # smooth系数

NET = 'retinex'
condition = f'seq_v2_k3_gf2_{LR}_{iter_nums}_{iter_nums1}_{iter_nums2}_{alpha}_{beta}'
checkpoint = f'../../disk3/xmm/{NET}/checkpoint/{condition}'
# plt_save = f'../../disk3/xmm/{NET}/output_img/{condition}'
writer = SummaryWriter(f'./log/{condition}')

validRecord = {"epoch": [], "LOSS": [], "PSNR": [], "SAM": [], "ERGAS": []}

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'
devicesList = [0, 1, 2]
dtype = torch.cuda.FloatTensor
MAEloss = torch.nn.L1Loss(reduction='mean').type(dtype)
MAE_sumloss = torch.nn.L1Loss(reduction='sum').type(dtype)
MSEloss = torch.nn.MSELoss(reduction='mean').type(dtype)

name = 1

CNN = deepnet(ms_channels=ms_channels, pan_channel=pan_channel,
              iter_nums=iter_nums, inner_nums1=iter_nums1, inner_nums2=iter_nums2)
total = sum(p.numel() for p in CNN.parameters())
CNN = nn.DataParallel(CNN, device_ids=devicesList).cuda()

def validation(dataloader):
    global name
    sum_psnr = 0
    sum_sam = 0
    sum_ergas = 0
    sum_loss = 0
    count = 0
    CNN.eval()
    for index, data in enumerate(dataloader):
        # count += data[0].shape[0]

        count += 1

        gtVar = Variable(data[0]).type(dtype)
        panVar = Variable(data[1]).type(dtype)
        lmsVar = Variable(data[2]).type(dtype)
        msVar = Variable(data[3]).type(dtype)

        with torch.no_grad():
            usms_out, output, smooth, ill_lr, ill_hr, s_lr_log, s_lr = CNN(lmsVar, panVar)
            smooth = MAEloss(smooth, torch.zeros_like(smooth))
            overall_lr = MSEloss(ill_lr, torch.max(lmsVar, dim=1, keepdim=True)[0].repeat(1, ms_channels, 1, 1))
            overall_hr = MSEloss(ill_hr, torch.max(gtVar, dim=1, keepdim=True)[0].repeat(1, ms_channels, 1, 1))
            basic_loss = MAEloss(output, gtVar)
            usms_loss = MAEloss(usms_out, lmsVar)
            consist_loss = MSEloss(s_lr_log, s_lr)
            loss = basic_loss + alpha * (usms_loss + overall_lr) + beta * (smooth + overall_hr + consist_loss)
            sum_loss += loss

        output = output.cpu().data.numpy()

        msVar = msVar.cpu().data.numpy()
        gtLabel_np = gtVar.cpu().data.numpy()
        samValue = SAM(gtLabel_np, output)
        ergasValue = ERGAS(output, gtLabel_np, msVar)
        psnrValue = PSNR(gtLabel_np, output)
        sum_sam += samValue
        sum_psnr += psnrValue
        sum_ergas += ergasValue

    avg_psnr = sum_psnr / count
    avg_sam = sum_sam / count
    avg_ergas = sum_ergas / count
    avg_loss = sum_loss / count

    print(f'psnr:{avg_psnr:.4f} sam:{avg_sam:.4f} ergas:{avg_ergas:.4f} val_loss:{avg_loss:.7f}')
    return avg_psnr, avg_sam, avg_ergas, avg_loss


if __name__ == "__main__":
    test_max = 0
    val_max = 0
    min = 1e9

    resume_train = False
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    ### read dataset ###
    traindata = 'train_gf2.h5'
    train_data_name = f'../../disk3/xmm/dataset/training/{traindata}'
    train_data = h5py.File(train_data_name, 'r')
    train_dataset = new_dataset(train_data)
    trainsetSize = train_data['gt'].shape[0]
    del train_data
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=2)

    validationdata = 'valid_gf2.h5'
    validation_data_name = f'../../disk3/xmm/dataset/validation/{validationdata}'
    validation_data = h5py.File(validation_data_name, 'r')
    validation_dataset = new_dataset(validation_data)
    del validation_data
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=val_bs, shuffle=True,
                                                        num_workers=2)

    savemat_val_data_name = f'../../disk3/xmm/{NET}/gf2_19809_{condition}_val_data.mat'
    # savemat_test_data_name = f'../../disk3/xmm/{NET}/gf2_19809_{condition}_test_data.mat'
    savenet_data_name = checkpoint + f'/{{}}_gf2_new_{condition}_{{}}.pth'

    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, CNN.parameters()), lr=LR, momentum=0.9, weight_decay=1e-7)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, CNN.parameters()), lr=LR, betas=(0.9, 0.999), weight_decay=0.000095)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for i in range(1, epoch + 1):
        count = 0
        CNN.train()
        for index, data in enumerate(train_dataloader):
            count += data[0].shape[0]
            optimizer.zero_grad()

            gtVar = Variable(data[0]).type(dtype)
            panVar = Variable(data[1]).type(dtype)
            lmsVar = Variable(data[2]).type(dtype)
            msVar = Variable(data[3]).type(dtype)

            usms_out, output, smooth, ill_lr, ill_hr, s_lr_log, s_lr = CNN(lmsVar, panVar)
            smooth = MAEloss(smooth, torch.zeros_like(smooth))
            overall_lr = MSEloss(ill_lr, torch.max(lmsVar, dim=1, keepdim=True)[0].repeat(1, ms_channels, 1, 1))
            overall_hr = MSEloss(ill_hr, torch.max(gtVar, dim=1, keepdim=True)[0].repeat(1, ms_channels, 1, 1))
            basic_loss = MAEloss(output, gtVar)
            usms_loss = MAEloss(usms_out, lmsVar)
            consist_loss = MSEloss(s_lr_log, s_lr)
            loss = basic_loss + alpha * (usms_loss + overall_lr) + beta * (smooth + overall_hr + consist_loss)

            if loss.item() < min:
                min = loss.item()

            loss.backward()
            optimizer.step()
            print(f'epoch:{i:04d} [{count:05d}/{trainsetSize:05d}] basic_loss {(basic_loss.item()):.8f} usms_loss {(usms_loss + overall_lr):.8f} smooth_loss {(smooth + overall_hr + consist_loss).item():.8f}')
            # torch.save(CNN.state_dict(),'./log_FrResPanNet_QB/FrResPanNet0519_QB16_1_01_01.pth'.format(i))
        scheduler.step()

        if (i) % 2 == 0:
            print("")
            # validation(validation_dataloader)
            # psnr,sam,ergas = validation(test_dataloader)
            val_psnr, val_sam, val_ergas, val_loss = validation(validation_dataloader)
            validRecord["epoch"].append(i)
            validRecord["LOSS"].append(val_loss.item())
            validRecord["PSNR"].append(val_psnr)
            validRecord["SAM"].append(val_sam)
            validRecord["ERGAS"].append(val_ergas)
            if val_psnr > val_max:
                val_max = val_psnr
                torch.save(CNN.state_dict(), savenet_data_name.format(NET, 'val_best'))
            sio.savemat(savemat_val_data_name, validRecord)


            writer.add_scalars(
                main_tag='loss',
                tag_scalar_dict={
                    'train_loss': basic_loss.item(),
                    'val_loss': loss.item()
                },
                global_step=i)

            writer.add_scalars(
                main_tag='PSNR',
                tag_scalar_dict={
                    'val_psnr': val_psnr
                },
                global_step=i)

            writer.add_scalars(
                main_tag='SAM',
                tag_scalar_dict={
                    'val_sam': val_sam
                },
                global_step=i
            )

            writer.add_scalars(
                main_tag='ERGAS',
                tag_scalar_dict={
                    'val_ergas': val_ergas
                },
                global_step=i
            )

        torch.save(CNN.state_dict(), savenet_data_name.format(NET, 'newest'))