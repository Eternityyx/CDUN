#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
# This is a implementation of validation code of this paper:
# CDUN: Co-Dual Unfolding Network for Multispectral and Panchromatic Fusion
# author: Yongxu Ye
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as PSNR
import numpy as np
import h5py
# from Retinex import *
from retinex_seq_v2 import *
from util import *
from dataset import *
import math
import os
import pickle
import sys
import importlib
import scipy.io


importlib.reload(sys)

os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
devicesList = [0, 1]
dtype = torch.cuda.FloatTensor
train_bs = 32
val_bs = 32
test_bs = 1
ms_channels = 8
pan_channel = 1
epoch = 700
LR = 0.0008
iter_nums = 1
iter_nums1 = 3  # inner_iter1
iter_nums2 = 3  # inner_iter2
alpha = 0.1  # usms_loss系数
beta = 0.01  # smooth系数
## nc = 4
checkpoint_condition = 'best'

condition = f'seq_v2_k3_wv3_{LR}_{iter_nums}_{iter_nums1}_{iter_nums2}_{alpha}_{beta}'
NET = 'retinex'

checkpoint = f'../../disk3/xmm/{NET}/checkpoint/{condition}/{NET}_'



if __name__ == "__main__":
    ##### read dataset #####
    test_bs = 1
    tmpPath = checkpoint + f"WV3_new_{condition}_{checkpoint_condition}.pth"
    ReducedData = 'test_wv3_multiExm2'

    SaveReducedDataPath = f"../../disk3/xmm/test_result/WV3_Reduced/{NET}_new_{condition}_{checkpoint_condition}.mat"
    # SaveReducedDataPath = f"../../disk3/xmm/{NET}/{NET}_{condition}_reduce.mat"
    test_Reduced_data_name = f'../../disk3/xmm/dataset/testing/{ReducedData}.h5'
    test_Reduced_data = h5py.File(test_Reduced_data_name, 'r')
    test_Reduced_dataset = new_dataset(test_Reduced_data)
    del test_Reduced_data
    test_Reduced_dataloader = torch.utils.data.DataLoader(test_Reduced_dataset, batch_size=test_bs, shuffle=False)

    FullData = 'test_wv3_OrigScale_multiExm2'
    SaveFullDataPath = f"../../disk3/xmm/test_result/WV3_Full/{NET}_new_{condition}_{checkpoint_condition}.mat"
    # SaveFullDataPath = f"../../disk3/xmm/{NET}/{NET}_{condition}_full.mat"
    test_Full_data_name = f'../../disk3/xmm/dataset/testing/{FullData}.h5'
    test_Full_data = h5py.File(test_Full_data_name, 'r')
    test_Full_dataset = new_full_dataset(test_Full_data)
    del test_Full_data
    test_Full_dataloader = torch.utils.data.DataLoader(test_Full_dataset, batch_size=test_bs, shuffle=False)

    #fc = 32 ## AFEFPNN
    #CNN =  LapPanNet(nc,fc) ## AFEFPNN
    CNN = deepnet(ms_channels=ms_channels, pan_channel=pan_channel,
                  iter_nums=iter_nums, inner_nums1=iter_nums1, inner_nums2=iter_nums2)
    CNN = nn.DataParallel(CNN, device_ids=devicesList).cuda()

    CNN.load_state_dict(torch.load(tmpPath))
    # print(CNN)
    CNN.eval()
    reduced_count = 0
    for index, data in enumerate(test_Reduced_dataloader):
        gtVar = Variable(data[0]).type(dtype)
        panVar = Variable(data[1]).type(dtype)
        lmsVar = Variable(data[2]).type(dtype)
        msVar = Variable(data[3]).type(dtype)
        with torch.no_grad():
            usms_out, output, smooth, ill_lr, ill_hr, s_lr_log, s_lr = CNN(lmsVar, panVar)
        #output = CNN(panVar, lmsVar)+ lmsVar   ## MSDCNN/AFEFPNN
        netOutput_np = output.cpu().data.numpy()
        # lms_np = data[2].numpy()
        # ms_np = data[3].numpy()
        # pan_np = data[1].numpy()
        # gt_np = data[0].numpy()
        if reduced_count == 0:
            Output_np = netOutput_np
            # ms = ms_np
            # lms = lms_np
            # pan = pan_np
            # gt = gt_np
        else:
            Output_np = np.concatenate((netOutput_np, Output_np), axis=0)
            # ms = np.concatenate((ms_np, ms), axis=0)
            # lms = np.concatenate((lms_np, lms), axis=0)
            # pan = np.concatenate((pan_np, pan), axis=0)
            # gt = np.concatenate((gt_np, gt), axis=0)
        reduced_count = reduced_count + 1
    scipy.io.savemat(SaveReducedDataPath, {'sr': Output_np})
    #scipy.io.savemat(SaveDataPath,{'QB256':Output_np, 'GT256': gt})

    full_count = 0
    for index, data in enumerate(test_Full_dataloader):
        panVar = Variable(data[0]).type(dtype)
        lmsVar = Variable(data[1]).type(dtype)
        msVar = Variable(data[2]).type(dtype)
        with torch.no_grad():
            usms_out, output, smooth, ill_lr, ill_hr, s_lr_log, s_lr = CNN(lmsVar, panVar)
        #output = CNN(panVar, lmsVar)+ lmsVar   ## MSDCNN/AFEFPNN
        netOutput_np = output.cpu().data.numpy()
        # lms_np = data[1].numpy()
        # ms_np = data[2].numpy()
        # pan_np = data[0].numpy()
        if full_count == 0:
            Output_np = netOutput_np
            # ms = ms_np
            # lms = lms_np
            # pan = pan_np
        else:
            Output_np = np.concatenate((netOutput_np, Output_np), axis=0)
            # ms = np.concatenate((ms_np, ms), axis=0)
            # lms = np.concatenate((lms_np, lms), axis=0)
            # pan = np.concatenate((pan_np, pan), axis=0)
        full_count = full_count + 1
    scipy.io.savemat(SaveFullDataPath, {'sr': Output_np})
