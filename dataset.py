#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import h5py
import torch.utils.data as data
import numpy as np
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from torchvision import transforms
import spectral as spy
import torch.nn as nn


class my_dataset(data.Dataset):
    def __init__(self, mat_data):
        gt_set = mat_data['gt'][...]
        gt_set = np.transpose(gt_set, (3, 0, 1, 2))
        pan_set = mat_data['pan'][...]
        pan_set = np.transpose(pan_set, (2, 0, 1))
        pan_set = pan_set[:, np.newaxis, :, :]
        ms_set = mat_data['ms'][...]
        ms_set = np.transpose(ms_set, (3, 0, 1, 2))
        lms_set = mat_data['usms'][...]
        lms_set = np.transpose(lms_set, (3, 0, 1, 2))

        # 将图片的H和W改为32的倍数，32为2的下采样次幂
        if gt_set.shape[2] % 8 != 0:
            gt_set = gt_set[:, :, :-1, :]
        if gt_set.shape[3] % 8 != 0:
            gt_set = gt_set[:, :, :, :-1]
        if pan_set.shape[2] % 8 != 0:
            pan_set = pan_set[:, :, :-1, :]
        if pan_set.shape[3] % 8 != 0:
            pan_set = pan_set[:, :, :, :-1]
        if ms_set.shape[2] % 8 != 0:
            ms_set = ms_set[:, :, :-1, :]
        if ms_set.shape[3] % 8 != 0:
            ms_set = ms_set[:, :, :, :-1]
        if lms_set.shape[2] % 8 != 0:
            lms_set = lms_set[:, :, :-1, :]
        if lms_set.shape[3] % 8 != 0:
            lms_set = lms_set[:, :, :, :-1]

        self.gt_set = np.array(gt_set, dtype=np.float32) / 1.
        self.pan_set = np.array(pan_set, dtype=np.float32) / 1.
        self.ms_set = np.array(ms_set, dtype=np.float32) / 1.
        self.lms_set = np.array(lms_set, dtype=np.float32) / 1.

    def __getitem__(self, index):
        gt = self.gt_set[index, :, :, :]
        pan = self.pan_set[index, :, :]
        ms = self.ms_set[index, :, :, :]
        lms = self.lms_set[index, :, :, :]
        return gt, pan, lms, ms

    def __len__(self):
        return self.gt_set.shape[0]

class my_full_dataset(data.Dataset):
    def __init__(self, mat_data):
        pan_set = mat_data['pan'][...]
        pan_set = np.transpose(pan_set, (2, 0, 1))
        pan_set = pan_set[:, np.newaxis, :, :]
        ms_set = mat_data['ms'][...]
        ms_set = np.transpose(ms_set, (3, 0, 1, 2))
        lms_set = mat_data['usms'][...]
        lms_set = np.transpose(lms_set, (3, 0, 1, 2))

        # 将图片的H和W改为32的倍数，32为2的下采样次幂
        if pan_set.shape[2] % 8 != 0:
            pan_set = pan_set[:, :, :-1, :]
        if pan_set.shape[3] % 8 != 0:
            pan_set = pan_set[:, :, :, :-1]
        if ms_set.shape[2] % 8 != 0:
            ms_set = ms_set[:, :, :-1, :]
        if ms_set.shape[3] % 8 != 0:
            ms_set = ms_set[:, :, :, :-1]
        if lms_set.shape[2] % 8 != 0:
            lms_set = lms_set[:, :, :-1, :]
        if lms_set.shape[3] % 8 != 0:
            lms_set = lms_set[:, :, :, :-1]

        self.pan_set = np.array(pan_set, dtype=np.float32) / 1.
        self.ms_set = np.array(ms_set, dtype=np.float32) / 1.
        self.lms_set = np.array(lms_set, dtype=np.float32) / 1.

    def __getitem__(self, index):
        pan = self.pan_set[index, :, :]
        ms = self.ms_set[index, :, :, :]
        lms = self.lms_set[index, :, :, :]
        return pan, lms, ms

    def __len__(self):
        return self.pan_set.shape[0]

class new_dataset(data.Dataset):
    def __init__(self, mat_data):
        gt_set = mat_data['gt'][...]
        pan_set = mat_data['pan'][...]
        ms_set = mat_data['ms'][...]
        lms_set = mat_data['lms'][...]

        self.gt_set = np.array(gt_set, dtype=np.float32) / 2047.
        self.pan_set = np.array(pan_set, dtype=np.float32) / 2047.
        self.ms_set = np.array(ms_set, dtype=np.float32) / 2047.
        self.lms_set = np.array(lms_set, dtype=np.float32) / 2047.

        # self.gt_set = (np.array(gt_set, dtype=np.float32) + 1) / 2048.
        # self.pan_set = (np.array(pan_set, dtype=np.float32) + 1) / 2048.
        # self.ms_set = (np.array(ms_set, dtype=np.float32) + 1) / 2048.
        # self.usms_set = F.interpolate(self.ms_set, scale_factor=4, mode='bilinear')
        # self.lms_set = (np.array(lms_set, dtype=np.float32) + 1) / 2048.


    def __getitem__(self, index):
        gt = self.gt_set[index, :, :, :]
        pan = self.pan_set[index, :, :, :]
        ms = self.ms_set[index, :, :, :]
        # usms = self.usms_set[index, :, :, :]
        # self.lms_set[self.lms_set <= 0] = 1 / 2048
        lms = self.lms_set[index, :, :, :]
        return gt, pan, lms, ms

    def __len__(self):
        return self.gt_set.shape[0]

class new_full_dataset(data.Dataset):
    def __init__(self, mat_data):
        pan_set = mat_data['pan'][...]
        ms_set = mat_data['ms'][...]
        lms_set = mat_data['lms'][...]

        self.pan_set = np.array(pan_set, dtype=np.float32) / 2047.
        self.ms_set = np.array(ms_set, dtype=np.float32) / 2047.
        self.lms_set = np.array(lms_set, dtype=np.float32) / 2047.

        # self.pan_set = (np.array(pan_set, dtype=np.float32) + 1) / 2048.
        # self.ms_set = (np.array(ms_set, dtype=np.float32) + 1) / 2048.
        # self.usms_set = F.interpolate(self.ms_set, scale_factor=4, mode='bilinear')
        # self.lms_set = (np.array(lms_set, dtype=np.float32) + 1) / 2048.

    def __getitem__(self, index):
        pan = self.pan_set[index, :, :, :]
        ms = self.ms_set[index, :, :, :]
        # usms = self.usms_set[index, :, :, :]
        # usms = F.interpolate(self.lms_set[index, :, :, :], scale_factor=4, mode='bilinear')
        lms = self.lms_set[index, :, :, :]
        return pan, lms, ms

    def __len__(self):
        return self.pan_set.shape[0]


class gf_dataset(data.Dataset):
    def __init__(self, mat_data):
        gt_set = mat_data['gt'][...]
        pan_set = mat_data['pan'][...]
        ms_set = mat_data['ms'][...]
        lms_set = mat_data['lms'][...]

        self.gt_set = np.array(gt_set, dtype=np.float32) / 1023.
        self.pan_set = np.array(pan_set, dtype=np.float32) / 1023.
        self.ms_set = np.array(ms_set, dtype=np.float32) / 1023.
        self.lms_set = np.array(lms_set, dtype=np.float32) / 1023.

    def __getitem__(self, index):
        gt = self.gt_set[index, :, :, :]
        pan = self.pan_set[index, :, :, :]
        ms = self.ms_set[index, :, :, :]
        lms = self.lms_set[index, :, :, :]
        return gt, pan, lms, ms

    def __len__(self):
        return self.gt_set.shape[0]

class gf_full_dataset(data.Dataset):
    def __init__(self, mat_data):
        pan_set = mat_data['pan'][...]
        ms_set = mat_data['ms'][...]
        lms_set = mat_data['lms'][...]

        self.pan_set = np.array(pan_set, dtype=np.float32) / 1023.
        self.ms_set = np.array(ms_set, dtype=np.float32) / 1023.
        self.lms_set = np.array(lms_set, dtype=np.float32) / 1023.

    def __getitem__(self, index):
        pan = self.pan_set[index, :, :, :]
        ms = self.ms_set[index, :, :, :]
        lms = self.lms_set[index, :, :, :]
        return pan, lms, ms

    def __len__(self):
        return self.pan_set.shape[0]


if __name__ == "__main__":
    validation_data_name = '../../disk3/xmm/dataset/testing/test_gf2_multiExm2.h5'  # your data path
    validation_data = h5py.File(validation_data_name, 'r')
    validation_dataset = new_dataset(validation_data)
    del validation_data
    data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)

    plt.figure(1)
    gauss_kernel_size = 5
    sigma = 10
    transform1 = transforms.GaussianBlur(gauss_kernel_size, sigma)
    for index, item in enumerate(data_loader):
        b, c, h, w = item[0].shape
        pan = item[1]
        pan_repeat = np.repeat(pan, c, 1)
        ms = item[0]
        lms = item[2]
        seed = random.randint(0, w - 20)
        pan_pixel = []
        ms_pixel = []
        pan_diff_x_p = []
        ms_diff_x_p = []
        pan_patch_pixel = []
        ms_patch_pixel = []

        pan_r = F.pad(pan, (0, 1, 0, 0))[:, :, :, 1:]
        pan_l = F.pad(pan, (1, 0, 0, 0))[:, :, :, :w]
        pan_t = F.pad(pan, (0, 0, 1, 0))[:, :, :h, :]
        pan_b = F.pad(pan, (0, 0, 0, 1))[:, :, 1:, :]
        pan_diff_x = pan_r - pan_l
        pan_diff_y = pan_t - pan_b

        pan_repeat_r = F.pad(pan_repeat, (0, 1, 0, 0))[:, :, :, 1:]
        pan_repeat_l = F.pad(pan_repeat, (1, 0, 0, 0))[:, :, :, :w]
        pan_repeat_t = F.pad(pan_repeat, (0, 0, 1, 0))[:, :, :h, :]
        pan_repeat_b = F.pad(pan_repeat, (0, 0, 0, 1))[:, :, 1:, :]
        pan_repeat_diff_x = pan_repeat_r - pan_repeat_l
        pan_repeat_diff_y = pan_repeat_t - pan_repeat_b

        ms_r = F.pad(ms, (0, 1, 0, 0))[:, :, :, 1:]
        ms_l = F.pad(ms, (1, 0, 0, 0))[:, :, :, :w]
        ms_t = F.pad(ms, (0, 0, 1, 0))[:, :, :h, :]
        ms_b = F.pad(ms, (0, 0, 0, 1))[:, :, 1:, :]
        ms_diff_x = ms_r - ms_l
        ms_diff_y = ms_t - ms_b

        # ms_f = F.pad(ms, (0, 0, 0, 0, 1, 0))[:, :c, :, :]
        ms_ba = F.pad(ms, (0, 0, 0, 0, 0, 1))[:, 1:, :, :]
        ms_diff_spc = ms - ms_ba
        # ms_diff_spc = transform1(ms_diff_spc)

        # lms_f = F.pad(lms, (0, 0, 0, 0, 1, 0))[:, :c, :, :]
        lms_ba = F.pad(lms, (0, 0, 0, 0, 0, 1))[:, 1:, :, :]
        lms_diff_spc = lms - lms_ba

        # pan_diff_x_pixel = []
        # pan_diff_y_pixel = []
        # ms_diff_x_pixel = []
        # ms_diff_y_pixel = []
        # for i in range(h):
        #     for j in range(w):
        #         pan_pixel.append(item[1][0][0][i][j])
        #         pan_diff_x_p.append(pan_diff_x[0][0][i][j])
        #         if i >= seed and i < seed + 20 and j >= seed and j < seed + 20:
        #             pan_patch_pixel.append(item[1][0][0][i][j])
        #             pan_diff_x_pixel.append(pan_diff_x[0][0][i][j])
        #             pan_diff_y_pixel.append(pan_diff_y[0][0][i][j])
        #
        # for j in range(h):
        #     for k in range(w):
        #         ms_band = []
        #         ms_diff_x_b = []
        #         ms_patch_band = []
        #         ms_diff_x_band = []
        #         ms_diff_y_band = []
        #         for i in range(c):
        #             ms_band.append(item[0][0][i][j][k])
        #             ms_diff_x_b.append(ms_diff_x[0][i][j][k])
        #         ms_pixel.append(ms_band)
        #         ms_diff_x_p.append(ms_diff_x_b)
        #         if j >= seed and j < seed + 20 and k >= seed and k < seed + 20:
        #             for i in range(c):
        #                 ms_patch_band.append(item[0][0][i][j][k])
        #                 ms_diff_x_band.append(ms_diff_x[0][i][j][k])
        #                 ms_diff_y_band.append(ms_diff_y[0][i][j][k])
        #             ms_patch_pixel.append(ms_patch_band)
        #             ms_diff_x_pixel.append(ms_diff_x_band)
        #             ms_diff_y_pixel.append(ms_diff_y_band)
        # ms_pixel = np.array(ms_pixel)
        # ms_diff_x_p = np.array(ms_diff_x_p)
        # theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(ms_pixel.T, ms_pixel)), ms_pixel.T), pan_pixel)
        # theta_dif_x = np.matmul(np.matmul(np.linalg.inv(np.matmul(ms_diff_x_p.T, ms_diff_x_p)), ms_diff_x_p.T),
        #                         pan_diff_x_p)
        # ms_sin_pixel = np.matmul(ms_patch_pixel, theta)
        # ms_diff_x = np.matmul(ms_diff_x_pixel, theta_dif_x)
        # ms_diff_y = np.matmul(ms_diff_y_pixel, theta)


        lms_np = lms[0].permute(1, 2, 0).cpu().data.numpy()
        ms_np = ms[0].permute(1, 2, 0).cpu().data.numpy()
        pan_np = pan[0].permute(1, 2, 0).cpu().data.numpy()
        ms_x_np = ms_diff_x[0].permute(1, 2, 0).cpu().data.numpy()
        ms_y_np = ms_diff_y[0].permute(1, 2, 0).cpu().data.numpy()
        pan_x_np = pan_diff_x[0].permute(1, 2, 0).cpu().data.numpy()
        pan_y_np = pan_diff_y[0].permute(1, 2, 0).cpu().data.numpy()
        ms_z_np = ms_diff_spc[0].permute(1, 2, 0).cpu().data.numpy()
        lms_z_np = lms_diff_spc[0].permute(1, 2, 0).cpu().data.numpy()

        spy.save_rgb(f'./vis_grad/gt_{index}.png', ms_np, [0, 1, 2], format='png')
        spy.save_rgb(f'./vis_grad/lms_{index}.png', lms_np, [0, 1, 2], format='png')
        spy.save_rgb(f'./vis_grad/pan_{index}.png', pan_np, format='png')
        spy.save_rgb(f'./vis_grad/gt_x_{index}.png', ms_x_np, [0, 1, 2], format='png')
        spy.save_rgb(f'./vis_grad/gt_y_{index}.png', ms_y_np, [0, 1, 2], format='png')
        spy.save_rgb(f'./vis_grad/pan_x_{index}.png', pan_x_np, format='png')
        spy.save_rgb(f'./vis_grad/pan_y_{index}.png', pan_y_np, format='png')
        spy.save_rgb(f'./vis_grad/gt_z_{index}.png', ms_z_np, [0, 1, 2], format='png')
        spy.save_rgb(f'./vis_grad/lms_z_{index}.png', lms_z_np, [0, 1, 2], format='png')



        # item[0]=item[0].permute(0,2,3,1)
        # item[1] = item[1].permute(0, 2, 3, 1)
        # item[2] = item[2].permute(0, 2, 3, 1)
        # item[3] = item[3].permute(0, 2, 3, 1)
        # print(item[0].size())
        # print(item[1].size())
        # print(item[2].size())
        # print(item[3].size())
        # view1=spy.imshow(data=item[0][0,:,:,:].numpy(),bands=(0,1,2),title='gt')
        # view2=spy.imshow(data=item[1][0,:,:,:].numpy(),title='pan')
        # view3=spy.imshow(data=item[2][0,:,:,:].numpy(),bands=(0,1,2),title='lms')
        # view4=spy.imshow(data=item[3][0,:,:,:].numpy(),bands=(0,1,2),title='ms')

        # plt.subplot(2, 2, 1)
        # plt.imshow(item[0][0,:,:,:])
        # plt.title('ground truth')
        # plt.subplot(2, 2, 2)
        # plt.imshow(item[1][0,:,:,:])
        # plt.title('pan image')
        # plt.subplot(2, 2, 3)
        # plt.imshow(item[2][0,:,:,:])
        # plt.title('lms image')
        # plt.subplot(2, 2, 4)
        # plt.imshow(item[3][0,:,:,:])
        # plt.title('ms image')
        # plt.show()
        if index == 3: break
