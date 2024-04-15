import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import spectral as spy

from thop import profile

dtype = torch.cuda.FloatTensor
MAEloss = torch.nn.L1Loss(reduction='mean').type(dtype)

def kaiming_init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0.0)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activate="relu",
                 bn=None, pad_model=None):
        super(Conv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = self.kernel_size // 2
        self.activate = activate
        self.bn = bn
        self.pad_model = pad_model

        if self.bn == 'bn':
            self.batch = nn.BatchNorm2d(self.out_channels)
        elif self.bn == 'in':
            self.batch = nn.InstanceNorm2d(self.out_channels)
        else:
            self.batch = None

        if activate == "lrelu":
            self.act = nn.LeakyReLU(0.2, True)
        elif activate == "tanh":
            self.act = nn.Tanh()
        elif activate == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activate == 'relu':
            self.act = nn.ReLU(True)
        elif activate == 'prelu':
            self.act = nn.PReLU(self.out_channels, init=0.5)
        elif activate == 'logsigmoid':
            self.act = nn.LogSigmoid()
        else:
            self.act = None

        if self.pad_model == 'reflection':
            self.padding = nn.Sequential(nn.ReflectionPad2d(self.padding))
            self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        else:
            self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)

        layers = filter(lambda x: x is not None, [self.conv, self.batch, self.act])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.pad_model is not None:
            x = self.padding(x)

        x = self.layers(x)

        return x


class smooth_gradient(nn.Module):
    def __init__(self, number=1):
        super(smooth_gradient, self).__init__()

        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        self.number = number

    # x:ill, y:ref
    def forward(self, x, y):
        d_X = []
        ave_d_X = []
        d_Y = []
        ave_d_Y = []
        self.smooth_kernel_x = self.smooth_kernel_x.to(x.device)
        self.smooth_kernel_y = self.smooth_kernel_y.to(x.device)
        for i in range(self.number):
            d_x_x = torch.abs(F.conv2d(torch.unsqueeze(x[:, i, :, :], 1), self.smooth_kernel_x))
            d_X.append(d_x_x)
            d_y_x = torch.abs(F.conv2d(torch.unsqueeze(y[:, i, :, :], 1), self.smooth_kernel_x))
            ave_d_X.append(F.avg_pool2d(d_y_x, kernel_size=3, stride=1, padding=1))

            d_x_y = torch.abs(F.conv2d(torch.unsqueeze(x[:, i, :, :], 1), self.smooth_kernel_y))
            d_Y.append(d_x_y)
            d_y_y = torch.abs(F.conv2d(torch.unsqueeze(y[:, i, :, :], 1), self.smooth_kernel_y))
            ave_d_Y.append(F.avg_pool2d(d_y_y, kernel_size=3, stride=1, padding=1))
        d_x = torch.cat(d_X, 1)
        ave_d_x = torch.cat(ave_d_X, 1)
        d_y = torch.cat(d_Y, 1)
        ave_d_y = torch.cat(ave_d_Y, 1)

        smooth_out = d_x * torch.exp(-10 * ave_d_x) + d_y * torch.exp(-10 * ave_d_y)

        return smooth_out


class Residual_Block(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size=3, stride=1, padding=1, activate='relu', bn=False, scale=1, pad_model=None):
        super(Residual_Block, self).__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = self.kernel_size // 2
        self.activate = activate
        self.bn = bn
        self.scale = scale
        self.pad_model = pad_model

        self.conv1 = Conv(self.in_channels, self.mid_channels, self.kernel_size, self.stride, self.padding,
                          activate='relu')
        self.conv2 = Conv(self.mid_channels, self.mid_channels, self.kernel_size, self.stride, self.padding,
                          activate='relu')
        self.conv3 = Conv(self.mid_channels, self.in_channels, self.kernel_size, self.stride, self.padding,
                          self.activate, self.bn, self.pad_model)

        layers = filter(lambda x: x is not None, [self.conv1, self.conv2, self.conv3])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        skip = x
        x = self.layers(x)
        x = x * self.scale
        x = torch.add(x, skip)

        return x

class Conv_up(nn.Module):
    def __init__(self, c_in, mid_c):
        super(Conv_up, self).__init__()

        body = [nn.Conv2d(in_channels=c_in, out_channels=mid_c, kernel_size=3, padding=3 // 2), nn.ReLU(),
                ]
        self.body = nn.Sequential(*body)
        modules_tail = [
            nn.Upsample(scale_factor=4),
            nn.Conv2d(mid_c, c_in, 3, padding=(3 // 2), bias=True),
            nn.Conv2d(c_in, c_in, 3, padding=(3 // 2), bias=True)
        ]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, input):
        out = self.body(input)
        out = self.tail(out)

        return out

class Conv_down(nn.Module):
    def __init__(self, c_in, mid_c):
        super(Conv_down, self).__init__()

        body = [nn.Conv2d(in_channels=c_in, out_channels=mid_c, kernel_size=3, padding=3 // 2), nn.ReLU(),
                ]
        self.body = nn.Sequential(*body)
        modules_tail = [
            nn.MaxPool2d(2),
            nn.Conv2d(mid_c, c_in, 3, padding=(3 // 2), bias=True),
            nn.Conv2d(c_in, c_in, 3, padding=(3 // 2), bias=True)
        ]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, input):
        out = self.body(input)
        out = self.tail(out)

        return out

class unet(nn.Module):
    def __init__(self, ms_channels):
        super(unet, self).__init__()

        self.ms_channels = ms_channels

        self.down1 = nn.Sequential(
            Conv(self.ms_channels + 1, 64),
            Residual_Block(64)
        )

        self.down2 = nn.Sequential(
            Conv(64, 128, kernel_size=4, stride=2, padding=1),
            Residual_Block(128)
        )

        self.layer = nn.Sequential(
            Conv(128, 256, kernel_size=4, stride=2, padding=1),
            Residual_Block(256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.up2 = nn.Sequential(
            Conv(256, 128),
            Residual_Block(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.up1 = nn.Sequential(
            Conv(128, 64),
            Conv(64, self.ms_channels)
        )

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        layer = self.layer(down2)
        up2 = self.up2(torch.cat([layer, down2], 1))
        out = self.up1(torch.cat([up2, down1], 1))

        return out

class Retinex(nn.Module):
    def __init__(self, ms_channels, iter_nums, out_channels=64):
        super(Retinex, self).__init__()

        self.ms_channels = ms_channels
        self.out_channels = out_channels
        self.iter_nums = iter_nums

        self.hypernet = nn.ModuleList()

        for i in range(self.iter_nums):
            self.hypernet.append(
                nn.Sequential(
                    Conv(self.ms_channels, self.out_channels, kernel_size=1, padding=0, activate=None),
                    Conv(self.out_channels, self.out_channels, activate=None),
                    Conv(self.out_channels, 4, kernel_size=1, padding=0, activate=None),
                    nn.Softplus()
                )
            )

        self.p_net = nn.Sequential(
            Conv(self.ms_channels * 2, self.out_channels, activate='prelu'),
            Conv(self.out_channels, self.out_channels, activate='prelu'),
            Conv(self.out_channels, self.ms_channels, activate='prelu')
        )

        self.q_net = nn.Sequential(
            Conv(self.ms_channels * 2, self.out_channels, activate='prelu'),
            Conv(self.out_channels, self.out_channels, activate='prelu'),
            Conv(self.out_channels, self.ms_channels, activate='prelu')
        )

        # self.s_net = nn.Sequential(
        #     Residual_Block(self.ms_channels, self.out_channels),
        #     Residual_Block(self.ms_channels, self.out_channels, activate=None),
        #     nn.LogSigmoid()
        #     # Conv(self.out_channels, self.ms_channels, activate='logsigmoid')
        # )
        #
        # self.r_net = nn.Sequential(
        #     Residual_Block(self.ms_channels, self.out_channels),
        #     Residual_Block(self.ms_channels, self.out_channels, activate=None),
        #     nn.LogSigmoid()
        #     # Conv(self.out_channels, self.ms_channels, activate='logsigmoid')
        # )

        kaiming_init_weights(self.p_net, self.q_net, self.hypernet)

    def forward(self, l_log, r_t, s_t, p_t, q_t):
        for i in range(self.iter_nums):
            hyperparam = self.hypernet[i](l_log)
            eta1 = hyperparam[:, 0, :, :].unsqueeze(1)
            eta2 = hyperparam[:, 1, :, :].unsqueeze(1)
            # eta3 = hyperparam[:, 2, :, :].unsqueeze(1)
            # eta4 = hyperparam[:, 3, :, :].unsqueeze(1)
            alpha = hyperparam[:, 2, :, :].unsqueeze(1)
            beta = hyperparam[:, 3, :, :].unsqueeze(1)

            # p_t = self.p_net(p_t - eta1 * alpha * (p_t - r_t))
            # q_t = self.q_net(q_t - eta2 * beta * (q_t - s_t))
            p_t = p_t + eta1 * self.p_net(torch.cat([p_t, r_t], 1))
            q_t = q_t + eta2 * self.q_net(torch.cat([q_t, s_t], 1))

            # r_t = r_t - eta3 * (r_t + s_t - l_log + alpha * (r_t - p_t))
            # s_t = s_t - eta4 * (r_t + s_t - l_log + beta * (s_t - q_t))
            r_t = torch.div(l_log + alpha * p_t - s_t, alpha + 1)
            s_t = torch.div(l_log + beta * q_t - r_t, beta + 1)
        # s_t = F.relu(self.s_net(s_t - eta1 * (r_t + s_t - l_log)) + (s_t - eta1 * (r_t + s_t - l_log)))
        # r_t = F.relu(self.r_net(r_t - eta2 * (r_t + s_t - l_log)) + (r_t - eta2 * (r_t + s_t - l_log)))
        #     s_t = self.s_net(s_t - eta1 / alpha * (r_t + s_t - l_log))
        #     r_t = self.r_net(r_t - eta2 / beta * (r_t + s_t - l_log))

        return r_t, s_t, p_t, q_t


class reconstruction(nn.Module):
    def __init__(self, ms_channels, pan_channel, iter_nums, out_channels=64):
        super(reconstruction, self).__init__()

        self.ms_channels = ms_channels
        self.pan_channel = pan_channel
        self.out_channels = out_channels
        self.iter_nums = iter_nums

        self.hypernet = nn.ModuleList()

        for i in range(self.iter_nums):
            self.hypernet.append(
                nn.Sequential(
                    Conv(self.ms_channels + self.pan_channel, self.out_channels, kernel_size=1, padding=0, activate=None, bn=False),
                    Conv(self.out_channels, self.out_channels, activate=None, bn=False),
                    Conv(self.out_channels, 5, kernel_size=1, padding=0, activate=None, bn=False),
                    nn.Softplus()
                )
            )

        self.r1 = nn.Sequential(
            Conv(self.ms_channels, self.out_channels),
            Conv(self.out_channels, self.ms_channels)
        )

        self.r1_t = nn.Sequential(
            Conv(self.ms_channels, self.out_channels),
            Conv(self.out_channels, self.ms_channels)
        )

        self.a = nn.Sequential(
            Conv(self.pan_channel, self.out_channels),
            Conv(self.out_channels, self.pan_channel)
        )

        self.a_t = nn.Sequential(
            Conv(self.pan_channel, self.out_channels),
            Conv(self.out_channels, self.ms_channels)
        )

        # self.s_net = nn.Sequential(
        #     Residual_Block(self.ms_channels, self.out_channels),
        #     Residual_Block(self.ms_channels, self.out_channels, activate=None),
        #     nn.Sigmoid()
        #     # Conv(self.out_channels, self.ms_channels, activate='sigmoid')
        # )
        self.s_net = nn.Sequential(
            Conv(self.ms_channels, self.out_channels),
            Residual_Block(self.out_channels, self.out_channels),
            Residual_Block(self.out_channels, self.out_channels),
            Conv(self.out_channels, self.ms_channels, activate='sigmoid')
        )
        # self.s_net = unet(self.ms_channels)

        kaiming_init_weights(self.r1, self.r1_t, self.a, self.a_t, self.s_net, self.hypernet)

    def forward(self, usms, pan, r_t, s_lr_t, b_t, z_t):
        r_t_T = torch.transpose(r_t, 2, 3)
        s_hr_t = self.r1_t(s_lr_t)

        for i in range(self.iter_nums):
            hyperparam2 = self.hypernet[i](torch.cat([usms, pan], 1))
            eta1 = hyperparam2[:, 0, :, :].unsqueeze(1)
            # eta2 = hyperparam2[:, 1, :, :].unsqueeze(1)
            # eta3 = hyperparam2[:, 2, :, :].unsqueeze(1)
            gamma1 = hyperparam2[:, 1, :, :].unsqueeze(1)
            gamma2 = hyperparam2[:, 2, :, :].unsqueeze(1)
            alpha = hyperparam2[:, 3, :, :].unsqueeze(1)
            beta = hyperparam2[:, 4, :, :].unsqueeze(1)

            z_numerator = beta * self.a_t(pan) + gamma1 * r_t * b_t
            z_denominator = self.a_t(self.a(beta)) + gamma1
            z_t = torch.div(z_numerator, z_denominator)

            b_numerator = alpha * self.r1_t(r_t_T) * usms + gamma1 * r_t_T * z_t + gamma2 * s_hr_t
            b_denominator = alpha * self.r1_t(r_t_T) * self.r1(r_t) + gamma1 * r_t_T * r_t + gamma2
            b_t = torch.div(b_numerator, b_denominator)

            # z_t = z_t * (1 - eta1 * z_denominator) + eta1 * z_numerator
            # b_t = b_t * (1 - eta2 * b_denominator) + eta2 * b_numerator

            # z_t = z_t - eta1 * (beta * self.a_t(self.a(z_t) - pan)
            #                     + eplison1 * (z_t - r_t * b_t))
            #
            # b_t = b_t - eta2 * (alpha * (self.r1_t(r_t_T * (self.r1(r_t * b_t) - usms)))
            #                     + eplison1 * r_t_T * (r_t * b_t - z_t)
            #                     + eplison2 * (b_t - s_hr_t))

            # s_hr_t = s_hr_t + eta1 * self.s_net(torch.cat([s_hr_t, b_t], 1))
            s_hr_t = self.s_net(s_hr_t - eta1 * gamma2 * (s_hr_t - b_t))

        s_lr_t = torch.log(self.r1(s_hr_t) + 1e-5)

        # s_l_res = self.r1_t(r_t_T) * (self.r1(r_t) * s_t - usms)
        # s_p_res = self.r2_t(r_t_T) * (self.r2(r_t * s_t) - pan)
        # # s_t = F.relu(self.s_net(s_t - eta3 * (alpha * s_l_res + beta * s_p_res)) + (s_t - eta3 * (alpha * s_l_res + beta * s_p_res)))
        # s_t = self.s_net(s_t - eta3 * (alpha * s_l_res + beta * s_p_res))
        #
        # r_l_res = self.s1_t(s_t_T) * (self.s1(r_t) * s_t - usms)
        # r_p_res = self.s2_t(s_t_T) * (self.s2(r_t * s_t) - pan)
        # # r_t = F.relu(self.r_net(r_t - eta4 * (alpha * r_l_res + beta * r_p_res)) + (r_t - eta4 * (alpha * r_l_res + beta * r_p_res)))
        # r_t = self.r_net(r_t - eta4 * (alpha * r_l_res + beta * r_p_res))

        return s_hr_t, s_lr_t, b_t, z_t


class deepnet(nn.Module):
    def __init__(self, ms_channels, pan_channel, iter_nums, inner_nums1, inner_nums2, out_channels=64):
        super(deepnet, self).__init__()

        self.ms_channels = ms_channels
        self.pan_channel = pan_channel
        self.out_channels = out_channels

        self.iter_nums = iter_nums
        self.inner_nums1 = inner_nums1
        self.inner_nums2 = inner_nums2

        # self.hypernet1 = nn.ModuleList()

        # self.hypernet2 = nn.ModuleList()

        # self.retinex = nn.ModuleList()
        #
        # self.reconstruction = nn.ModuleList()

        # for i in range(self.iter_nums):
            # self.hypernet1.append(
            #     nn.Sequential(
            #         Conv(self.ms_channels, self.out_channels, kernel_size=1, padding=0, activate=None, bn=False),
            #         # Conv(16, 16, activate=None, bn=False),
            #         Conv(self.out_channels, 6, kernel_size=1, padding=0, activate=None, bn=False),
            #         nn.Softplus()
            #     )
            # )

            # self.hypernet2.append(
            #     nn.Sequential(
            #         Conv(self.ms_channels + self.pan_channel, self.out_channels, kernel_size=1, padding=0,
            #              activate=None, bn=False),
            #         # Conv(16, 16, activate=None, bn=False),
            #         Conv(self.out_channels, 7, kernel_size=1, padding=0, activate=None, bn=False),
            #         nn.Softplus()
            #     )
            # )

            # self.retinex.append(Retinex(self.ms_channels, self.out_channels))
            #
            # self.reconstruction.append(reconstruction(self.ms_channels, self.pan_channel, self.out_channels))

        self.r_0 = nn.Sequential(
            Conv(self.ms_channels, self.out_channels, activate='prelu'),
            Conv(self.out_channels, self.ms_channels, activate='logsigmoid')
        )

        self.s_lr_0 = nn.Sequential(
            Conv(self.ms_channels, self.out_channels, activate='prelu'),
            Conv(self.out_channels, self.ms_channels, activate='logsigmoid')
        )

        self.p_0 = nn.Sequential(
            Conv(self.ms_channels, self.out_channels, activate='prelu'),
            Conv(self.out_channels, self.ms_channels, activate='prelu')
        )

        self.q_0 = nn.Sequential(
            Conv(self.ms_channels, self.out_channels, activate='prelu'),
            Conv(self.out_channels, self.ms_channels, activate='prelu')
        )

        # self.s_hr_0 = nn.Sequential(
        #     Conv(self.ms_channels + self.pan_channel, self.out_channels),
        #     Conv(self.out_channels, self.ms_channels, activate='prelu')
        # )

        self.b_0 = nn.Sequential(
            Conv(self.ms_channels + self.pan_channel, self.out_channels),
            Conv(self.out_channels, self.ms_channels, activate=None)
        )

        self.z_0 = nn.Sequential(
            Conv(self.ms_channels + self.pan_channel, self.out_channels),
            Conv(self.out_channels, self.ms_channels, activate=None)
        )

        self.smooth = smooth_gradient(number=self.ms_channels)

        self.retinex = Retinex(self.ms_channels, self.inner_nums1, self.out_channels)

        self.reconstruction = reconstruction(self.ms_channels, self.pan_channel, self.inner_nums2, self.out_channels)

        kaiming_init_weights(self.p_0, self.q_0, self.r_0, self.s_lr_0, self.b_0, self.z_0)

    def forward(self, usms, pan):
        usms_log = (usms + 1 + 2 / 2047) / (2 + 2 / 2047)
        l_log = torch.log(usms_log)
        r_t = self.r_0(l_log)
        s_lr_t = self.s_lr_0(l_log)
        p_t = self.p_0(l_log)
        q_t = self.q_0(l_log)
        # s_hr_t = self.s_hr_0(torch.cat([usms, pan], 1))
        b_t = self.b_0(torch.cat([usms, pan], 1))
        z_t = self.z_0(torch.cat([usms, pan], 1))

        for i in range(self.iter_nums):
            # hyperparam1 = self.hypernet1[i](l_log)
            # eta1 = hyperparam1[:, 0, :, :].unsqueeze(1)
            # eta2 = hyperparam1[:, 1, :, :].unsqueeze(1)
            # eta3 = hyperparam1[:, 2, :, :].unsqueeze(1)
            # eta4 = hyperparam1[:, 3, :, :].unsqueeze(1)
            # alpha = hyperparam1[:, 4, :, :].unsqueeze(1)
            # beta = hyperparam1[:, 5, :, :].unsqueeze(1)

            r_t, s_log, p_t, q_t = self.retinex(l_log, r_t, s_lr_t, p_t, q_t)

            r_exp = torch.exp(r_t)
            s_exp = torch.exp(s_log)

            # hyperparam2 = self.hypernet2[i](torch.cat([usms, pan], 1))
            # eta1 = hyperparam2[:, 0, :, :].unsqueeze(1)
            # eta2 = hyperparam2[:, 1, :, :].unsqueeze(1)
            # eta3 = hyperparam2[:, 2, :, :].unsqueeze(1)
            # eplison1 = hyperparam2[:, 3, :, :].unsqueeze(1)
            # eplison2 = hyperparam2[:, 4, :, :].unsqueeze(1)
            # alpha = hyperparam2[:, 5, :, :].unsqueeze(1)
            # beta = hyperparam2[:, 6, :, :].unsqueeze(1)

            s_hr_t, s_lr_t, b_t, z_t = self.reconstruction(usms, pan, r_exp, s_exp, b_t, z_t)

        smooth = self.smooth(s_hr_t, r_exp)

        usms_out = s_exp * r_exp

        out = s_hr_t * r_exp

        return usms_out, out, smooth, s_lr_t, s_hr_t, s_log, s_lr_t
        # return usms_out, out, smooth, s_lr_t, s_hr_t, s_log, s_lr_t, r_exp

if __name__ == '__main__':
    net = 'cdun'
    ms_channels = 4
    pan_channel = 1
    iter_nums = 1
    iter_nums1 = 3  # inner_iter1
    iter_nums2 = 3  # inner_iter2
    alpha = 0.1  # usms_loss系数
    beta = 0.01  # smooth系数
    qb_channels = 4
    wv_channels = 8
    batch_size = 4
    ms_reduce = 64
    ms_full = 128
    lms_reduce = 256
    lms_full = 512

    model = deepnet(ms_channels=qb_channels, pan_channel=pan_channel,
                  iter_nums=iter_nums, inner_nums1=iter_nums1, inner_nums2=iter_nums2)
    input = (torch.randn(batch_size, 4, lms_reduce, lms_reduce), torch.randn(batch_size, 1, 256, 256))
    flops, params = profile(model, (input[0], input[1],))
    print(f'{net} QB_reduce FLOPS={flops / 1e9:.4f}G params={params / 1e6:.4f}M')
    model = deepnet(ms_channels=qb_channels, pan_channel=pan_channel,
                  iter_nums=iter_nums, inner_nums1=iter_nums1, inner_nums2=iter_nums2)
    input = (torch.randn(batch_size, 4, lms_full, lms_full), torch.randn(batch_size, 1, 512, 512))
    flops, params = profile(model, (input[0], input[1],))
    print(f'{net} QB_full FLOPS={flops / 1e9:.4f}G params={params / 1e6:.4f}M')

    model = deepnet(ms_channels=wv_channels, pan_channel=pan_channel,
                  iter_nums=iter_nums, inner_nums1=iter_nums1, inner_nums2=iter_nums2)
    input = (torch.randn(batch_size, 8, lms_reduce, lms_reduce), torch.randn(batch_size, 1, 256, 256))
    flops, params = profile(model, (input[0], input[1],))
    print(f'{net} wv3_reduce FLOPS={flops / 1e9:.4f}G params={params / 1e6:.4f}M')
    model = deepnet(ms_channels=wv_channels, pan_channel=pan_channel,
                  iter_nums=iter_nums, inner_nums1=iter_nums1, inner_nums2=iter_nums2)
    input = (torch.randn(batch_size, 8, lms_full, lms_full), torch.randn(batch_size, 1, 512, 512))
    flops, params = profile(model, (input[0], input[1],))
    print(f'{net} wv3_full FLOPS={flops / 1e9:.4f}G params={params / 1e6:.4f}M')
