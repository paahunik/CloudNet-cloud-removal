import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
import glob
import matplotlib.pyplot as plt
import random
import numpy as np
from torch import optim
from torch.autograd import Variable
import os
import torchvision
import socket
from datetime import datetime
import gdal
from skimage.transform import resize
import time
from math import log10, sqrt
from os.path import exists
from sewar.full_ref import psnr, psnrb, rase, mse, ssim, uqi, ergas


class Sat_Dataset(Dataset):
    def __init__(self, start, stop):
        dirName = '/s/' + socket.gethostname() + '/a/nobackup/galileo/'
        file1 = open(dirName + 'InputDatasetInfo.txt', 'r')
        lines = file1.readlines()
        file_list = []
        for line in lines:
            file_list.append(line)

        file_list.sort()
        file_list = file_list[start:stop]

        self.clean_images = []
        self.cloud_masks = []
        self.geohashes = []
        if (socket.gethostname() in ["lattice-211", "lattice-212", "lattice-213", "lattice-214", "lattice-215",
                                     "lattice-216", "lattice-217", "lattice-218", "lattice-219", "lattice-220",
                                     "lattice-221", "lattice-222", "lattice-223", "lattice-224", "lattice-225"]):
            for file2 in file_list:
                geohash, _, clean_image, cloud_mask, _, _ = file2.split(",")

                self.clean_images.append(
                    clean_image.replace(clean_image[:34], "/s/" + socket.gethostname() + "/a/nobackup/galileo/sarmst/"))
                self.cloud_masks.append(
                    cloud_mask.replace(cloud_mask[:34], "/s/" + socket.gethostname() + "/a/nobackup/galileo/sarmst/"))
                self.geohashes.append(geohash)
        else:
            for file2 in file_list:
                geohash, _, clean_image, cloud_mask, _, _ = file2.split(",")
                self.clean_images.append(clean_image)
                self.cloud_masks.append(cloud_mask)
                self.geohashes.append(geohash)

        #         self.labels = [l.split("/")[-1] for l in glob.glob(root_dir + "/*")]
        #         self.files = glob.glob(root_dir + "/*/*.npy")
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.clean_images)

    def getGeohashes(self):
        return self.geohashes

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cloudless = np.array(gdal.Open(self.clean_images[idx]).ReadAsArray()).astype(np.float32)
        cloudy = np.array(gdal.Open(self.cloud_masks[idx]).ReadAsArray()).astype(np.float32)
        cloudy = cloudy.transpose([0,2,1])
        cloudless = resize(cloudless, (11, 164, 123), preserve_range=True)
        cloudy = resize(cloudy, (11, 164, 123), preserve_range=True)
        M = (((2800 <= cloudy[-1, :, :]) & (cloudy[-1, :, :] < 2804)) == 0).astype(np.float32)
        cloudy = M.reshape((1, M.shape[0], M.shape[1])).repeat(11, axis=0)
        cloudless = cloudless.clip(0, 65535) / 65535
        cloudy_temp = cloudy.copy()
        cloudy[cloudy_temp == 1] = cloudless[cloudy_temp == 1]

        cloudy = torch.from_numpy(cloudy)
        cloudless = torch.from_numpy(cloudless)
        M_copy = M.copy()
        M[M_copy == 1] = 0
        M[M_copy == 0] = 1
        M = torch.from_numpy(M)
        return cloudy, cloudless, M.reshape((1, M.shape[0], M.shape[1]))


class Test_Dataset(Dataset):
    #     def __init__(self, root_dir= socket.gethostname() + "_dataset/train"):
    def __init__(self, isTrain=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        dirName = '/s/' + socket.gethostname() + '/a/nobackup/galileo/paahuni/cloud_removal/'
        if isTrain:
            file1 = open(dirName + 'InputDatasetInfo.txt', 'r')
            lines = file1.readlines()
            # random.Random(22).shuffle(lines)
        else:
            file1 = open(dirName + 'InputDatasetInfoValidation.txt', 'r')
            lines = file1.readlines()
            # random.Random(99).shuffle(lines)

        self.file_list = []
        self.geohashes=[]
        self.cloudC = []
        # Strips the newline character
        for line in lines:
            self.file_list.append(line)

        self.file_list.sort()

        self.clean_images = []
        self.cloud_masks = []
        if (socket.gethostname() in ["lattice-211", "lattice-212", "lattice-213", "lattice-214", "lattice-215",
                                     "lattice-216", "lattice-217", "lattice-218", "lattice-219", "lattice-220",
                                     "lattice-221", "lattice-222", "lattice-223", "lattice-224", "lattice-225"]):
            for file2 in self.file_list:
                geohash, cc, clean_image, cloud_mask, _, _ = file2.split(",")

                self.clean_images.append(
                    clean_image.replace(clean_image[:34], "/s/" + socket.gethostname() + "/a/nobackup/galileo/sarmst/"))
                self.cloud_masks.append(
                    cloud_mask.replace(cloud_mask[:34], "/s/" + socket.gethostname() + "/a/nobackup/galileo/sarmst/"))
                self.geohashes.append(geohash)
                self.cloudC.append(cc)
        else:
            for file2 in self.file_list:

                geohash, cc, clean_image, cloud_mask, _, _ = file2.split(",")

                self.clean_images.append(clean_image)
                self.cloud_masks.append(cloud_mask)
                self.geohashes.append(geohash)
                self.cloudC.append(cc)
        #         for file2 in self.file_list:
        #             _, _, clean_image, cloud_mask, _, _ = file2.split(",")
        #             self.clean_images.append(clean_image)
        #             self.cloud_masks.append(cloud_mask)

        #         self.labels = [l.split("/")[-1] for l in glob.glob(root_dir + "/*")]
        #         self.files = glob.glob(root_dir + "/*/*.npy")
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    #         self.init_dataloader = DataLoader(dataset=Init_Dataset())
    #         self.max_pixel = 0
    #         for _, img in self.init_dataloader:
    #             if (self.max_pixel < float(img.max())):
    #                 self.max_pixel = float(img.max())

    #         print("Max Pixel Value is " + str(self.max_pixel))

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cloudless = np.array(gdal.Open(self.clean_images[idx]).ReadAsArray()).astype(np.float32)
        cloudy = np.array(gdal.Open(self.cloud_masks[idx]).ReadAsArray()).astype(np.float32)
        cloudy = cloudy.transpose([0,2, 1])
        cloudless = resize(cloudless, (11, 164, 123), preserve_range=True)
        cloudy = resize(cloudy, (11, 164, 123), preserve_range=True)
        M = (((2800 <= cloudy[-1, :, :]) & (cloudy[-1, :, :] < 2804)) == 0).astype(np.float32)
        cloudy = M.reshape((1, M.shape[0], M.shape[1])).repeat(11, axis=0)
        cloudless = cloudless.clip(0, 65535) / 65535
        cloudy_temp = cloudy.copy()
        cloudy[cloudy_temp == 1] = cloudless[cloudy_temp == 1]

        cloudy = torch.from_numpy(cloudy)
        cloudless = torch.from_numpy(cloudless)
        M_copy = M.copy()
        M[M_copy == 1] = 0
        M[M_copy == 0] = 1
        M = torch.from_numpy(M)
        #         cloudy = torch.nn.functional.interpolate(cloudy.view(1, cloudy.shape[0], cloudy.shape[1], cloudy.shape[2]), 256)[0]
        #         cloudless = torch.nn.functional.interpolate(cloudless.view(1, cloudless.shape[0], cloudless.shape[1], cloudless.shape[2]), 256)[0]
        #         M = torch.nn.functional.interpolate(M.view(1, M.shape[0], M.shape[1], M.shape[2]), 256)[0]

        final_cloudy, final_cloudless = normalize2(cloudy[[3, 2, 1]], cloudless[[3, 2, 1]])
        return final_cloudy, final_cloudless, M.reshape((1, M.shape[0], M.shape[1])), self.file_list[idx],  self.geohashes[idx], self.cloudC[idx]


def normalize2(cloudy, cloudless):
    cloudless_min, cloudless_max = cloudless.min(), cloudless.max()
    return torch.clamp(((cloudy - cloudless_min) / (cloudless_max - cloudless_min)), min=0, max=1), (
                cloudless - cloudless_min) / (cloudless_max - cloudless_min)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class CBR(nn.Module):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=nn.ReLU(True), dropout=False):
        super().__init__()
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        if sample == 'down':
            self.c = nn.Conv2d(ch0, ch1, 4, 2, 1)
        else:
            self.c = nn.ConvTranspose2d(ch0, ch1, 4, 2, 1)
        if bn:
            self.batchnorm = nn.BatchNorm2d(ch1, affine=True)
        if dropout:
            self.Dropout = nn.Dropout()

    def forward(self, x):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h)
        if self.dropout:
            h = self.Dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h


###### Layer
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(Bottleneck, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x)
        return out


class irnn_layer(nn.Module):
    def __init__(self, in_channels):
        super(irnn_layer, self).__init__()
        self.left_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.right_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.up_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.down_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)

    def forward(self, x):
        _, _, H, W = x.shape
        top_left = x.clone()
        top_right = x.clone()
        top_up = x.clone()
        top_down = x.clone()
        top_left[:, :, :, 1:] = F.relu(self.left_weight(x)[:, :, :, :W - 1] + x[:, :, :, 1:], inplace=False)
        top_right[:, :, :, :-1] = F.relu(self.right_weight(x)[:, :, :, 1:] + x[:, :, :, :W - 1], inplace=False)
        top_up[:, :, 1:, :] = F.relu(self.up_weight(x)[:, :, :H - 1, :] + x[:, :, 1:, :], inplace=False)
        top_down[:, :, :-1, :] = F.relu(self.down_weight(x)[:, :, 1:, :] + x[:, :, :H - 1, :], inplace=False)
        return (top_up, top_right, top_down, top_left)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.out_channels = int(in_channels / 2)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels, 4, kernel_size=1, padding=0, stride=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out


class SAM(nn.Module):
    def __init__(self, in_channels, out_channels, attention=1):
        super(SAM, self).__init__()
        self.out_channels = out_channels
        self.irnn1 = irnn_layer(self.out_channels)
        self.irnn2 = irnn_layer(self.out_channels)
        self.conv_in = conv3x3(in_channels, self.out_channels)
        self.relu1 = nn.ReLU(True)

        self.conv1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.out_channels * 4, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(self.out_channels * 4, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        self.conv_out = conv1x1(self.out_channels, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv1(x)
        top_up, top_right, top_down, top_left = self.irnn1(out)

        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])
        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv2(out)
        top_up, top_right, top_down, top_left = self.irnn2(out)

        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])

        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv3(out)
        out = self.relu2(out)
        mask = torch.sigmoid(self.conv_out(out))
        return mask


###### Network
class SPANet(nn.Module):
    def __init__(self):
        super(SPANet, self).__init__()

        self.channels = 32

        self.conv_in = nn.Sequential(
            conv3x3(11, self.channels),
            nn.ReLU(True)
        )

        self.SAM1 = SAM(self.channels, self.channels, 1)
        self.res_block1 = Bottleneck(self.channels, self.channels)
        self.res_block2 = Bottleneck(self.channels, self.channels)
        self.res_block3 = Bottleneck(self.channels, self.channels)
        self.res_block4 = Bottleneck(self.channels, self.channels)
        self.res_block5 = Bottleneck(self.channels, self.channels)
        self.res_block6 = Bottleneck(self.channels, self.channels)
        self.res_block7 = Bottleneck(self.channels, self.channels)
        self.res_block8 = Bottleneck(self.channels, self.channels)
        self.res_block9 = Bottleneck(self.channels, self.channels)
        self.res_block10 = Bottleneck(self.channels, self.channels)
        self.res_block11 = Bottleneck(self.channels, self.channels)
        self.res_block12 = Bottleneck(self.channels, self.channels)
        self.res_block13 = Bottleneck(self.channels, self.channels)
        self.res_block14 = Bottleneck(self.channels, self.channels)
        self.res_block15 = Bottleneck(self.channels, self.channels)
        self.res_block16 = Bottleneck(self.channels, self.channels)
        self.res_block17 = Bottleneck(self.channels, self.channels)
        self.conv_out = nn.Sequential(
            conv3x3(self.channels, 11)
        )

    def forward(self, x):
        out = self.conv_in(x)
        out = F.relu(self.res_block1(out) + out)
        out = F.relu(self.res_block2(out) + out)
        out = F.relu(self.res_block3(out) + out)

        Attention1 = self.SAM1(out)
        out = F.relu(self.res_block4(out) * Attention1 + out)
        out = F.relu(self.res_block5(out) * Attention1 + out)
        out = F.relu(self.res_block6(out) * Attention1 + out)
        Attention2 = self.SAM1(out)
        out = F.relu(self.res_block7(out) * Attention2 + out)
        out = F.relu(self.res_block8(out) * Attention2 + out)
        out = F.relu(self.res_block9(out) * Attention2 + out)
        Attention3 = self.SAM1(out)
        out = F.relu(self.res_block10(out) * Attention3 + out)
        out = F.relu(self.res_block11(out) * Attention3 + out)
        out = F.relu(self.res_block12(out) * Attention3 + out)
        Attention4 = self.SAM1(out)
        out = F.relu(self.res_block13(out) * Attention4 + out)
        out = F.relu(self.res_block14(out) * Attention4 + out)
        out = F.relu(self.res_block15(out) * Attention4 + out)

        out = F.relu(self.res_block16(out) + out)
        out = F.relu(self.res_block17(out) + out)
        out = self.conv_out(out)

        return Attention4, out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.gen = nn.Sequential(OrderedDict([('gen', SPANet())]))

        self.gen.apply(weights_init)

    def forward(self, x):
        return self.gen(x)


class _Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch

        self.c0_0 = CBR(in_ch, 32, bn=False, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c0_1 = CBR(out_ch, 32, bn=False, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c1 = CBR(64, 128, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c2 = CBR(128, 256, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c3 = CBR(256, 512, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c4 = nn.Conv2d(512, 1, 3, 1, 1)

    def forward(self, x):
        x_0 = x[:, :self.in_ch]
        x_1 = x[:, self.in_ch:]
        h = torch.cat((self.c0_0(x_0), self.c0_1(x_1)), 1)
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)

        return h


class Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.dis = nn.Sequential(OrderedDict([('dis', _Discriminator(in_ch, out_ch))]))

        self.dis.apply(weights_init)

    def forward(self, x):
        return self.dis(x)


def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)


def PSNR(original, compressed):
    psnrs = []
    for i in range(original.shape[0]):
        mse = torch.mean((original[i] - compressed[i]) ** 2)
        if (mse == 0):  # MSE is zero means no noise is present in the signal .
            # Therefore PSNR have no importance.
            return 48
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        psnrs.append(float(psnr))
    return sum(psnrs) / len(psnrs)


def train(start, stop):
    # don;t forget to add num_workers back to dataloaders
    lattice_dict = {"lattice-211": "lattice-178",
                    "lattice-212": "lattice-178",
                    "lattice-213": "lattice-190",
                    "lattice-214": "lattice-191",
                    "lattice-215": "lattice-196",
                    "lattice-216": "lattice-196",
                    "lattice-217": "lattice-199",
                    "lattice-218": "lattice-199",
                    "lattice-219": "lattice-201",
                    "lattice-220": "lattice-202",
                    "lattice-221": "lattice-203",
                    "lattice-222": "lattice-203",
                    "lattice-223": "lattice-204",
                    "lattice-224": "lattice-209",
                    "lattice-225": "lattice-219",
                    }
    if (socket.gethostname() in ["lattice-211", "lattice-212", "lattice-213", "lattice-214", "lattice-215",
                                 "lattice-216", "lattice-217", "lattice-218", "lattice-219", "lattice-220",
                                 "lattice-221", "lattice-222", "lattice-223", "lattice-224", "lattice-225"]):
        name = lattice_dict[socket.gethostname()] + "_" + str(start) + "_" + str(stop)
    else:
        name = socket.gethostname() + "_" + str(start) + "_" + str(stop)
    batch_size = 10
    in_ch = 3
    out_ch = 3
    width = 164
    height = 123
    num_step_update_dis = 1
    lamb = 1000
    epochs = 300
    os.makedirs("SPAGAN", exist_ok=True)
    os.makedirs("SPAGAN/log" + "_" + name, exist_ok=True)
    os.makedirs("SPAGAN/checkpoint" + "_" + name, exist_ok=True)
    train_dataset = Sat_Dataset(start=start, stop=stop)
    train_size = len(train_dataset)
    print('Train Dataset Length:', train_size)
    #     , num_workers=5
    training_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                      num_workers=5)
    print("Total Epochs: " + str(epochs))

    gen = Generator()

    dis = Discriminator(in_ch, out_ch)

    # setup optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=0.00001)
    opt_dis = optim.Adam(dis.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=0.00001)

    #     opt_gen = optim.Adam(gen.parameters(), lr=0.001, weight_decay=0.00001)
    #     opt_dis = optim.Adam(dis.parameters(), lr=0.001, weight_decay=0.00001)

    real_a = torch.FloatTensor(batch_size, in_ch, width, height)
    real_b = torch.FloatTensor(batch_size, out_ch, width, height)
    real_test = torch.FloatTensor(batch_size, out_ch, width, height)
    M = torch.FloatTensor(batch_size, 1, width, height)

    criterionL1 = nn.L1Loss()
    criterionMSE = nn.MSELoss()
    criterionSoftplus = nn.Softplus()

    if torch.cuda.is_available():
        gen = gen.cuda()
        dis = dis.cuda()
        criterionL1 = criterionL1.cuda()
        criterionMSE = criterionMSE.cuda()
        criterionSoftplus = criterionSoftplus.cuda()
        real_a = real_a.cuda()
        real_b = real_b.cuda()
        M = M.cuda()

    real_a = Variable(real_a)
    real_b = Variable(real_b)
    real_test = Variable(real_test)

    #     is_cuda = True

    # main
    for epoch in range(1, epochs + 1):
        for iteration, batch in enumerate(training_data_loader, 1):

            real_a_cpu, real_b_cpu, M_cpu = batch[0], batch[1], batch[2]
            real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
            real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)
            M.data.resize_(M_cpu.size()).copy_(M_cpu)

            #             now = datetime.now()
            #             current_time = now.strftime("%H:%M")
            #             hours, mins = current_time.split(":")
            #             hours, mins = int(hours), int(mins)
            #             total_minutes = (hours * 60) + mins

            #             while (total_minutes <= ((10*60) + 1)) or (total_minutes >= ((22*60) - 1)):
            #                 if (torch.cuda.is_available() and is_cuda):
            #                     gen = gen.cpu()
            #                     dis = dis.cpu()
            #                     criterionL1 = criterionL1.cpu()
            #                     criterionMSE = criterionMSE.cpu()
            #                     criterionSoftplus = criterionSoftplus.cpu()
            #                     real_a = real_a.cpu()
            #                     real_b = real_b.cpu()
            #                     M = M.cpu()
            #                     is_cuda = False
            #                 time.sleep(1)
            #                 now = datetime.now()
            #                 current_time = now.strftime("%H:%M")
            #                 hours, mins = current_time.split(":")
            #                 hours, mins = int(hours), int(mins)
            #                 total_minutes = (hours * 60) + mins

            #             if (torch.cuda.is_available() and (is_cuda == False)):
            #                 gen = gen.cuda()
            #                 dis = dis.cuda()
            #                 criterionL1 = criterionL1.cuda()
            #                 criterionMSE = criterionMSE.cuda()
            #                 criterionSoftplus = criterionSoftplus.cuda()
            #                 real_a = real_a.cuda()
            #                 real_b = real_b.cuda()
            #                 M = M.cuda()
            #                 is_cuda = True

            att, fake_b = gen.forward(real_a)

            ################
            ### Update D ###
            ################

            opt_dis.zero_grad()

            # train with fake
            fake_ab = torch.cat((real_a, fake_b), 1)

            pred_fake = dis.forward(fake_ab.detach())
            batchsize, _, w, h = pred_fake.size()

            loss_d_fake = torch.sum(criterionSoftplus(pred_fake)) / batchsize / w / h

            # train with real
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = dis.forward(real_ab)
            loss_d_real = torch.sum(criterionSoftplus(-pred_real)) / batchsize / w / h

            # Combined loss
            loss_d = loss_d_fake + loss_d_real

            loss_d.backward()

            if epoch % num_step_update_dis == 0:
                opt_dis.step()

            ################
            ### Update G ###
            ################

            opt_gen.zero_grad()

            # First, G(A) should fake the discriminator
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = dis.forward(fake_ab)
            loss_g_gan = (torch.sum(criterionSoftplus(-pred_fake)) / batchsize / w / h)

            # Second, G(A) = B
            loss_g_l1 = criterionL1(fake_b, real_b) * lamb
            psnr_value = PSNR(fake_b, real_b)
            loss_g_att = criterionMSE(att, M) * 25
            loss_g = loss_g_gan + loss_g_l1 + loss_g_att

            loss_g.backward()

            opt_gen.step()

            print(
                "===> Epoch[{}]({:.2f}% Done): Gen_Loss: {:.4f}, Gen_SoftPlus_Loss: {:.4f}, Gen_L1_Loss: {:.4f}, Gen_Attn_Loss: {:.4f}, Dis_Loss: {:.4f},  Dis_Real_Loss: {:.4f}, Dis_Fake_Loss: {:.4f}, PSNR: {:.4f}".format(
                    epoch, ((iteration / len(training_data_loader)) * 100), loss_g.item(), loss_g_gan.item(),
                    loss_g_l1.item(), loss_g_att.item(), loss_d.item(), loss_d_real.item(), loss_d_fake.item(),
                    psnr_value))
        with torch.no_grad():
            att, fake_b = gen.forward(real_a[0:1])

            fake_heat_map = att.repeat(1, 3, 1, 1)
            real_heat_map = M[0:1].repeat(1, 3, 1, 1)
            output_image = torchvision.utils.make_grid([
                normalize(real_a[0:1, [3, 2, 1]])[0],
                real_heat_map[0],
                fake_heat_map[0],
                normalize(real_b[0:1, [3, 2, 1]])[0],
                normalize(fake_b[:, [3, 2, 1]])[0],
            ], nrow=5)

            torchvision.utils.save_image(output_image, "SPAGAN/log" + "_" + name + "/Epoch_" + str(epoch) + ".png")

        torch.save(dis, "SPAGAN/checkpoint" + "_" + name + "/Dis" + "_" + name + ".pth")
        torch.save(gen, "SPAGAN/checkpoint" + "_" + name + "/Gen" + "_" + name + ".pth")
        print("Epoch " + str(epoch) + " Done")

def display_training_images(epoch,
                                landsat_batch_x_cloudy, landsat_batch_y_cloud_free, predicted_image,cc, is_train=True):

        dirName = '/s/' + socket.gethostname() + '/a/nobackup/galileo/paahuni/cloud_removal/5/'
        if is_train:
            output_dir = dirName + 'trainALL/'
        else:
            output_dir = dirName + 'test/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        r, c = 1,3

        titles = ['Input Cloudy', 'Target Cloud Free', 'Predicted']

        fig, axarr = plt.subplots(r, c, figsize=(15, 12))
        np.vectorize(lambda axarr: axarr.axis('off'))(axarr)

        axarr[0].imshow(landsat_batch_x_cloudy)
        axarr[0].set_title(titles[0], fontdict={'fontsize': 15})

        axarr[1].imshow(landsat_batch_y_cloud_free)
        axarr[1].set_title(titles[1], fontdict={'fontsize': 15})

        axarr[2].imshow(predicted_image)
        axarr[2].set_title(titles[2], fontdict={'fontsize': 15})

        psnrV = psnrb(landsat_batch_y_cloud_free, predicted_image)

        if psnrV > 48:
            psnrV = 48.0

        plt.suptitle("PSNR: {} cc: {}%".format( round(psnrV,3), cc * 100, fontsize=20 ))

        fig.savefig(output_dir + "%s_%s.png" % (epoch, cc * 100))
        plt.close()

        return


if __name__ == '__main__':
    #     Lattice-176 to 180, 182-188, 190-210.
    hostname = socket.gethostname()
    runs = []
    dirName = '/s/' + socket.gethostname() + '/a/nobackup/galileo/'
    if (hostname == "lattice-176"):
        runs = [(0, 500), (500, 963)]
    elif (hostname == "lattice-177"):
        runs = [(0, 700), (700, 1400), (1400, 1923)]
    elif (hostname == "lattice-178"):
        runs = [(0, 700), (700, 1400), (1400, 2100), (2100, 2800), (2800, 3480)]
    elif (hostname == "lattice-179"):
        runs = [(0, 474)]
    elif (hostname == "lattice-180"):
        runs = [(0, 377)]
    elif (hostname == "lattice-182"):
        runs = [(0, 772)]
    elif (hostname == "lattice-183"):
        runs = [(0, 600), (600, 1200), (1200, 1700)]
    elif (hostname == "lattice-184"):
        runs = [(0, 400), (400, 801)]
    elif (hostname == "lattice-185"):
        runs = [(0, 500), (500, 1082)]
    elif (hostname == "lattice-186"):
        runs = [(0, 452)]
    elif (hostname == "lattice-187"):
        runs = [(0, 500), (500, 1000), (1000, 1440)]
    elif (hostname == "lattice-188"):
        runs = [(0, 645)]
    elif (hostname == "lattice-190"):
        runs = [(0, 600), (600, 1200), (1200, 1800), (1800, 2353)]
    elif (hostname == "lattice-191"):
        runs = [(0, 600), (600, 1200), (1200, 1800), (1800, 2384)]
    elif (hostname == "lattice-192"):
        runs = [(0, 450), (450, 908)]
    elif (hostname == "lattice-193"):
        runs = [(0, 500), (500, 1056)]
    elif (hostname == "lattice-194"):
        runs = [(0, 700), (700, 1400), (1400, 2114)]
    elif (hostname == "lattice-195"):
        runs = [(0, 700), (700, 1400), (1400, 1997)]
    elif (hostname == "lattice-196"):
        runs = [(0, 700), (700, 1400), (1400, 2100), (2100, 2800), (2800, 3395)]
    elif (hostname == "lattice-197"):
        runs = [(0, 700), (700, 1400), (1400, 2058)]
    elif (hostname == "lattice-198"):
        runs = [(0, 700), (700, 1400), (1400, 2144)]
    elif (hostname == "lattice-199"):
        runs = [(0, 600), (600, 1200), (1200, 1800), (1800, 2400), (2400, 3080)]
    elif (hostname == "lattice-200"):
        runs = [(0, 500), (500, 1049), (1400, 2058)]
    elif (hostname == "lattice-201"):
        runs = [(0, 600), (600, 1200), (1200, 1800), (1800, 2411)]
    elif (hostname == "lattice-202"):
        runs = [(0, 700), (700, 1400), (1400, 2100), (2100, 2627)]
    elif (hostname == "lattice-203"):
        runs = [(0, 600), (600, 1200), (1200, 1800), (1800, 2400), (2400, 3061)]
    elif (hostname == "lattice-204"):
        runs = [(0, 700), (700, 1400), (1400, 2100), (2100, 2849)]
    elif (hostname == "lattice-205"):
        runs = [(0, 700), (700, 1400), (1400, 2000)]
    elif (hostname == "lattice-206"):
        runs = [(0, 600), (600, 1235)]
    elif (hostname == "lattice-207"):
        runs = [(0, 600), (600, 1200), (1200, 1753)]
    elif (hostname == "lattice-208"):
        runs = [(0, 750), (750, 1566)]
    elif (hostname == "lattice-209"):
        runs = [(0, 700), (700, 1400), (1400, 2100), (2100, 2891)]
    elif (hostname == "lattice-210"):
        runs = [(0, 600), (600, 1200), (1200, 1719)]
    elif (hostname == "lattice-225"):
        runs = [(1800, 2411)]  # 201

    test_dataset = Test_Dataset(isTrain=True)
    test_size = len(test_dataset)
    print('Test Dataset Length:', test_size)
    #     , num_workers=5
    test_data_loader = DataLoader(dataset=test_dataset)

    batch_size = 1
    in_ch = 3
    out_ch = 3
    width = 164
    height = 123

    real_a = torch.FloatTensor(batch_size, in_ch, width, height)
    real_b = torch.FloatTensor(batch_size, out_ch, width, height)
    real_test = torch.FloatTensor(batch_size, out_ch, width, height)
    M = torch.FloatTensor(batch_size, 1, width, height)

    if torch.cuda.is_available():
        real_a = real_a.cuda()
        real_b = real_b.cuda()
        M = M.cuda()

    real_a = Variable(real_a)
    real_b = Variable(real_b)
    real_test = Variable(real_test)

    results = {}

    # Iteraterates Through Test Images
    psnr_ary = []
    psnrb_ary = []
    rase_ary = []
    mse_ary = []
    ssim_ary = []
    ugi_ary = []
    ergas_ary = []

    for iteration, batch in enumerate(test_data_loader, 1):

        real_a_cpu, real_b_cpu, M_cpu, data, geohash, cc = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)
        M.data.resize_(M_cpu.size()).copy_(M_cpu)
        # if float(cc[0]) * 100 < 60:
        #     continue

        best_performance = 0
        # Each run represents a different model on the same machine
        for run in runs:
            name = socket.gethostname() + "_" + str(run[0]) + "_" + str(run[1])
            file_exists = exists(dirName + "SPAGAN/checkpoint" + "_" + name + "/Gen" + "_" + name + ".pth")
            if file_exists:
                model = torch.load(dirName + "SPAGAN/checkpoint" + "_" + name + "/Gen" + "_" + name + ".pth").cuda()


                _, fake_b = model.forward(real_a)

                real_bb = resize(real_b[0].permute(1, 2, 0).detach().cpu().numpy(), (128,128,3), preserve_range=True)
                real_aa = resize(real_a[0].permute(1, 2, 0).detach().cpu().numpy(), (128,128,3), preserve_range=True)
                fake_bb = resize(fake_b[0].permute(1, 2, 0).detach().cpu().numpy(), (128,128,3), preserve_range=True)


                psnr_1 = psnr(real_bb,fake_bb, MAX=1.0)
                psnrb_1 = psnrb(real_bb,fake_bb)
                rase_1 = rase(real_bb,fake_bb)
                mse_1 = mse((real_bb * 2 - 1),(fake_bb * 2 - 1))
                ssim_1 = ssim(real_bb,fake_bb, MAX=1.0)
                uqi_1 = uqi(real_bb,fake_bb)
                ergas_1 = ergas(real_bb,fake_bb)
                train_dataset = Sat_Dataset(start=run[0], stop=run[1])
                geohashes = train_dataset.getGeohashes()

                if geohash[0] in geohashes:
                    print(iteration)
                    psnr_ary.append(psnr_1)
                    psnrb_ary.append(psnrb_1)
                    rase_ary.append(rase_1)
                    mse_ary.append(mse_1)
                    ssim_ary.append(ssim_1[0])
                    ugi_ary.append(uqi_1)
                    ergas_ary.append(ergas_1)
                    display_training_images(iteration, real_aa, real_bb, fake_bb, float(cc[0]), is_train=True)
            else:
                continue
    print("Train")
    print("PSNR", sum(psnr_ary) / len(psnr_ary))
    print("PSNR-B", sum(psnrb_ary) / len(psnrb_ary))
    print("RASE", sum(rase_ary) / len(rase_ary))
    print("MSE", sum(mse_ary) / len(mse_ary))
    print("SSIM", sum(ssim_ary) / len(ssim_ary))
    print("ERGAS", sum(ergas_ary) / len(ergas_ary))
    # prediction
#             print(fake_b)

# real
#             print(real_b)

# Some code should be added below to decide which run to use on each image e.g.
#             if best_performance < PSNR(real_b, fake_b):
#                 best_performance = PSNR(real_b, fake_b)
