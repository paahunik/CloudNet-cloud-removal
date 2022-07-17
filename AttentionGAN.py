from sklearn import preprocessing
from tensorflow.keras.layers import Flatten, ReLU, TimeDistributed,Activation,Dropout, RepeatVector, Input, Conv2D,Bidirectional, Conv2DTranspose, LeakyReLU,Reshape, Dense, BatchNormalization, Concatenate
from tensorflow.keras.models import Model,model_from_json
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Lambda
import tensorflow as tf
from tensorflow.keras import backend as K
import argparse
import socket
import os
from matplotlib import pyplot as plt
from time import strptime, mktime
import datetime
import data_loader_clouds, train
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import plot_model
import horovod.tensorflow.keras as hvd

# import torch
# from torch import nn
# import torch.nn.functional as F
# from collections import OrderedDict
# from models.models_utils import weights_init, print_network

class AttentionGAN():

    def __init__(self,timeStep=1, batch_size=1, latent_dim=1024, w=128,h=128):
        self.targetH = h
        self.targetW = w
        self.targetShape = (self.targetW, self.targetH, 8)
        self.inputShape = (self.targetW, self.targetH, 8)
        self.sentShape = (self.targetH,self.targetW, 6)
        self.no_of_timesteps = timeStep

        self.batch_size = batch_size

        self.host_addr = socket.gethostbyname(socket.gethostname()) + ':15606'
        self.data_load = data_loader_clouds.DatasetHandling(h=h, w=w,no_of_timesteps=timeStep,
                                                                    batch_size=self.batch_size)

        self.dirName = "/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/"

        hvd.init()
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        self.latent_dim=latent_dim
        # self.gan_optimizer = hvd.DistributedOptimizer(tf.keras.optimizers.Adam(0.0002 * hvd.size(), beta_1=0.5))

        # self.g_model = self.getGeneratorModel()


    def getGeneratorModel(self):
        in_src_image = Input(shape=self.targetShape, name= "InputCloudyImage")
        inpPrevLand = Input(shape=self.inputShape, name='PrevLandsat')
        inpSentImage = Input(shape=self.sentShape, name="Sentinel2layer")

        model = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same", activation='relu')(in_src_image)
        model = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        sentFeatures = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu')(inpSentImage)
        sentFeatures = Conv2D(32, (1, 1), strides=(1, 1), padding='same')(sentFeatures)
        sentFeatures = ReLU()(sentFeatures)
        sentFeatures = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(sentFeatures)
        sentFeatures = ReLU()(sentFeatures)
        sentFeatures = Conv2D(128, (1, 1), strides=(1, 1), padding='same')(sentFeatures)
        sentFeatures = ReLU()(sentFeatures)

        prevLanFeature = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu')(inpPrevLand)
        prevLanFeature = Conv2D(32, (1, 1), strides=(1, 1), padding='same')(prevLanFeature)
        prevLanFeature = ReLU()(prevLanFeature)

        merged = Concatenate()([model, prevLanFeature])

        model = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same")(merged)
        model = ReLU()(model)

        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        merged = Concatenate()([model, sentFeatures])

        model = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same")(merged)
        model = ReLU()(model)

        model = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=7, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)

        genmodel = Model([in_src_image,inpPrevLand,inpSentImage], [model], name="ConvModel")
        # plot_model(genmodel, to_file='./Model3.png', rankdir='TB', show_shapes=True, show_layer_names=True)
        # model = BatchNormalization()(model)
        return genmodel




# def conv1x1(in_channels, out_channels, stride = 1):
#         return nn.Conv2d(in_channels,out_channels,kernel_size = 1,
#                         stride =stride, padding=0,bias=False)
#
# def conv3x3(in_channels, out_channels, stride = 1):
#         return nn.Conv2d(in_channels,out_channels,kernel_size = 3,
#             stride =stride, padding=1,bias=False)
#
#
# class Bottleneck(nn.Module):
#     def __init__(self,in_channels,out_channels,):
#         super(Bottleneck,self).__init__()
#         m  = OrderedDict()
#         m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         m['relu1'] = nn.ReLU(True)
#         m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False,dilation=2)
#         m['relu2'] = nn.ReLU(True)
#         m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
#         self.group1 = nn.Sequential(m)
#         self.relu= nn.Sequential(nn.ReLU(True))
#
#     def forward(self, x):
#         out = self.group1(x)
#         return out
#
#
# class SPANet(nn.Module):
#     def __init__(self):
#         super(SPANet, self).__init__()
#
#         self.conv_in = nn.Sequential(
#             conv3x3(3, 32),
#             nn.ReLU(True)
#         )
#
#         self.res_block1 = Bottleneck(32, 32)
#         # self.res_block2 = Bottleneck(32,32)
#
#
#
#     def forward(self, x):
#         out = self.conv_in(x)
#         out = F.relu(self.res_block1(out) )
#         # out = F.relu(self.res_block2(out) + out)
#         return  out

if __name__ == '__main__':
    #test

    # model = SPANet()

    # print("modle")
    #
    # print(model)

    OV = AttentionGAN()
    s = OV.getGeneratorModel()
    print(s.summary())



