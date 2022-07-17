from sklearn import preprocessing
from tensorflow.keras.layers import Flatten, ReLU, TimeDistributed,Activation,Dropout, RepeatVector, Input, Conv2D,Bidirectional, Conv2DTranspose, LeakyReLU,Reshape, Dense, BatchNormalization, Concatenate
from tensorflow.keras.models import Model,model_from_json
import numpy as np
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Lambda, Add
import tensorflow as tf
from tensorflow.keras import backend as K
import argparse
import socket
import os
from matplotlib import pyplot as plt
from time import strptime, mktime
import datetime
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir + '/../')

import data_loader_clouds, train
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import plot_model

class cloud_removal_models():

    def __init__(self,timeStep=1, batch_size=1, latent_dim=1024, w=128,h=128):
        self.targetH = h
        self.targetW = w
        self.targetShape = (self.targetW, self.targetH, 8)
        self.inputShape = (self.targetW, self.targetH, 8)
        self.sentShape = (self.targetH,self.targetW, 8)
        self.no_of_timesteps = timeStep
        self.batch_size = batch_size
        self.host_addr = socket.gethostbyname(socket.gethostname()) + ':15606'
        self.dirName = "/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/"
        self.latent_dim=latent_dim
        self.g_model = self.getGeneratorModel2()

    def resNet_LandSat_model(self, num_layers=16, layer_output_feature_size=256):
        def resBlock(input_l, layer_output_feature_size, kernel_size, scale=0.1):
            tmp = Conv2D(layer_output_feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(
                input_l)
            tmp = Activation('relu')(tmp)
            tmp = Conv2D(layer_output_feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(tmp)
            tmp = Lambda(lambda x: x * scale)(tmp)
            return Add()([input_l, tmp])

        input_data = Input(shape=self.targetShape)
        x = input_data
        x = Conv2D(layer_output_feature_size, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
        x = Activation('relu')(x)
        for i in range(num_layers):
            x = resBlock(x, layer_output_feature_size, kernel_size=[3, 3])
        x = Conv2D(self.targetShape[2], (3, 3), kernel_initializer='he_uniform', padding='same')(x)
        x = Add()([x, input_data])
        model = Model(inputs=input_data, outputs=x)
        return model

    def SSIMLoss(self, y_true, y_pred):
        return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def ssim_loss_custom(self, y_true, y_pred):
        ss = tf.image.ssim(y_true, y_pred, 2.0)
        ss = tf.where(tf.math.is_nan(ss), -K.ones_like(ss), ss)
        return -tf.reduce_mean(ss)


    def getEncodingDecodingModel(self):
        init = RandomNormal(stddev=0.2)
        in_src_image = Input(shape=self.targetShape)
        out_target_image = Input(shape=self.targetShape)

        model = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", init=init)(in_src_image)
        model = ReLU(alpha=0.2)(model)
        model = BatchNormalization()(model)

        model = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same")(model)
        model = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same")(model)
        model = ReLU(alpha=0.2)(model)
        model = BatchNormalization(axis=-1)(model)

        model = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same")(model)
        model = ReLU(alpha=0.2)(model)
        model = BatchNormalization(axis=-1)(model)

        model = Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same")(model)
        model = ReLU(alpha=0.2)(model)
        model = BatchNormalization(axis=-1)(model)

        model = Conv2D(filters=1024, kernel_size=(4, 4), strides=(2, 2), padding="same")(model)
        model = ReLU(alpha=0.2)(model)
        model = BatchNormalization(axis=-1)(model)

        model = Conv2D(filters=2048, kernel_size=(4, 4), strides=(2, 2), padding="same")(model)

        volumeSize = K.int_shape(model)
        model = Flatten()(model)
        latent = Dense(self.latent_dim)(model)

        encode_model = Model(in_src_image, latent, name="encoder")


        # model.add(Reshape((model.output_shape[-1], model.output_shape[0] * model.output_shape[1])))
        # model = LeakyReLU(alpha=0.2)(model)
        # model = Dense(units=out_features)(model)
        #
        # model = Conv2D()(model)
        # model = LeakyReLU(alpha=0.2)(model)
        #
            # encode_model = Model([in_src_image, out_target_image], model)

        image_shape = (self.inputShape, 7)
        init = RandomNormal(stddev=0.2)
        out_target_image = Input(shape=image_shape)
        latentInputs = Input(shape=(self.latent_dim,))
        x = Dense(np.prod(volumeSize[1:]))(latentInputs)
        model = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

        model = LeakyReLU(alpha=0.2)(model)
        model = Conv2D(filters=1024, kernel_size=(4, 4), strides=(2, 2), padding="same", init=init)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization()(model)

        model = Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same")(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization()(model)

        model = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same")(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization()(model)

        model = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same")(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization()(model)

        model = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same")(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization()(model)

        model = Conv2D(filters=7, kernel_size=(4, 4), strides=(2, 2), padding="same")(model)
        model = Activation('tanh')(model)

        decoder_model = Model(latentInputs, model, name="encoder")
        #
        # autoencoder = Model(inputs, decoder(encoder(inputs)),
        #                     name="autoencoder")

        return encode_model, decoder_model
    def getGeneratorModel3(self):
        in_src_image = Input(shape=self.targetShape, name="InputCloudyImage")
        inpPrevLand = Input(shape=self.inputShape, name='PrevLandsat')
        inpSentImage = Input(shape=self.sentShape, name="Sentinel2layer")
        model = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same", activation='relu')(in_src_image)
        model = Dropout(0.2)(model)

        model = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(model)
        model = BatchNormalization()(model)
        model = ReLU()(model)

        model = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)
        # model = Dropout(0.2)(model)

        model = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = BatchNormalization()(model)
        model = ReLU()(model)

        model = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same")(model)
        model = BatchNormalization()(model)
        model = ReLU()(model)
        model = Dropout(0.2)(model)

        sentFeatures = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu')(inpSentImage)


        sentFeatures = Conv2D(32, (1, 1), strides=(1, 1), padding='same')(sentFeatures)
        sentFeatures = ReLU()(sentFeatures)
        sentFeatures = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(sentFeatures)
        sentFeatures = ReLU()(sentFeatures)
        sentFeatures = Conv2D(128, (1, 1), strides=(1, 1), padding='same')(sentFeatures)
        model = BatchNormalization()(model)
        sentFeatures = ReLU()(sentFeatures)
        # sentFeatures = Dropout(0.2)(sentFeatures)

        prevLanFeature = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu')(inpPrevLand)
        prevLanFeature = Dropout(0.2)(prevLanFeature)
        prevLanFeature = Conv2D(32, (1, 1), strides=(1, 1), padding='same')(prevLanFeature)
        prevLanFeature = ReLU()(prevLanFeature)

        merged = Concatenate()([model, prevLanFeature])

        model = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding="same")(merged)
        model = ReLU()(model)

        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = BatchNormalization()(model)
        model = ReLU()(model)
        # model = Dropout(0.2)(model)

        model = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same")(model)
        model = BatchNormalization()(model)
        model = ReLU()(model)
        model = Dropout(0.2)(model)

        merged = Concatenate()([model, sentFeatures])

        model = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same")(merged)
        model = ReLU()(model)

        model = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = BatchNormalization()(model)
        model = ReLU()(model)
        model = Dropout(0.2)(model)

        model = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)
        # model = Dropout(0.2)(model)

        model = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = BatchNormalization()(model)
        model = ReLU()(model)

        model = Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)
        model = Dropout(0.2)(model)

        model = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)

        genmodel = Model([in_src_image, inpPrevLand, inpSentImage], [model], name="ConvModel")
        # if self.lossMethod == 'w':
        #     genmodel.compile(loss=self.weightedLoss, optimizer=self.gan_optimizer, experimental_run_tf_function=False)
        # elif self.lossMethod == 'mse':
        # genmodel.compile(loss='mse', optimizer=self.gan_optimizer, experimental_run_tf_function=False)

        # plot_model(genmodel, to_file='./Model3.png', rankdir='TB', show_shapes=True, show_layer_names=True)
        return genmodel

    def getDiscriminatorModel(self):
        image_shape = (self.targetH, self.targetH, 7)
        init = RandomNormal(stddev=0.02)
        in_src_image = Input(shape=image_shape)
        in_target_image = Input(shape=image_shape)

        merged = Concatenate()([in_src_image, in_target_image])
        d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(512, (4, 4), strides=(2, 2), padding='same')(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(512, (4, 4), padding='same')(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(1, (4, 4), padding='same')(d)
        patch_out = Activation('sigmoid')(d)

        model = Model([in_src_image, in_target_image], patch_out)

        # model.compile(loss='binary_crossentropy', optimizer=self.discriminator_optimizer, loss_weights=[0.5])

        return model

    def encoder_new(self, layer_in, n_filters, batchnorm=True):
        '''
        This function protrays the architecture of an encoder block
        '''
        # weight initialization
        init = RandomNormal(stddev=0.02)

        # add downsampling layer
        g = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(
            layer_in)  # add downsampling layer

        # conditionally add batch normalization
        if batchnorm:
            g = BatchNormalization()(g, training=True)

        # Activating Leaky RelU
        g = LeakyReLU(alpha=0.2)(g)
        return g

    def getGeneratorModel2(self):
        in_src_image = Input(shape=self.targetShape, name= "InputCloudyImage")
        inpPrevLand = Input(shape=self.inputShape, name='PrevLandsat')
        inpSentImage = Input(shape=self.sentShape, name="Sentinel2layer")


        model = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same", activation='relu')(in_src_image)
        model = Dropout(0.2)(model)
        model = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)


        model = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)
        model = Dropout(0.2)(model)

        model = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)
        model = Dropout(0.2)(model)

        sentFeatures = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu')(inpSentImage)


        sentFeatures = Conv2D(32, (1, 1), strides=(1, 1), padding='same')(sentFeatures)
        sentFeatures = ReLU()(sentFeatures)
        sentFeatures = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(sentFeatures)
        sentFeatures = ReLU()(sentFeatures)
        sentFeatures = Conv2D(128, (1, 1), strides=(1, 1), padding='same')(sentFeatures)
        sentFeatures = ReLU()(sentFeatures)
        sentFeatures = Dropout(0.2)(sentFeatures)

        prevLanFeature = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu')(inpPrevLand)
        prevLanFeature = Dropout(0.2)(prevLanFeature)
        prevLanFeature = Conv2D(32, (1, 1), strides=(1, 1), padding='same')(prevLanFeature)
        prevLanFeature = ReLU()(prevLanFeature)

        merged = Concatenate()([model, prevLanFeature])

        model = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same")(merged)
        model = ReLU()(model)

        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)
        model = Dropout(0.2)(model)

        model = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)
        model = Dropout(0.2)(model)

        merged = Concatenate()([model, sentFeatures])

        model = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same")(merged)
        model = ReLU()(model)

        model = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)
        model = Dropout(0.2)(model)

        merged = Concatenate()([model, prevLanFeature])

        model = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same")(merged)
        model = ReLU()(model)

        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)
        model = Dropout(0.2)(model)

        model = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)
        model = Dropout(0.2)(model)

        model = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)

        genmodel = Model([in_src_image,inpPrevLand,inpSentImage], [model], name="ConvModel")
        # plot_model(genmodel, to_file='./Model3.png', rankdir='TB', show_shapes=True, show_layer_names=True)
        # model = BatchNormalization()(model)
        return genmodel

    def getGeneratorModel(self):
        # weight initialization
        init = RandomNormal(stddev=0.02)

        inpCloudyLand = Input(shape=self.targetShape, name='CloudyLandsat')
        inpPrevLand = Input(shape=self.inputShape, name='PrevLandsat')
        inpSentImage = Input(shape=self.sentShape, name="Sentinel2")

        model = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same"))(inpPrevLand)
        model = LeakyReLU(0.2)(model)

        model = TimeDistributed(Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same"))(model)
        model = BatchNormalization()(model, training=True)
        model = LeakyReLU(0.2)(model)

        # model = TimeDistributed(Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same"))(model)
        # model = BatchNormalization()(model, training=True)
        # model = LeakyReLU(0.2)(model)
        #
        # model = TimeDistributed(Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same"))(model)
        # model = BatchNormalization()(model, training=True)
        # model = LeakyReLU(0.2)(model)
        #
        # model = TimeDistributed(Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding="same"))(model)
        # model = BatchNormalization()(model, training=True)
        # model = LeakyReLU(0.2)(model)
        #
        # model = TimeDistributed(Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding="same"))(model)
        # model = BatchNormalization()(model, training=True)
        # model = LeakyReLU(0.2)(model)
        model = Reshape((model.shape[2],model.shape[3], model.shape[1] * model.shape[-1]))(model)
        model = Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)

        sentFeatures = self.encoder_new(inpSentImage, 64, batchnorm=False)
        sentFeatures = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(sentFeatures)
        #sentFeatures = self.encoder_new(sentFeatures, 128)
        # sentFeatures = self.encoder_new(sentFeatures, 256)
        # sentFeatures = self.encoder_new(sentFeatures, 512)
        # sentFeatures = self.encoder_new(sentFeatures, 256)
        # sentFeatures = Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding="same")(sentFeatures)

        cloudyLandFeatures = self.encoder_new(inpCloudyLand, 64, batchnorm=False)
        # cloudyLandFeatures = self.encoder_new(cloudyLandFeatures, 128)
        cloudyLandFeatures = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same")(cloudyLandFeatures)
        # cloudyLandFeatures = self.encoder_new(cloudyLandFeatures, 256)

        merged = Concatenate()([model, sentFeatures, cloudyLandFeatures])
        merged = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(merged)
        merged = Activation('relu')(merged)

        # merged = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(merged)
        # merged = Activation('relu')(merged)

        # merged = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(merged)
        # merged = Activation('relu')(merged)

        # clouds_merged = Concatenate()([merged, cloudyLandFeatures])

        # decoder model
        finmodel = Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same')(merged)
        finmodel = BatchNormalization()(finmodel, training=True)
        finmodel = Dropout(0.5)(finmodel, training=True)
        finmodel = Activation('relu')(finmodel)

        finmodel = Conv2DTranspose(32, (1, 1), strides=(1, 1), padding='same')(finmodel)
        finmodel = BatchNormalization()(finmodel, training=True)
        finmodel = Activation('relu')(finmodel)

        # output
        finmodel = Conv2DTranspose(7, (1, 1), strides=(1, 1), padding='same')(finmodel)
        finmodel = Activation('tanh')(finmodel)

        genModel = Model(inputs=[inpCloudyLand,inpPrevLand,inpSentImage], outputs=[finmodel], name='Generator')
        # genModel.compile(loss='mse', optimizer=self.gan_optimizer, experimental_run_tf_function=False)
        plot_model(genModel, to_file='Model2.png', rankdir='TB', show_shapes=True, show_layer_names=True)
        return genModel

    def get_model_memory_usage(self,batch_size, model):
        shapes_mem_count = 0
        internal_model_mem_count = 0
        for l in model.layers:
            layer_type = l.__class__.__name__
            if layer_type == 'Model':
                internal_model_mem_count += self.get_model_memory_usage(batch_size, l)
            single_layer_mem = 1
            out_shape = l.output_shape
            if type(out_shape) is list:
                out_shape = out_shape[0]
            for s in out_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

        number_size = 4.0
        if K.floatx() == 'float16':
            number_size = 2.0
        if K.floatx() == 'float64':
            number_size = 8.0

        total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
        return gbytes

    def saveModelGen(self):

        model_json = self.g_model.to_json()
        self.g_model.save_weights(self.dirName + "ModelCheckp/GeneratorModel.h5")

        with open(self.dirName + "ModelCheckp/GeneratorModel.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5

        print("Saved model to disk")
        return

    # def define_gan_new(self, g_model, d_model, image_shape):
    #     d_model.trainable = False
    #     in_src = Input(shape=seimage_shape)
    #     gen_out = g_model(in_src)
    #     dis_out = d_model([in_src, gen_out])
    #     model = Model(in_src, [dis_out, gen_out])
    #     opt = Adam(lr=0.00002, beta_1=0.5)
    #     model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    #     return model


if __name__ == '__main__':
    train_cl = cloud_removal_models(w=128, h=128, timeStep=1, batch_size=1)
    # Resnet model: 37,802,248
    # My model : 1,030,120
    model = train_cl.getGeneratorModel3()
    print(model.summary())

    # train_cl.get_model_memory_usage(6,model)
    #
    # parser = argparse.ArgumentParser()
    # # parser.add_argument('--outputDir', type=int, default=930)
    # parser.add_argument('--epochs', type=int, default=3000)
    # parser.add_argument('--batch', type=int, default=1)
    # # parser.add_argument('--cloudCov', type=float, default=0.4)
    # # parser.add_argument('--onlyGenerator', type=bool, default=True)
    # args = parser.parse_args()
    # train_model_obj = CloudRemoval(img_width=128, img_height=128, batch_size=args.batch)
    # #train_model_obj.trainGAN(epochs=args.epochs, sample_interval=100)
    # train_model_obj.plot_train_test_samples_test()
