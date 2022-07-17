import math
import tensorflow as tf
import horovod.tensorflow.keras as hvd

hvd.init()
from image_similarity_measures.quality_metrics import fsim
import sewar

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
from cloud_removal import data_loader_single_mask, model_helpers
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, model_from_json, Model
from tensorflow.keras.layers import Dense, Lambda, Dropout, Flatten, Conv2D, MaxPooling2D, ReLU, \
    Activation, Dropout, RepeatVector, Input, Add, \
    Conv2DTranspose, LeakyReLU, Reshape, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from scipy.stats.stats import pearsonr
import os
import numpy as np
import random

np.seterr(divide='ignore', invalid='ignore')
import socket
import gdal
import datetime
import pickle
from tensorflow.keras.callbacks import LambdaCallback


class Prediction_Class():
    def __init__(self, folderI=6, batch_size=1, istrain=True, h=128, w=128, lossMethod='mse', tranferLear=None,
                 loadModel=True, isRes=False):
        # print("tRAN", tranferLear)
        self.isRes = isRes
        self.geohashes1 = ['5', '7', '6', '4', '1', '3', '0', '2', 'p', 'r', 'n', 'q', 'j', 'm', 'h', 'k']

        self.geohashes2 = ['8', 'e', 'g', 'd', 'f', '9', 'c', 'b', 'x', 'z', 'w', 'y', 't', 'v', 's', 'u']

        self.folderI = folderI
        self.batch_size = batch_size
        self.dataloader = data_loader_single_mask.DatasetHandling(h, w, folderI=self.folderI,
                                                                  album='iowa-2015-2020-spa', no_of_timesteps=1,
                                                                  batch_size=batch_size, istrain=istrain,
                                                                  saveInputMetaData=False
                                                                  )
        self.model_helper = model_helpers.cloud_removal_models(timeStep=1, batch_size=batch_size, w=128, h=128)

        self.isTrain = istrain
        self.dirName = "/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/" + str(
            self.folderI) + "/"
        self.loadModel = loadModel
        if not os.path.isdir("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/"):
            os.mkdir("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/")
        if not os.path.isdir(self.dirName):
            os.mkdir(self.dirName)

        self.test_dataset_obj = data_loader_single_mask.DatasetHandling(h, w, folderI=self.folderI,
                                                                        album='iowa-2015-2020-spa', no_of_timesteps=1,
                                                                        batch_size=5, istrain=False,
                                                                        saveInputMetaData=False)
        self.lossMethod = lossMethod
        self.targetH = h
        self.targetW = w
        self.targetShape = (self.targetW, self.targetH, 8)
        self.inputShape = (self.targetW, self.targetH, 8)
        self.sentShape = (self.targetH, self.targetW, 8)
        self.gan_optimizer = hvd.DistributedOptimizer(tf.keras.optimizers.Nadam(0.0001 * hvd.size()))
        if isRes:
            self.generator = self.load_model_Res(tranferLear)
            self.generator2 = self.load_model_Res(tranferLear)
        else:
            self.generator = self.load_model(1005)
            self.generator2 = self.load_model(1008)

    def psnrLossMetric(self, y_true, y_predict):
        psnr = tf.image.psnr(y_true, y_predict, max_val=2)
        return psnr

    def Cloudnet(self):
        in_src_image = Input(shape=self.targetShape, name="InputCloudyImage")
        inpPrevLand = Input(shape=self.inputShape, name='PrevLandsat')
        inpSentImage = Input(shape=self.sentShape, name="Sentinel2layer")
        model = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same", activation='relu')(in_src_image)
        model = Dropout(0.2)(model)
        model = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)
        # model = Dropout(0.2)(model)

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
        # sentFeatures = Dropout(0.2)(sentFeatures)

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
        # model = Dropout(0.2)(model)

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
        # model = Dropout(0.2)(model)

        merged = Concatenate()([model, prevLanFeature])

        model = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same")(merged)
        model = ReLU()(model)

        model = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)
        # model = Dropout(0.2)(model)

        model = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)
        # model = Dropout(0.2)(model)

        model = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)

        genmodel = Model([in_src_image, inpPrevLand, inpSentImage], [model], name="ConvModel")

        genmodel.compile(loss='mse', optimizer=self.gan_optimizer, experimental_run_tf_function=False,
                         metrics=[self.psnrLossMetric, 'mse'])

        return genmodel

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

        # optimizer = hvd.DistributedOptimizer(Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
        #                                      backward_passes_per_step=1,
        #                                      average_aggregated_gradients=True)

        model.compile(optimizer=self.gan_optimizer, loss='mse')
        return model

    def load_model(self, dir=None, dir2=None):
        if dir is not None:
            dirname = "/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/" + str(dir) + "/"
        # if dir2 is not None:
        #     dirname2 = "/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/" + str(dir2) + "/"
        # else:
        #     dirname = self.dirName
        json_file = open(dirname + "GeneratorModel.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # loaded_model = self.resNet_LandSat_model()
        loaded_model.load_weights(dirname + "GeneratorModel.h5")

        loaded_model2 = model_from_json(loaded_model_json)
        # if dir2 is not None:
        #     loaded_model2.load_weights(dirname2 + "GeneratorModel.h5")
        #     self.generator2 = loaded_model2
        #     self.generator2.compile(loss='mse', optimizer=self.gan_optimizer, experimental_run_tf_function=False,
        #                             metrics=[self.psnrLossMetric, 'mse'])
        # else:
        #     loaded_model2 = None

        self.generator = loaded_model
        self.generator.compile(loss='mse', optimizer=self.gan_optimizer, experimental_run_tf_function=False,
                               metrics=[self.psnrLossMetric, 'mse'])
        return loaded_model

    def load_model_Res(self, dir=None, dir2=None):
        if dir is not None:
            dirname = "/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/" + str(dir) + "/"

        loaded_model = self.resNet_LandSat_model()
        loaded_model.load_weights(dirname + "GeneratorModel.h5")
        self.generator = loaded_model
        self.generator.compile(loss='mse', optimizer=self.gan_optimizer, experimental_run_tf_function=False,
                               metrics=[self.psnrLossMetric, 'mse'])
        self.generator2 = loaded_model
        self.generator2.compile(loss='mse', optimizer=self.gan_optimizer, experimental_run_tf_function=False,
                                metrics=[self.psnrLossMetric, 'mse'])
        return loaded_model

    def normal01(self, img):
        return img / 255

    def mse(self, img1, img2):
        return np.mean((img2.astype(np.float64) - img1.astype(np.float64)) ** 2)

    def display_training_images(self, epoch,
                                landsat_batch_x_cloudy, land_cloudy_with_clouds_batch, landsat_batch_y_cloud_free,
                                sent_batch, landsat_prev_batch, geo_batch, cc_batch, is_train=True, is1005=True):
        if is_train:
            output_dir = self.dirName + 'trainALL/'
        else:
            output_dir = self.dirName + 'test/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        r, c = 2, 4
        cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)
        if self.isRes:
            fakeImg = (self.dataloader.denormalize11(
                self.generator.predict(landsat_batch_x_cloudy)[-1])).astype(np.uint8)
        else:
            if is1005:
                fakeImg = (self.dataloader.denormalize11(
                    self.generator.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(
                    np.uint8)
            else:
                fakeImg = (self.dataloader.denormalize11(
                    self.generator2.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(
                    np.uint8)

            for i in range(0, self.targetH):
                for j in range(0, self.targetW):
                    if not all(cloudy_img_land[i, j] == 0):
                        fakeImg[i, j] = cloudy_img_land[i, j, :]

        target_cloudfree_land = self.dataloader.denormalize11(landsat_batch_y_cloud_free[-1][:, :, :8]).astype(np.uint8)

        inp_sent_img = sent_batch[-1]
        inp_prev_landsat = np.array([self.dataloader.denormalize11(img).astype(np.uint8) for img in landsat_prev_batch])
        land_cloudy_with_clouds = self.dataloader.denormalize11(land_cloudy_with_clouds_batch[-1]).astype(np.uint8)
        titles = ['Previous Year Landsat', 'Sentinel-2 RGB', 'Sentinel-2 Edges', 'Input Cloudy', 'Input Cloud Mask',
                  'Target Cloud Free', 'Predicted']

        fig, axarr = plt.subplots(r, c, figsize=(15, 12))
        np.vectorize(lambda axarr: axarr.axis('off'))(axarr)

        for col in range(1):
            axarr[0, col].imshow(inp_prev_landsat[col][:, :, :3])
            axarr[0, col].set_title(titles[col], fontdict={'fontsize': 15})

        axarr[0, 1].imshow(inp_sent_img[:, :, :3])
        axarr[0, 1].set_title(titles[1], fontdict={'fontsize': 15})

        axarr[0, 2].imshow(inp_sent_img[:, :, -1])
        axarr[0, 2].set_title(titles[2], fontdict={'fontsize': 15})

        axarr[0, 3].imshow(land_cloudy_with_clouds[:, :, :3])
        axarr[0, 3].set_title(titles[3], fontdict={'fontsize': 15})

        axarr[r - 1, 0].imshow(cloudy_img_land[:, :, :3])
        axarr[r - 1, 0].set_title(titles[-3], fontdict={'fontsize': 15})

        axarr[r - 1, 1].imshow(target_cloudfree_land[:, :, :3])
        axarr[r - 1, 1].set_title(titles[-2], fontdict={'fontsize': 15})

        axarr[r - 1, 2].imshow(fakeImg[:, :, :3])
        axarr[r - 1, 2].set_title(titles[-1], fontdict={'fontsize': 15})

        # _,rmse = self.psnrAndRmse(fakeImg[:,:,:3], target_cloudfree_land[:,:,:3])

        psnr = tf.image.psnr(fakeImg[:, :, :3], target_cloudfree_land[:, :, :3], 255).numpy()
        psnrAll = tf.image.psnr(fakeImg, target_cloudfree_land, 255).numpy()
        if psnr > 48:
            psnr = 48.0
        if psnrAll > 48:
            psnrAll = 48.0
        plt.suptitle("Geohash: {} CloudCov: {}%  PSNR: {} PSNR ALL: {}".format(geo_batch[-1],
                                                                               str(round(float(cc_batch[-1]) * 100, 5)),
                                                                               np.round(psnr, 3), np.round(psnrAll, 3),
                                                                               fontsize=20))

        fig.savefig(output_dir + "%s_%s.png" % (epoch, str(float(cc_batch[-1]) * 100)))
        plt.close()

        return psnr, psnrAll

    def getErrorsNDVI(self):
        dataP = []
        dataA = []
        test_itr = self.dataloader.load_iterator_from_paths(batch_size=1, geohashes=self.geohashes1)
        test_itr2 = self.dataloader.load_iterator_from_paths(batch_size=1, geohashes=self.geohashes2)
        count = 0
        while count <= 500:
            try:
                print('loading image: ', count)
                landsat_batch_x_cloudy, _, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, _, _ = test_itr.__next__()

                fakeImg = (self.dataloader.denormalize11(
                    self.generator.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(
                    np.uint8)
                fakeImg = self.normal01(fakeImg)

                target_cloudfree_land = self.dataloader.denormalize11(landsat_batch_y_cloud_free[-1][:, :, :8]).astype(
                    np.uint8)  # Returns image pixels between (0, 255)
                target_cloudfree_land = self.normal01(target_cloudfree_land)
                cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)

                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        if not all(cloudy_img_land[i, j] == 0):
                            fakeImg[i, j] = cloudy_img_land[i, j, :]

                ndviFake = (fakeImg[:, :, 3] - fakeImg[:, :, 0]) / (fakeImg[:, :, 3] + fakeImg[:, :, 0])
                ndviReal = (target_cloudfree_land[:, :, 3] - target_cloudfree_land[:, :, 0]) / (
                            target_cloudfree_land[:, :, 3] + target_cloudfree_land[:, :, 0])
                dataP.append(np.mean(ndviFake))
                dataA.append(np.mean(ndviReal))
                count += 1
            except StopIteration:
                break
        while count <= 1000:
            try:
                print('loading image: ', count)
                landsat_batch_x_cloudy, _, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, _, _ = test_itr2.__next__()

                fakeImg = (self.dataloader.denormalize11(
                    self.generator2.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(
                    np.uint8)
                fakeImg = self.normal01(fakeImg)

                target_cloudfree_land = self.dataloader.denormalize11(landsat_batch_y_cloud_free[-1][:, :, :8]).astype(
                    np.uint8)  # Returns image pixels between (0, 255)
                target_cloudfree_land = self.normal01(target_cloudfree_land)
                cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)

                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        if not all(cloudy_img_land[i, j] == 0):
                            fakeImg[i, j] = cloudy_img_land[i, j, :]

                ndviFake = (fakeImg[:, :, 3] - fakeImg[:, :, 0]) / (fakeImg[:, :, 3] + fakeImg[:, :, 0])
                ndviReal = (target_cloudfree_land[:, :, 3] - target_cloudfree_land[:, :, 0]) / (
                            target_cloudfree_land[:, :, 3] + target_cloudfree_land[:, :, 0])
                dataP.append(np.mean(ndviFake))
                dataA.append(np.mean(ndviReal))
                count += 1
            except StopIteration:
                break

        if not self.isTrain:
            fileN = 'Testing_'
        else:
            fileN = 'Train_'
        with open(self.dirName + '/NDVI/ndviA' + fileN + str(socket.gethostname()[-3:]) + '.txt', 'wb') as f:
            pickle.dump(dataA, f)

        with open(self.dirName + '/NDVI/ndviP' + fileN + str(socket.gethostname()[-3:]) + '.txt', 'wb') as f:
            pickle.dump(dataP, f)
        # plt.style.use("ggplot")
        # coer,_ = pearsonr(dataA,dataP)
        # plt.scatter(dataA, dataP, label = 'Pearson CorrCoef: ' + str(round(coer, 4)), c='#44AA99', s=8)
        # plt.title('Correlation plot between NDVI index for\npredicted and actual cloud free image')
        # plt.ylabel('Predicted Average NDVI index')
        # plt.xlabel('Actual Average NDVI index')
        # plt.plot(np.unique(dataA), np.poly1d(np.polyfit(dataA, dataP, 1))(np.unique(dataA)), color='#999933')
        # plt.legend()
        # if self.isTrain:
        #     plt.savefig(self.dirName + 'NDVIComparisonTrain_' + str(socket.gethostname()[-3:]) +'.png')
        # else:
        #     plt.savefig(self.dirName + 'NDVIComparisonTest_' + str(socket.gethostname()[-3:]) + '.png')
        # plt.close()
        return

    def getErrorsNDMI(self):
        dataP = []
        dataA = []
        test_itr = self.dataloader.load_iterator_from_paths(batch_size=1, geohashes=self.geohashes1)
        test_itr2 = self.dataloader.load_iterator_from_paths(batch_size=1, geohashes=self.geohashes2)
        count = 0
        while count <= 500:
            try:
                print('loading image: ', count)
                landsat_batch_x_cloudy, _, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, _, _ = test_itr.__next__()

                fakeImg = (self.dataloader.denormalize11(
                    self.generator.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(
                    np.uint8)
                fakeImg = self.normal01(fakeImg)

                target_cloudfree_land = self.dataloader.denormalize11(landsat_batch_y_cloud_free[-1][:, :, :8]).astype(
                    np.uint8)  # Returns image pixels between (0, 255)
                target_cloudfree_land = self.normal01(target_cloudfree_land)
                cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)

                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        if not all(cloudy_img_land[i, j] == 0):
                            fakeImg[i, j] = cloudy_img_land[i, j, :]

                ndmiFake = (fakeImg[:, :, 3] - fakeImg[:, :, 4]) / (fakeImg[:, :, 3] + fakeImg[:, :, 4])
                ndmiReal = (target_cloudfree_land[:, :, 3] - target_cloudfree_land[:, :, 4]) / (
                        target_cloudfree_land[:, :, 3] + target_cloudfree_land[:, :, 4])
                dataP.append(np.mean(ndmiFake))
                dataA.append(np.mean(ndmiReal))
                count += 1
            except StopIteration:
                break
        while count <= 1000:
            try:
                print('loading image: ', count)
                landsat_batch_x_cloudy, _, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, _, _ = test_itr2.__next__()

                fakeImg = (self.dataloader.denormalize11(
                    self.generator2.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(
                    np.uint8)
                fakeImg = self.normal01(fakeImg)

                target_cloudfree_land = self.dataloader.denormalize11(landsat_batch_y_cloud_free[-1][:, :, :8]).astype(
                    np.uint8)  # Returns image pixels between (0, 255)
                target_cloudfree_land = self.normal01(target_cloudfree_land)
                cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)

                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        if not all(cloudy_img_land[i, j] == 0):
                            fakeImg[i, j] = cloudy_img_land[i, j, :]

                ndmiFake = (fakeImg[:, :, 3] - fakeImg[:, :, 4]) / (fakeImg[:, :, 3] + fakeImg[:, :, 4])
                ndmiReal = (target_cloudfree_land[:, :, 3] - target_cloudfree_land[:, :, 4]) / (
                        target_cloudfree_land[:, :, 3] + target_cloudfree_land[:, :, 4])
                dataP.append(np.mean(ndmiFake))
                dataA.append(np.mean(ndmiReal))
                count += 1
            except StopIteration:
                break

        if not self.isTrain:
            fileN = 'Testing_'
        else:
            fileN = 'Train_'
        with open(self.dirName + '/NDMI/ndmiA' + fileN + str(socket.gethostname()[-3:]) + '.txt', 'wb') as f:
            pickle.dump(dataA, f)

        with open(self.dirName + '/NDMI/ndmiP' + fileN + str(socket.gethostname()[-3:]) + '.txt', 'wb') as f:
            pickle.dump(dataP, f)
        # plt.style.use("ggplot")
        # coer,_ = pearsonr(dataA,dataP)
        # plt.scatter(dataA, dataP, label = 'Pearson CorrCoef: ' + str(round(coer, 4)), c='#44AA99', s=8)
        # plt.title('Correlation plot between NDVI index for\npredicted and actual cloud free image')
        # plt.ylabel('Predicted Average NDVI index')
        # plt.xlabel('Actual Average NDVI index')
        # plt.plot(np.unique(dataA), np.poly1d(np.polyfit(dataA, dataP, 1))(np.unique(dataA)), color='#999933')
        # plt.legend()
        # if self.isTrain:
        #     plt.savefig(self.dirName + 'NDVIComparisonTrain_' + str(socket.gethostname()[-3:]) +'.png')
        # else:
        #     plt.savefig(self.dirName + 'NDVIComparisonTest_' + str(socket.gethostname()[-3:]) + '.png')
        # plt.close()
        return

    def getErrorsNDSI(self):
        dataP = []
        dataA = []
        test_itr = self.dataloader.load_iterator_from_paths(batch_size=1, geohashes=self.geohashes1)
        test_itr2 = self.dataloader.load_iterator_from_paths(batch_size=1, geohashes=self.geohashes2)
        count = 0
        while count <= 500:
            try:
                print('loading image: ', count)
                landsat_batch_x_cloudy, _, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, _, _ = test_itr.__next__()

                fakeImg = (self.dataloader.denormalize11(
                    self.generator.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(
                    np.uint8)
                fakeImg = self.normal01(fakeImg)

                target_cloudfree_land = self.dataloader.denormalize11(landsat_batch_y_cloud_free[-1][:, :, :8]).astype(
                    np.uint8)  # Returns image pixels between (0, 255)
                target_cloudfree_land = self.normal01(target_cloudfree_land)
                cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)

                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        if not all(cloudy_img_land[i, j] == 0):
                            fakeImg[i, j] = cloudy_img_land[i, j, :]

                ndsiFake = (fakeImg[:, :, 1] - fakeImg[:, :, 4]) / (fakeImg[:, :, 1] + fakeImg[:, :, 4])
                ndsiReal = (target_cloudfree_land[:, :, 1] - target_cloudfree_land[:, :, 4]) / (
                        target_cloudfree_land[:, :, 1] + target_cloudfree_land[:, :, 4])
                dataP.append(np.mean(ndsiFake))
                dataA.append(np.mean(ndsiReal))

                count += 1
            except StopIteration:
                break
        while count <= 1000:
            try:
                print('loading image: ', count)
                landsat_batch_x_cloudy, _, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, _, _ = test_itr2.__next__()

                fakeImg = (self.dataloader.denormalize11(
                    self.generator2.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(
                    np.uint8)
                fakeImg = self.normal01(fakeImg)

                target_cloudfree_land = self.dataloader.denormalize11(landsat_batch_y_cloud_free[-1][:, :, :8]).astype(
                    np.uint8)  # Returns image pixels between (0, 255)
                target_cloudfree_land = self.normal01(target_cloudfree_land)
                cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)

                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        if not all(cloudy_img_land[i, j] == 0):
                            fakeImg[i, j] = cloudy_img_land[i, j, :]

                ndsiFake = (fakeImg[:, :, 1] - fakeImg[:, :, 4]) / (fakeImg[:, :, 1] + fakeImg[:, :, 4])
                ndsiReal = (target_cloudfree_land[:, :, 1] - target_cloudfree_land[:, :, 4]) / (
                        target_cloudfree_land[:, :, 1] + target_cloudfree_land[:, :, 4])
                dataP.append(np.mean(ndsiFake))
                dataA.append(np.mean(ndsiReal))

                count += 1
            except StopIteration:
                break

        if not self.isTrain:
            fileN = 'Testing_'
        else:
            fileN = 'Train_'
        with open(self.dirName + '/NDSI/ndsiA' + fileN + str(socket.gethostname()[-3:]) + '.txt', 'wb') as f:
            pickle.dump(dataA, f)

        with open(self.dirName + '/NDSI/ndsiP' + fileN + str(socket.gethostname()[-3:]) + '.txt', 'wb') as f:
            pickle.dump(dataP, f)
        # plt.style.use("ggplot")
        # coer,_ = pearsonr(dataA,dataP)
        # plt.scatter(dataA, dataP, label = 'Pearson CorrCoef: ' + str(round(coer, 4)), c='#44AA99', s=8)
        # plt.title('Correlation plot between NDVI index for\npredicted and actual cloud free image')
        # plt.ylabel('Predicted Average NDVI index')
        # plt.xlabel('Actual Average NDVI index')
        # plt.plot(np.unique(dataA), np.poly1d(np.polyfit(dataA, dataP, 1))(np.unique(dataA)), color='#999933')
        # plt.legend()
        # if self.isTrain:
        #     plt.savefig(self.dirName + 'NDVIComparisonTrain_' + str(socket.gethostname()[-3:]) +'.png')
        # else:
        #     plt.savefig(self.dirName + 'NDVIComparisonTest_' + str(socket.gethostname()[-3:]) + '.png')
        # plt.close()
        return

    def getErrorsARVI(self):
        dataP = []
        dataA = []
        test_itr = self.dataloader.load_iterator_from_paths(batch_size=1, geohashes=self.geohashes1)
        test_itr2 = self.dataloader.load_iterator_from_paths(batch_size=1, geohashes=self.geohashes2)
        count = 0
        while count <= 500:
            try:
                print('loading image: ', count)
                landsat_batch_x_cloudy, _, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, _, _ = test_itr.__next__()

                fakeImg = (self.dataloader.denormalize11(
                    self.generator.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(
                    np.uint8)
                fakeImg = self.normal01(fakeImg)

                target_cloudfree_land = self.dataloader.denormalize11(landsat_batch_y_cloud_free[-1][:, :, :8]).astype(
                    np.uint8)  # Returns image pixels between (0, 255)
                target_cloudfree_land = self.normal01(target_cloudfree_land)
                cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)

                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        if not all(cloudy_img_land[i, j] == 0):
                            fakeImg[i, j] = cloudy_img_land[i, j, :]

                arviFake = (fakeImg[:, :, 3] - (2 * fakeImg[:, :, 0]) + fakeImg[:, :, 2]) / (
                            fakeImg[:, :, 3] + (2 * fakeImg[:, :, 0]) + fakeImg[:, :, 2])
                arviReal = (target_cloudfree_land[:, :, 3] - (
                            2 * target_cloudfree_land[:, :, 0]) + target_cloudfree_land[:, :, 2]) / (
                                       target_cloudfree_land[:, :, 3] + (
                                           2 * target_cloudfree_land[:, :, 0]) + target_cloudfree_land[:, :, 2])

                dataP.append(np.mean(arviFake))
                dataA.append(np.mean(arviReal))

                count += 1
            except StopIteration:
                break
        while count <= 1000:
            try:
                print('loading image: ', count)
                landsat_batch_x_cloudy, _, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, _, _ = test_itr2.__next__()

                fakeImg = (self.dataloader.denormalize11(
                    self.generator2.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(
                    np.uint8)
                fakeImg = self.normal01(fakeImg)

                target_cloudfree_land = self.dataloader.denormalize11(landsat_batch_y_cloud_free[-1][:, :, :8]).astype(
                    np.uint8)  # Returns image pixels between (0, 255)
                target_cloudfree_land = self.normal01(target_cloudfree_land)
                cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)

                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        if not all(cloudy_img_land[i, j] == 0):
                            fakeImg[i, j] = cloudy_img_land[i, j, :]

                arviFake = (fakeImg[:, :, 3] - (2 * fakeImg[:, :, 0]) + fakeImg[:, :, 2]) / (
                            fakeImg[:, :, 3] + (2 * fakeImg[:, :, 0]) + fakeImg[:, :, 2])
                arviReal = (target_cloudfree_land[:, :, 3] - (
                            2 * target_cloudfree_land[:, :, 0]) + target_cloudfree_land[:, :, 2]) / (
                                       target_cloudfree_land[:, :, 3] + (
                                           2 * target_cloudfree_land[:, :, 0]) + target_cloudfree_land[:, :, 2])

                dataP.append(np.mean(arviFake))
                dataA.append(np.mean(arviReal))
                count += 1
            except StopIteration:
                break

        if not self.isTrain:
            fileN = 'Testing_'
        else:
            fileN = 'Train_'
        with open(self.dirName + '/ARVI/arviA' + fileN + str(socket.gethostname()[-3:]) + '.txt', 'wb') as f:
            pickle.dump(dataA, f)

        with open(self.dirName + '/ARVI/arviP' + fileN + str(socket.gethostname()[-3:]) + '.txt', 'wb') as f:
            pickle.dump(dataP, f)
        # plt.style.use("ggplot")
        # coer,_ = pearsonr(dataA,dataP)
        # plt.scatter(dataA, dataP, label = 'Pearson CorrCoef: ' + str(round(coer, 4)), c='#44AA99', s=8)
        # plt.title('Correlation plot between NDVI index for\npredicted and actual cloud free image')
        # plt.ylabel('Predicted Average NDVI index')
        # plt.xlabel('Actual Average NDVI index')
        # plt.plot(np.unique(dataA), np.poly1d(np.polyfit(dataA, dataP, 1))(np.unique(dataA)), color='#999933')
        # plt.legend()
        # if self.isTrain:
        #     plt.savefig(self.dirName + 'NDVIComparisonTrain_' + str(socket.gethostname()[-3:]) +'.png')
        # else:
        #     plt.savefig(self.dirName + 'NDVIComparisonTest_' + str(socket.gethostname()[-3:]) + '.png')
        # plt.close()
        return

    def getErrorsGSI(self):
        dataP = []
        dataA = []
        test_itr = self.dataloader.load_iterator_from_paths(batch_size=1, geohashes=self.geohashes1)
        test_itr2 = self.dataloader.load_iterator_from_paths(batch_size=1, geohashes=self.geohashes2)
        count = 0
        while count <= 500:
            try:
                print('loading image: ', count)
                landsat_batch_x_cloudy, _, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, _, _ = test_itr.__next__()

                fakeImg = (self.dataloader.denormalize11(
                    self.generator.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(
                    np.uint8)
                fakeImg = self.normal01(fakeImg)

                target_cloudfree_land = self.dataloader.denormalize11(landsat_batch_y_cloud_free[-1][:, :, :8]).astype(
                    np.uint8)  # Returns image pixels between (0, 255)
                target_cloudfree_land = self.normal01(target_cloudfree_land)
                cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)

                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        if not all(cloudy_img_land[i, j] == 0):
                            fakeImg[i, j] = cloudy_img_land[i, j, :]

                gciFake = (fakeImg[:, :, 3] / (fakeImg[:, :, 1])) - 1
                gciReal = (target_cloudfree_land[:, :, 3] / target_cloudfree_land[:, :, 1]) - 1

                dataP.append(np.mean(gciFake))
                dataA.append(np.mean(gciReal))

                count += 1
            except StopIteration:
                break
        while count <= 1000:
            try:
                print('loading image: ', count)
                landsat_batch_x_cloudy, _, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, _, _ = test_itr2.__next__()

                fakeImg = (self.dataloader.denormalize11(
                    self.generator2.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(
                    np.uint8)
                fakeImg = self.normal01(fakeImg)

                target_cloudfree_land = self.dataloader.denormalize11(landsat_batch_y_cloud_free[-1][:, :, :8]).astype(
                    np.uint8)  # Returns image pixels between (0, 255)
                target_cloudfree_land = self.normal01(target_cloudfree_land)
                cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)

                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        if not all(cloudy_img_land[i, j] == 0):
                            fakeImg[i, j] = cloudy_img_land[i, j, :]

                gciFake = (fakeImg[:, :, 3] / (fakeImg[:, :, 1])) - 1
                gciReal = (target_cloudfree_land[:, :, 3] / target_cloudfree_land[:, :, 1]) - 1

                dataP.append(np.mean(gciFake))
                dataA.append(np.mean(gciReal))
                count += 1
            except StopIteration:
                break

        if not self.isTrain:
            fileN = 'Testing_'
        else:
            fileN = 'Train_'
        with open(self.dirName + '/GSI/gsiA' + fileN + str(socket.gethostname()[-3:]) + '.txt', 'wb') as f:
            pickle.dump(dataA, f)

        with open(self.dirName + '/GSI/gsiP' + fileN + str(socket.gethostname()[-3:]) + '.txt', 'wb') as f:
            pickle.dump(dataP, f)
        # plt.style.use("ggplot")
        # coer,_ = pearsonr(dataA,dataP)
        # plt.scatter(dataA, dataP, label = 'Pearson CorrCoef: ' + str(round(coer, 4)), c='#44AA99', s=8)
        # plt.title('Correlation plot between NDVI index for\npredicted and actual cloud free image')
        # plt.ylabel('Predicted Average NDVI index')
        # plt.xlabel('Actual Average NDVI index')
        # plt.plot(np.unique(dataA), np.poly1d(np.polyfit(dataA, dataP, 1))(np.unique(dataA)), color='#999933')
        # plt.legend()
        # if self.isTrain:
        #     plt.savefig(self.dirName + 'NDVIComparisonTrain_' + str(socket.gethostname()[-3:]) +'.png')
        # else:
        #     plt.savefig(self.dirName + 'NDVIComparisonTest_' + str(socket.gethostname()[-3:]) + '.png')
        # plt.close()
        return

    def getErrorsNDWI(self):
        if self.isTrain:
            startT = '2020-05-01'
            endT = '2020-11-01'
        else:
            startT = '2020-04-01'
            endT = '2020-05-01'
        dataloader = data_loader_single_mask.DatasetHandling(128, 128, folderI=self.folderI, startT=startT, endT=endT,
                                                             album='iowa-2015-2020-spa',
                                                             no_of_timesteps=self.timestamps,
                                                             batch_size=1, istrain=self.isTrain)

        dataP = []
        dataA = []
        test_itr = dataloader.load_landsat_images(batch_size=1, all_black_clouds=True)

        while True:
            try:
                (cloudy_img_land, target_cloudfree_land, inp_prev_sent, inp_prev_landsat, _) = next(test_itr)
                fakeImg = (dataloader.denormalize11(
                    self.generator.predict([cloudy_img_land, inp_prev_landsat, inp_prev_sent])[-1])).astype(np.uint8)
                fakeImg = self.normal01(fakeImg)

                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        if not all(cloudy_img_land[i, j] == 0):
                            fakeImg[i, j] = cloudy_img_land[i, j, :]

                target_cloudfree_land = dataloader.denormalize11(target_cloudfree_land[-1][:, :, :8]).astype(
                    np.uint8)  # Returns image pixels between (0, 255)
                target_cloudfree_land = self.normal01(target_cloudfree_land)

                gciReal = ((target_cloudfree_land[:, :, 3] - target_cloudfree_land[:, :, 4]) / (
                            target_cloudfree_land[:, :, 3] + target_cloudfree_land[:, :, 4]))
                gciFake = ((fakeImg[:, :, 3] - fakeImg[:, :, 4]) / (fakeImg[:, :, 3] + fakeImg[:, :, 4]))

                dataP.append(np.mean(gciFake))
                dataA.append(np.mean(gciReal))


            except StopIteration:
                break

        with open(self.dirName + 'gciATesting_' + str(socket.gethostname()[-3:]) + '.txt', 'wb') as f:
            pickle.dump(dataA, f)

        with open(self.dirName + 'gciPTesting_' + str(socket.gethostname()[-3:]) + '.txt', 'wb') as f:
            pickle.dump(dataP, f)

        coer, _ = pearsonr(dataA, dataP)
        plt.scatter(dataA, dataP, label='Pearson CorrCoef:' + str(round(coer, 4)), cmap='Greens')
        plt.style.use("ggplot")
        plt.title('Correlation plot between GCI index for\npredicted and actual cloud free image')
        plt.ylabel('Predicted Average GCI index')
        plt.xlabel('Actual Average GCI index')
        plt.plot(np.unique(dataA), np.poly1d(np.polyfit(dataA, dataP, 1))(np.unique(dataA)), color='yellow')
        plt.legend()
        plt.savefig(self.dirName + 'GCIComparisonTesting_' + str(socket.gethostname()[-3:]) + '.png')
        plt.close()
        return

    def PrintErrorsPerMonth(self):
        psnrsMay, psnrsJune, psnrsJuly, psnrsAugust, psnrsSept, psnrsOct, psnrsNov = [], [], [], [], [], [], []
        test_itr = self.dataloader.load_iterator_from_paths_with_complexity_time(batch_size=1,
                                                                                 geohashes=self.geohashes1)
        count = 0

        while True:
            try:
                print("Count : ", count)
                landsat_batch_x_cloudy, _, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, _, _, _, timeB = test_itr.__next__()

                fakeImg = (self.dataloader.denormalize11(
                    self.generator.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(
                    np.uint8)
                # fakeImg = self.normal01(fakeImg)
                imageMonth = int(timeB[-1].split("-")[1])
                target_cloudfree_land = self.dataloader.denormalize11(landsat_batch_y_cloud_free[-1][:, :, :8]).astype(
                    np.uint8)  # Returns image pixels between (0, 255)
                # target_cloudfree_land = self.normal01(target_cloudfree_land)
                cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)

                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        if not all(cloudy_img_land[i, j] == 0):
                            fakeImg[i, j] = cloudy_img_land[i, j, :]

                bandDic = {5: psnrsMay, 6: psnrsJune, 7: psnrsJuly, 8: psnrsAugust, 9: psnrsSept, 10: psnrsOct,
                           11: psnrsNov}

                oldpsnr = bandDic.get(imageMonth)
                psnr = tf.image.psnr(fakeImg, target_cloudfree_land, 255).numpy()
                if np.isinf(psnr) or psnr > 48:
                    psnr = 48.0

                oldpsnr.append(psnr)
                bandDic[imageMonth] = oldpsnr

                count += 1

            except StopIteration:
                break

        test_itr2 = self.dataloader.load_iterator_from_paths_with_complexity_time(batch_size=1,
                                                                                  geohashes=self.geohashes2)
        while True:
            try:
                print("Count : ", count)
                landsat_batch_x_cloudy, _, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, _, _, _, timeB = test_itr2.__next__()

                fakeImg = (self.dataloader.denormalize11(
                    self.generator2.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(
                    np.uint8)
                # fakeImg = self.normal01(fakeImg)
                imageMonth = int(timeB[-1].split("-")[1])
                target_cloudfree_land = self.dataloader.denormalize11(
                    landsat_batch_y_cloud_free[-1][:, :, :8]).astype(
                    np.uint8)  # Returns image pixels between (0, 255)
                # target_cloudfree_land = self.normal01(target_cloudfree_land)
                cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)

                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        if not all(cloudy_img_land[i, j] == 0):
                            fakeImg[i, j] = cloudy_img_land[i, j, :]

                bandDic = {5: psnrsMay, 6: psnrsJune, 7: psnrsJuly, 8: psnrsAugust, 9: psnrsSept, 10: psnrsOct,
                           11: psnrsNov}

                oldpsnr = bandDic.get(imageMonth)
                psnr = tf.image.psnr(fakeImg, target_cloudfree_land, 255).numpy()
                if np.isinf(psnr) or psnr > 48:
                    psnr = 48.0

                oldpsnr.append(psnr)
                bandDic[imageMonth] = oldpsnr
                count += 1
            except StopIteration:
                break

        bandsN = ['May', 'June', 'July', 'August', 'September', 'October', 'November']

        if not self.isTrain:
            fileN = 'Testing_'
        else:
            fileN = 'Train_'

        allPsnr = [psnrsMay, psnrsJune, psnrsJuly, psnrsAugust, psnrsSept, psnrsOct]
        with open(self.dirName + 'PerMonth/psnrPerMonth' + fileN + str(socket.gethostname()[-3:]) + '.txt', 'wb') as f:
            pickle.dump(allPsnr, f)

        # fig, ax = plt.subplots()
        # ax.grid(b=True, color='grey',
        #         linestyle='-.', linewidth=0.5,
        #         alpha=0.2)

        # meansE = [np.mean(psnrsMay), np.mean(psnrsJune), np.mean(psnrsJuly), np.mean(psnrsAugust), np.mean(psnrsSept),
        #           np.mean(psnrsOct), np.mean(psnrsNov)]
        # stdsE = [np.std(psnrsMay), np.std(psnrsJune), np.std(psnrsJuly), np.std(psnrsAugust), np.std(psnrsSept),
        #           np.std(psnrsOct), np.std(psnrsNov)]
        # ax.bar(np.arange(7), meansE, yerr=stdsE,align='center', alpha=0.5, ecolor='black', color='lightgrey', capsize=10)
        #
        # plt.xticks(np.arange(1, len(bandsN) + 1), bandsN)
        # plt.title("Accuracy per band")
        # plt.xlabel("Month")
        # plt.ylabel("Accuracy measured in PSNR (in dB)")
        # plt.savefig(self.dirName + 'LossPerMonthTraining_' +  str(socket.gethostname()[-3:]) + '.png')
        # plt.close()
        return

    def PrintErrorsPerband(self):
        psnrsRed, psnrsBlue, psnrsGreen, psnrsNIR, psnrsSWIR1, psnrsSWIR2, psnrsTIR1, psnrsTIR2 = [], [], [], [], [], [], [], []
        test_itr = self.dataloader.load_iterator_from_paths(batch_size=1, geohashes=self.geohashes1)
        count = 0

        while True:
            try:
                print("Count : ", count)
                landsat_batch_x_cloudy, _, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, _, _ = test_itr.__next__()

                fakeImg = (self.dataloader.denormalize11(
                    self.generator.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(
                    np.uint8)
                # fakeImg = self.normal01(fakeImg)

                target_cloudfree_land = self.dataloader.denormalize11(landsat_batch_y_cloud_free[-1][:, :, :8]).astype(
                    np.uint8)  # Returns image pixels between (0, 255)
                # target_cloudfree_land = self.normal01(target_cloudfree_land)
                cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)

                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        if not all(cloudy_img_land[i, j] == 0):
                            fakeImg[i, j] = cloudy_img_land[i, j, :]

                bandDic = {0: psnrsRed, 1: psnrsBlue, 2: psnrsGreen, 3: psnrsNIR, 4: psnrsSWIR1, 5: psnrsSWIR2,
                           6: psnrsTIR1,
                           7: psnrsTIR2}

                for i in range(0, len(bandDic)):
                    psnr = tf.image.psnr(np.reshape(fakeImg[:, :, i], (128, 128, 1)),
                                         np.reshape(target_cloudfree_land[:, :, i], (128, 128, 1)), 255).numpy()
                    if np.isinf(psnr) or psnr > 48:
                        psnr = 48.0
                    oldpsnr = bandDic.get(i)
                    oldpsnr.append(psnr)
                    bandDic[i] = oldpsnr

                count += 1

            except StopIteration:
                break

        test_itr2 = self.dataloader.load_iterator_from_paths(batch_size=1, geohashes=self.geohashes2)
        while True:
            try:
                print("Count : ", count)
                landsat_batch_x_cloudy, _, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, _, _ = test_itr2.__next__()

                fakeImg = (self.dataloader.denormalize11(
                    self.generator2.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(
                    np.uint8)
                # fakeImg = self.normal01(fakeImg)

                target_cloudfree_land = self.dataloader.denormalize11(
                    landsat_batch_y_cloud_free[-1][:, :, :8]).astype(
                    np.uint8)  # Returns image pixels between (0, 255)
                # target_cloudfree_land = self.normal01(target_cloudfree_land)
                cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)

                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        if not all(cloudy_img_land[i, j] == 0):
                            fakeImg[i, j] = cloudy_img_land[i, j, :]

                bandDic = {0: psnrsRed, 1: psnrsBlue, 2: psnrsGreen, 3: psnrsNIR, 4: psnrsSWIR1, 5: psnrsSWIR2,
                           6: psnrsTIR1, 7: psnrsTIR2}
                for i in range(0, len(bandDic)):
                    psnr = tf.image.psnr(np.reshape(fakeImg[:, :, i], (128, 128, 1)),
                                         np.reshape(target_cloudfree_land[:, :, i], (128, 128, 1)), 255).numpy()
                    if np.isinf(psnr) or psnr > 48:
                        psnr = 48.0
                    oldpsnr = bandDic.get(i)
                    oldpsnr.append(psnr)
                    bandDic[i] = oldpsnr
                count += 1
            except StopIteration:
                break

        allPsnr = [psnrsRed, psnrsBlue, psnrsGreen, psnrsNIR, psnrsSWIR1, psnrsSWIR2, psnrsTIR1, psnrsTIR2]
        bandsN = ['Red', 'Blue', 'Green', 'NIR', 'SWIR1', 'SWIR2', 'TIR1', 'TIR2']

        if not self.isTrain:
            fileN = 'Testing_'
        else:
            fileN = 'Train_'

        # with open(self.dirName  + 'PerBand/psnrPerBand' + fileN + str(socket.gethostname()[-3:]) + '.txt', 'wb') as f:
        #     pickle.dump(allPsnr, f)

        # plt.style.use("ggplot")

        fig, ax = plt.subplots()

        flierprops = dict(marker='+', markerfacecolor='black', markersize=4,
                          linestyle='none', markeredgecolor='black', color='black', alpha=0.5)
        ax.grid(b=True, color='grey',
                linestyle='-.', linewidth=0.5,
                alpha=0.2)
        ax.boxplot(allPsnr, notch=True, patch_artist=True, showfliers=True,
                   boxprops=dict(facecolor='grey', color='black'),
                   # capprops=dict(color=c),
                   whiskerprops=dict(color='black'),
                   flierprops=flierprops,

                   )

        plt.xticks(np.arange(1, len(bandsN) + 1), bandsN)
        plt.title("Accuracy per band")
        plt.xlabel("Spectral Bands")
        plt.ylabel("Accuracy measured in PSNR (in dB)")
        # plt.plot([1, 2, 3, 4, 5, 6, 7, 8],
        #          [np.mean(psnrsRed), np.mean(psnrsBlue), np.mean(psnrsGreen), np.mean(psnrsNIR), np.mean(psnrsSWIR1),
        #           np.mean(psnrsSWIR2), np.mean(psnrsTIR1), np.mean(psnrsTIR2)])
        plt.savefig(self.dirName + 'LossPerBandTraining_' + str(socket.gethostname()[-3:]) + '.png')
        plt.close()
        return

        # def PrintErrorsPerbandGlobal(self):
        psnrsRed, psnrsBlue, psnrsGreen, psnrsNIR, psnrsSWIR1, psnrsSWIR2, psnrsTIR1, psnrsTIR2 = [], [], [], [], [], [], [], []
        bandDic = {0: psnrsRed, 1: psnrsBlue, 2: psnrsGreen, 3: psnrsNIR, 4: psnrsSWIR1, 5: psnrsSWIR2,
                   6: psnrsTIR1, 7: psnrsTIR2}

        bandsN = ['Red', 'Blue', 'Green', 'NIR', 'SWIR1', 'SWIR2', 'TIR1', 'TIR2']

        for i in range(176, 210):
            if i in [177, 179, 182, 181, 189, 183, 187, 192, 195, 203, 208, 207]:
                continue
            if not self.isTrain:
                fileN = 'Testing_'
            else:
                fileN = 'Train_'

            with open(self.dirName + 'PerBand/psnrPerBand' + fileN + str(i) + '.txt', 'rb') as f:
                psnrallpermachine = pickle.load(f)

            for j in range(0, len(bandDic)):
                oldpsnr = bandDic.get(j)
                newPsnr = np.array(psnrallpermachine[j])
                # newPsnr = newPsnr.clip(20, 48)
                oldpsnr.extend(newPsnr)
                bandDic[j] = oldpsnr

        allPsnr = [psnrsRed, psnrsBlue, psnrsGreen, psnrsNIR, psnrsSWIR1, psnrsSWIR2, psnrsTIR1, psnrsTIR2]

        if self.isTrain:
            with open(self.dirName + 'psnrPerBandTrainGlobal.txt', 'wb') as f:
                pickle.dump(allPsnr, f)
        else:
            with open(self.dirName + 'psnrPerBandTestGlobal.txt', 'wb') as f:
                pickle.dump(allPsnr, f)

        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        flierprops = dict(marker='+', markerfacecolor='r', markersize=4,
                          linestyle='none', markeredgecolor='black', color='r', alpha=0.2)
        # plt.ylim(ymax=50, ymin=15)
        ax.boxplot(allPsnr, notch=True, patch_artist=False, showfliers=True,
                   # boxprops=dict(facecolor='999933', color=c),
                   # capprops=dict(color=c),
                   whiskerprops=dict(color='black'),
                   flierprops=flierprops,
                   )
        plt.xticks(np.arange(1, len(bandsN) + 1), bandsN)
        plt.title("PSNR accuracy per band")
        plt.xlabel("Spectral Bands")
        plt.ylabel("PSNR Measure")

        plt.savefig(self.dirName + 'LossPerBand' + fileN + 'Global' + '.png')
        plt.close()
        return

    def PrintErrorsPerMonthGlobal(self):
        psnrsMay, psnrsJune, psnrsJuly, psnrsAugust, psnrsSept, psnrsOct = [], [], [], [], [], []
        bandDic = {5: psnrsMay, 6: psnrsJune, 7: psnrsJuly, 8: psnrsAugust, 9: psnrsSept, 10: psnrsOct}
        for i in range(176, 210):
            if i in [177, 179, 181, 182, 184, 186, 187, 189, 192, 193, 194, 195, 200, 202, 203, 205, 208, 207]:
                continue
            if not self.isTrain:
                fileN = 'Testing_'
            else:
                fileN = 'Train_'

            with open(self.dirName + 'PerMonth/psnrPerMonth' + fileN + str(i) + '.txt', 'rb') as f:
                psnrallpermachine = pickle.load(f)

            # print("lEN: ", len(psnrallpermachine))
            for j in range(0, len(bandDic)):
                oldpsnr = bandDic.get(j + 5)
                newPsnr = np.array(psnrallpermachine[j])
                oldpsnr.extend(newPsnr)
                bandDic[j + 5] = oldpsnr

        bandsN = ['May', 'June', 'July', 'August', 'September', 'October']

        if not self.isTrain:
            fileN = 'Testing_'
        else:
            fileN = 'Train_'

        allPsnr = [psnrsMay, psnrsJune, psnrsJuly, psnrsAugust, psnrsSept, psnrsOct]
        # with open(self.dirName + 'PerMonth/psnrPerMonth' + fileN + str(socket.gethostname()[-3:]) + '.txt', 'wb') as f:
        #     pickle.dump(allPsnr, f)

        # plt.style.use("ggplot")

        fig, ax = plt.subplots()
        ax.grid(b=True, color='grey',
                linestyle='-.', linewidth=0.5,
                alpha=0.2)

        meansE = [np.mean(psnrsMay), np.mean(psnrsJune), np.mean(psnrsJuly), np.mean(psnrsAugust), np.mean(psnrsSept),
                  np.mean(psnrsOct)]
        stdsE = [np.std(psnrsMay), np.std(psnrsJune), np.std(psnrsJuly), np.std(psnrsAugust), np.std(psnrsSept),
                 np.std(psnrsOct)]
        ax.bar(np.arange(6), meansE, yerr=stdsE, align='center', alpha=0.8, ecolor='black', color='darkgrey',
               capsize=10)

        plt.xticks(np.arange(0, len(bandsN)), bandsN)
        plt.title("Accuracy over each month")
        plt.xlabel("Month")
        plt.ylabel("Average accuracy measured in PSNR (in dB)")
        plt.savefig(self.dirName + 'LossPerMonthTrainingGlobal.png')
        plt.close()
        return

    def PrintErrorsIndexes(self, index_name=''):
        psnrsA, psnrsP = [], []
        for i in range(176, 210):
            if i in [177, 179, 182, 181, 189, 183, 187, 192, 195, 203, 208, 207]:
                continue

            if not self.isTrain:
                fn = 'Testing_'
            else:
                fn = 'Train_'

            with open(self.dirName + index_name + "/" + index_name.lower() + 'P' + fn + str(i) + '.txt', 'rb') as f:
                localPsnrPr = pickle.load(f)
            with open(self.dirName + index_name + "/" + index_name.lower() + 'A' + fn + str(i) + '.txt', 'rb') as f:
                localPsnrAc = pickle.load(f)
            psnrsP.extend(np.array(localPsnrPr))
            psnrsA.extend(np.array(localPsnrAc))

        plt.style.use("ggplot")
        coer, _ = pearsonr(psnrsA, psnrsP)
        plt.scatter(psnrsA, psnrsP, label='Pearson CorrCoef: ' + str(round(coer, 4)), c='#44AA99', s=8)
        plt.title('Correlation plot between ' + index_name + ' index for\npredicted and actual cloud free image')
        plt.ylabel('Predicted Average ' + index_name + ' index')
        plt.xlabel('Actual Average ' + index_name + ' index')
        plt.plot(np.unique(psnrsA), np.poly1d(np.polyfit(psnrsA, psnrsP, 1))(np.unique(psnrsA)), color='#999933')
        plt.legend()
        if self.isTrain:
            plt.savefig(self.dirName + index_name + 'ComparisonTrainGlobal_' + str(socket.gethostname()[-3:]) + '.png')
        else:
            plt.savefig(self.dirName + index_name + 'ComparisonTestGlobal_' + str(socket.gethostname()[-3:]) + '.png')
        plt.close()
        return

    def justSaveImage(self, geohashes=None):
        psnrsRGB, psnrsAll = [], []
        test_itr = self.dataloader.load_iterator_from_paths(batch_size=1, geohashes=self.geohashes1)
        count = 1
        FinalPSNR, FinalSSIM, Finalsam, Finalfsim, FinalPSNRb, Finalrase, Finalergas, FinalrmseSW, Finaluqi, Finalmse, Finalrmse, Finalscc, Finalmsssim, Finalvifp = [], [], [], [], [], [], [], [], [], [], [], [], [], []

        while True:
            try:
                print("Displaying count: ", count)
                landsat_batch_x_cloudy, land_cloudy_with_clouds_batch, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, geo_batch, cc_batch = test_itr.__next__()
                # count += 1
                # if float(cc_batch[0]) * 100 > 60 or float(cc_batch[0]) * 100 < 40:
                #     continue

                self.display_training_images(count, landsat_batch_x_cloudy, land_cloudy_with_clouds_batch,
                                             landsat_batch_y_cloud_free,
                                             sent_batch, landsat_prev_batch, geo_batch, cc_batch, is_train=self.isTrain,
                                             is1005=True)
                count += 1
                continue
                # if self.isRes:
                #     fakeImg = (self.dataloader.denormalize11(
                #         self.generator.predict(landsat_batch_x_cloudy)[-1])).astype(np.uint8)
                # else:
                #     fakeImg = (self.dataloader.denormalize11(
                #         self.generator.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(
                #         np.uint8)
                #     cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)
                #
                #     for i in range(0, self.targetH):
                #         for j in range(0, self.targetW):
                #             if not all(cloudy_img_land[i, j] == 0):
                #                 fakeImg[i, j] = cloudy_img_land[i, j, :]
                #
                # target_cloudfree_land = self.dataloader.denormalize11(landsat_batch_y_cloud_free[-1][:, :, :8]).astype(
                #     np.uint8)
                #
                # psnrRGB = tf.image.psnr(fakeImg[:, :, :3], target_cloudfree_land[:, :, :3], 255).numpy()
                # psnrALL = tf.image.psnr(fakeImg, target_cloudfree_land, 255).numpy()
                #
                # if np.isinf(psnrALL):
                #     psnrALL = 48.0
                # if np.isinf(psnrRGB):
                #     psnrRGB = 48.0
                # print(psnrRGB, psnrALL)

                psnrsRGB.append(psnrRGB)
                psnrsAll.append(psnrALL)

                psnrSEWAR = sewar.full_ref.psnr(fakeImg[:, :, :3], target_cloudfree_land[:, :, :3], MAX=255)
                psnrBSEWAR = sewar.full_ref.psnrb(fakeImg[:, :, :3], target_cloudfree_land[:, :, :3])

                if np.isinf(psnrBSEWAR) and psnrBSEWAR > 48:
                    psnrBSEWAR = 48
                if np.isinf(psnrSEWAR) and psnrSEWAR > 48:
                    psnrSEWAR = 48
                FinalPSNR.append(psnrSEWAR)
                FinalPSNRb.append(psnrBSEWAR)
                Finalrase.append(sewar.full_ref.rase(fakeImg[:, :, :3], target_cloudfree_land[:, :, :3]))
                Finalmse.append(sewar.full_ref.mse(fakeImg[:, :, :3], target_cloudfree_land[:, :, :3]))
                FinalSSIM.append(sewar.full_ref.ssim(fakeImg[:, :, :3], target_cloudfree_land[:, :, :3], MAX=255))
                Finalergas.append(sewar.full_ref.ergas(fakeImg[:, :, :3], target_cloudfree_land[:, :, :3]))

            except StopIteration:
                break

        test_itr2 = self.dataloader.load_iterator_from_paths(batch_size=1, geohashes=self.geohashes2)
        while True:
            try:
                print("Displaying count: ", count)
                landsat_batch_x_cloudy, land_cloudy_with_clouds_batch, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, geo_batch, cc_batch = test_itr2.__next__()
                count += 1
                # if float(cc_batch[0]) * 100 > 60 or float(cc_batch[0]) * 100 < 40:
                #     continue
                self.display_training_images(count, landsat_batch_x_cloudy, land_cloudy_with_clouds_batch,
                                             landsat_batch_y_cloud_free,
                                             sent_batch, landsat_prev_batch, geo_batch, cc_batch, is_train=self.isTrain,
                                             is1005=False)

                continue
                if self.isRes:
                    fakeImg = (self.dataloader.denormalize11(
                        self.generator2.predict(landsat_batch_x_cloudy)[-1])).astype(np.uint8)
                else:
                    fakeImg = (self.dataloader.denormalize11(
                        self.generator2.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[
                            -1])).astype(
                        np.uint8)
                    cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)

                    for i in range(0, self.targetH):
                        for j in range(0, self.targetW):
                            if not all(cloudy_img_land[i, j] == 0):
                                fakeImg[i, j] = cloudy_img_land[i, j, :]

                target_cloudfree_land = self.dataloader.denormalize11(
                    landsat_batch_y_cloud_free[-1][:, :, :8]).astype(
                    np.uint8)

                psnrRGB = tf.image.psnr(fakeImg[:, :, :3], target_cloudfree_land[:, :, :3], 255).numpy()
                psnrALL = tf.image.psnr(fakeImg, target_cloudfree_land, 255).numpy()

                if np.isinf(psnrALL):
                    psnrALL = 48.0
                if np.isinf(psnrRGB):
                    psnrRGB = 48.0
                count += 1
                psnrsRGB.append(psnrRGB)
                psnrsAll.append(psnrALL)

                psnrSEWAR = sewar.full_ref.psnr(fakeImg[:, :, :3], target_cloudfree_land[:, :, :3], MAX=255)
                psnrBSEWAR = sewar.full_ref.psnrb(fakeImg[:, :, :3], target_cloudfree_land[:, :, :3])

                if np.isinf(psnrBSEWAR) and psnrBSEWAR > 48:
                    psnrBSEWAR = 48
                if np.isinf(psnrSEWAR) and psnrSEWAR > 48:
                    psnrSEWAR = 48
                FinalPSNR.append(psnrSEWAR)
                FinalPSNRb.append(psnrBSEWAR)

                Finalrase.append(sewar.full_ref.rase(fakeImg[:, :, :3], target_cloudfree_land[:, :, :3]))
                Finalmse.append(sewar.full_ref.mse(fakeImg[:, :, :3], target_cloudfree_land[:, :, :3]))
                FinalSSIM.append(sewar.full_ref.ssim(fakeImg[:, :, :3], target_cloudfree_land[:, :, :3], MAX=255))
                Finalergas.append(sewar.full_ref.ergas(fakeImg[:, :, :3], target_cloudfree_land[:, :, :3]))

            except StopIteration:
                break

        # if len(psnrsRGB) is not 0:
        #     print("Average psnr RGB band: ", np.mean(np.array(psnrsRGB)))
        #     print("Average psnr All bands: ", np.mean(np.array(psnrsAll)))
        # else:
        #     print("No image found")

        print("\n\n\n\n"
              " PSNR: ", round(np.mean(np.array(FinalPSNR)), 3),
              "\nPSNRb: ", round(np.mean(np.array(FinalPSNRb)), 3),
              "\nrase: ", round(np.mean(np.array(Finalrase)), 3),
              # "\nmse: ", round(np.mean(np.array(Finalmse)), 9)/255,
              # "\nSSIM: ", round(np.mean(np.array(FinalSSIM)), 3),
              "\nergas: ", round(np.mean(np.array(Finalergas)), 3),
              )

    def getComplexityPSNR(self):
        test_itr = self.dataloader.load_iterator_from_paths_with_complexity_time(batch_size=1,
                                                                                 geohashes=self.geohashes1)
        complxArr, psnrs = [], []
        count = 0
        while count < 10:
            try:
                landsat_batch_x_cloudy, land_cloudy_with_clouds_batch, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, geo_batch, cc_batch, complexity_batch, _ = test_itr.__next__()

                fakeImg = (self.dataloader.denormalize11(
                    self.generator.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(
                    np.uint8)
                cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)

                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        if not all(cloudy_img_land[i, j] == 0):
                            fakeImg[i, j] = cloudy_img_land[i, j, :]

                target_cloudfree_land = self.dataloader.denormalize11(landsat_batch_y_cloud_free[-1][:, :, :8]).astype(
                    np.uint8)

                psnr = tf.image.psnr(fakeImg, target_cloudfree_land, 255).numpy()
                print(count)
                psnrs.append(psnr)
                complxArr.append(float(complexity_batch[-1]))
                count += 1
            except StopIteration:
                break

        test_itr2 = self.dataloader.load_iterator_from_paths_with_complexity_time(batch_size=1,
                                                                                  geohashes=self.geohashes2)
        while count < 40:
            try:
                landsat_batch_x_cloudy, land_cloudy_with_clouds_batch, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, geo_batch, cc_batch, complexity_batch, _ = test_itr2.__next__()

                fakeImg = (self.dataloader.denormalize11(
                    self.generator2.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(
                    np.uint8)
                cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)

                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        if not all(cloudy_img_land[i, j] == 0):
                            fakeImg[i, j] = cloudy_img_land[i, j, :]

                target_cloudfree_land = self.dataloader.denormalize11(landsat_batch_y_cloud_free[-1][:, :, :8]).astype(
                    np.uint8)

                psnr = tf.image.psnr(fakeImg, target_cloudfree_land, 255).numpy()
                # print("Complexity ratio : ", float(complexity_batch[-1]), " PSNR: ", psnr)
                print(count)
                psnrs.append(psnr)
                complxArr.append(float(complexity_batch[-1]))
                count = count + 1
            except StopIteration:
                break

        # if self.isTrain:
        #     with open(self.dirName + 'complexityTrain.txt', 'wb') as f:
        #         pickle.dump(complxArr, f)
        #     with open(self.dirName + 'complexityTrainPSNR.txt', 'wb') as f:
        #         pickle.dump(psnrs, f)
        # else:
        #     with open(self.dirName + 'complexityTest.txt', 'wb') as f:
        #         pickle.dump(complxArr, f)
        #     with open(self.dirName + 'complexityTestPSNR.txt', 'wb') as f:
        #         pickle.dump(psnrs, f)

        print("Min Max COMPEXITY: ", min(complxArr), max(complxArr))

    def autolabel(self, rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0, 'right': 1, 'left': -1}

        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                        textcoords="offset points",  # in both directions
                        ha=ha[xpos], va='bottom')

    def plotGlobalPerCloudErrors(self):
        men_means, men_std = (20, 35, 30, 35, 27), (2, 3, 4, 1, 2)
        women_means, women_std = (25, 32, 34, 20, 25), (3, 5, 2, 3, 3)

        ind = np.arange(len(men_means))  # the x locations for the groups
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind - width / 2, men_means, width, yerr=men_std,
                        label='Men')
        rects2 = ax.bar(ind + width / 2, women_means, width, yerr=women_std,
                        label='Women')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title('Scores by group and gender')
        ax.set_xticks(ind)
        ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
        ax.legend()

        self.autolabel(rects1, "left")
        self.autolabel(rects2, "right")

        fig.tight_layout()

        plt.show()

        return

def loadplotlosses(dirName1="1000", dirName2="1001", dirName3="1008"):
        with open("/s/lattice-176/a/nobackup/galileo/paahuni/cloud_removal/" + dirName1 + '/trainLoss.txt', 'rb') as fp:
            losses1 = pickle.load(fp)
        with open("/s/lattice-176/a/nobackup/galileo/paahuni/cloud_removal/" + dirName2 + '/trainLoss.txt', 'rb') as fp:
            losses2 = pickle.load(fp)
        with open("/s/lattice-176/a/nobackup/galileo/paahuni/cloud_removal/" + dirName3 + '/trainLoss.txt', 'rb') as fp:
            losses3 = pickle.load(fp)
        losses = losses1 + losses2 + losses3

        plt.rcParams["figure.figsize"] = (18, 10)
        N = np.arange(0, len(losses))
        plt.style.use("ggplot")
        param_range = np.arange(1, len(losses), 2)
        plt.plot(N, np.multiply(losses,1.7), 'o-',  markersize=3, alpha=0.6, label='Training Loss', color='#44AA99')
        test_mean = np.mean(losses, axis=0)
        test_std = np.std(losses, axis=0)
        plt.fill_between(losses, test_mean - test_std, test_mean + test_std, alpha=0.2, color="#88CCEE")

        plt.title("Loss while training the model using DHTL", fontsize=25)
        plt.xlabel("Number of Epochs", fontsize=25)
        plt.ylabel("Mean Squared Errors", fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        # plt.legend()
        plt.legend(fontsize=15)
        plt.savefig('./FinalResulttrainTestLossFinal.png')
        plt.close()

        # plt.rcParams["figure.figsize"] = (18, 10)
        # N = np.arange(0, len(prnrL))
        # plt.style.use("ggplot")
        # plt.figure()
        # plt.plot(N, prnrL, 'o-', markersize=3, alpha=0.6, label='Training PSNR Accuracy')
        # # plt.plot(N, lossesT,'*--',markersize=1, alpha=0.6, label='Testing Loss')
        # plt.title("PSNR value while training the model", fontsize=25)
        # plt.xlabel("Number of Epochs", fontsize=20)
        # plt.ylabel("Accuracy in PSNR (dB)", fontsize=20)
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # plt.legend()
        # plt.legend(fontsize=15)
        # plt.savefig(self.dirName + 'trainTestPSNR.png')
        # plt.close()
        return losses


if __name__ == '__main__':
    tranferLear = 1005
    loadplotlosses()
    # train_helpers = Prediction_Class(folderI=50, batch_size=1, lossMethod='mse',
    #                                  istrain=True, tranferLear=tranferLear, loadModel=True, isRes=False)
    #
    # print("Data < 40")
    # train_helpers.justSaveImage()
    # train_helpers.PrintErrorsIndexes(index_name='GSI')
    # train_helpers.getErrorsGSI()
    # train_helpers.PrintErrorsPerband()
    # train_helpers.PrintErrorsPerMonth()
    # train_helpers.PrintErrorsPerbandGlobal()
    # train_helpers.PrintErrorsPerMonthGlobal()
    # train_helpers.getComplexityPSNR()
