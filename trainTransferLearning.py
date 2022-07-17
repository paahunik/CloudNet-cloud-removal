import math
import tensorflow as tf
import horovod.tensorflow.keras as hvd
hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
from cloud_removal import data_loader_single_mask, model_helpers
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, model_from_json,Model
from tensorflow.keras.layers import Dense,Lambda, Dropout, Flatten,Conv2D, MaxPooling2D, ReLU,\
    Activation,Dropout, RepeatVector, Input,Add,\
    Conv2DTranspose, LeakyReLU,Reshape, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from scipy.stats.stats import pearsonr
import os
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import socket
import gdal
import datetime
import pickle
from tensorflow.keras.callbacks import LambdaCallback

class train_helpers():
    def __init__(self, folderI=6, timestamps=1, batch_size=1, istrain=True, h=128, w=128, lossMethod='mse', onlyPredict=False, tranferLear = None, loadModel=True):
        self.folderI = folderI
        self.batch_size = batch_size
        self.dataloader = data_loader_single_mask.DatasetHandling(h, w,folderI=self.folderI,
                         album='iowa-2015-2020-spa', no_of_timesteps=timestamps, batch_size=batch_size, istrain=istrain,
                                                                  saveInputMetaData=False
                                                                  )
        self.model_helper = model_helpers.cloud_removal_models(timeStep=timestamps, batch_size=batch_size, w=128,h=128)
        self.timestamps = timestamps

        self.isTrain = istrain
        self.dirName = "/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/" + str(self.folderI)  +"/"
        self.loadModel = loadModel
        if not os.path.isdir("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/" ):
            os.mkdir("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/")
        if not os.path.isdir(self.dirName):
            os.mkdir(self.dirName)

        self.test_dataset_obj = data_loader_single_mask.DatasetHandling(h, w,folderI=self.folderI,
                                                             album='iowa-2015-2020-spa', no_of_timesteps=timestamps,
                                                             batch_size=5, istrain=False, saveInputMetaData = False)
        self.lossMethod = lossMethod
        self.targetH = h
        self.targetW = w
        self.targetShape = (self.targetW, self.targetH, 8)
        self.inputShape = (self.targetW, self.targetH, 8)
        self.sentShape = (self.targetH, self.targetW, 8)
        self.gan_optimizer = hvd.DistributedOptimizer(tf.keras.optimizers.Nadam(0.0001 * hvd.size()))
        # optimizer = hvd.DistributedOptimizer(Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
        #                                      backward_passes_per_step=1,
        #                                      average_aggregated_gradients=True)

        self.tranferLear = tranferLear

        self.onlyPredict = onlyPredict
        if tranferLear is not None:
            self.generator = self.load_model(tranferLear)
        elif self.onlyPredict:
            self.generator = self.load_model()
        else:
            self.generator = self.getGeneratorModel2()


    def getNeigbhours(self, input_machine):
        closest = []
        machineId = socket.gethostname().split("-")[-1]
        if machineId == '178':
            closest = ['lattice-178','lattice-179', 'lattice-189','lattice-186']
        elif machineId == '179':
            closest = ['lattice-179', 'lattice-180','lattice-189', 'lattice-186', 'lattice-182']

        with open('/s/chopin/a/grad/paahuni/cl/hosts2', "w+") as f:
            for m in closest:
                f.write(m + ' slots=1\n')
            f.close()

        return closest

    def totalBatchSizeAndEpoch(self, level):
        batchS, epoch  = 0, 0
        if level == 1:
            batchS = 400
            epoch = 320
        if level == 2:
            batchS = 800
            epoch = 320
        if level == 3:
            batchS = 1200
            epoch = 320

    def getGeneratorModel1(self):
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
        merged = Concatenate()([model, prevLanFeature])

        model = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same")(merged)
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

        model = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        genmodel = Model([in_src_image,inpPrevLand,inpSentImage], [model], name="ConvModel")
        if self.lossMethod == 'w':
            genmodel.compile(loss=self.weightedLoss, optimizer=self.gan_optimizer, experimental_run_tf_function=False, metrics=[self.psnrLossMetric, 'mse'])
        elif self.lossMethod == 'mse':
            genmodel.compile(loss='mse', optimizer=self.gan_optimizer, experimental_run_tf_function=False, metrics=[self.psnrLossMetric, 'mse'])

        # plot_model(genmodel, to_file='./Model3.png', rankdir='TB', show_shapes=True, show_layer_names=True)
        return genmodel

    def getGeneratorModel2(self):
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
        if self.lossMethod == 'w':
            genmodel.compile(loss=self.weightedLoss, optimizer=self.gan_optimizer, experimental_run_tf_function=False,
                             metrics=[self.psnrLossMetric, 'mse'])
        elif self.lossMethod == 'mse':
            genmodel.compile(loss='mse', optimizer=self.gan_optimizer, experimental_run_tf_function=False,
                             metrics=[self.psnrLossMetric, 'mse'])

        # plot_model(genmodel, to_file='./Model3.png', rankdir='TB', show_shapes=True, show_layer_names=True)
        return genmodel

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
        # model = BatchNormalization()(model)
        model = ReLU()(model)

        model = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same")(model)
        # model = BatchNormalization()(model)
        model = ReLU()(model)
        model = Dropout(0.2)(model)

        sentFeatures = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu')(inpSentImage)


        sentFeatures = Conv2D(32, (1, 1), strides=(1, 1), padding='same')(sentFeatures)
        sentFeatures = ReLU()(sentFeatures)
        sentFeatures = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(sentFeatures)
        sentFeatures = ReLU()(sentFeatures)
        sentFeatures = Conv2D(128, (1, 1), strides=(1, 1), padding='same')(sentFeatures)
        # model = BatchNormalization()(model)
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
        # model = BatchNormalization()(model)
        model = ReLU()(model)
        # model = Dropout(0.2)(model)

        model = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)

        model = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same")(model)
        # model = BatchNormalization()(model)
        model = ReLU()(model)
        model = Dropout(0.2)(model)

        merged = Concatenate()([model, sentFeatures])

        model = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same")(merged)
        model = ReLU()(model)

        model = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        # model = BatchNormalization()(model)
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
        # model = BatchNormalization()(model)
        model = ReLU()(model)

        model = Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same")(model)
        model = ReLU()(model)

        model = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)
        model = ReLU()(model)
        model = Dropout(0.2)(model)

        model = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding="same")(model)

        genmodel = Model([in_src_image, inpPrevLand, inpSentImage], [model], name="ConvModel")
        if self.lossMethod == 'w':
            genmodel.compile(loss=self.weightedLoss, optimizer=self.gan_optimizer, experimental_run_tf_function=False,
                             metrics=[self.psnrLossMetric, 'mse'])
        elif self.lossMethod == 'mse':
            genmodel.compile(loss='mse', optimizer=self.gan_optimizer, experimental_run_tf_function=False,
                             metrics=[self.psnrLossMetric, 'mse'])

        # plot_model(genmodel, to_file='./Model3.png', rankdir='TB', show_shapes=True, show_layer_names=True)
        return genmodel

    def saveModel(self, model=None):
        model_json = self.generator.to_json()
        self.generator.save_weights(self.dirName + "GeneratorModel.h5")
        with open(self.dirName + "GeneratorModel.json", "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk")
        return

    def mse(self, img1, img2):
        return np.mean((img2.astype(np.float64) - img1.astype(np.float64)) ** 2)

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

    def load_model(self, dir=None):
        if dir is not None:
            dirname = "/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/" + str(dir)  +"/"
        else:
            dirname = self.dirName
        json_file = open( dirname + "GeneratorModel.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # loaded_model = self.resNet_LandSat_model()
        loaded_model.load_weights(dirname + "GeneratorModel.h5")
        self.generator = loaded_model
        if self.lossMethod == 'w':
            self.generator.compile(loss=self.weightedLoss, optimizer=self.gan_optimizer, experimental_run_tf_function=False,
                             metrics=[self.psnrLossMetric, 'mse'])
        elif self.lossMethod == 'mse':
            self.generator.compile(loss='mse', optimizer=self.gan_optimizer, experimental_run_tf_function=False,
                             metrics=[self.psnrLossMetric, 'mse'])

        return loaded_model

    def rmseM(self, image1, image2):
        return np.sqrt(np.mean((image2.astype(np.float64) - image1.astype(np.float64)) ** 2))

    def psnrAndRmse(self, target, ref):
            rmseVU = self.rmseM(self.normal01(target), self.normal01(ref))
            rmseV = self.rmseM(target, ref)
            return round(20 * math.log10(255. / rmseV), 1), round(rmseVU,5)

    def psnrLossMetricUpdated(self, y_true, y_predict):
        psnr = tf.image.psnr(y_true, y_predict, max_val=255)
        return psnr

    def psnrLossMetric(self, y_true, y_predict):
        psnr = tf.image.psnr(y_true, y_predict, max_val=2)
        return psnr

    def normal01(self, img):
        return img/255

    def saveLoss(self, losses, psnrs, istesting=False):
        if istesting:
            with open(self.dirName + 'testLoss.txt', 'wb') as f:
                    pickle.dump(losses, f)
        else:
            with open(self.dirName + 'trainPSNR.txt', 'wb') as f1:
                pickle.dump(psnrs, f1)
            with open(self.dirName + 'trainLoss.txt', 'wb') as f:
                    pickle.dump(losses, f)

    def loadplotlosses(self):
        with open(self.dirName + 'trainLoss.txt', 'rb') as fp:
            losses = pickle.load(fp)
        with open(self.dirName + 'trainPSNR.txt', 'rb') as fp:
            prnrL = pickle.load(fp)

        plt.rcParams["figure.figsize"] = (18, 10)
        N = np.arange(0, len(losses))
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, losses, 'o-',  markersize=3, alpha=0.6, label='Training Loss')
        # plt.plot(N, lossesT,'*--',markersize=1, alpha=0.6, label='Testing Loss')
        plt.title("Loss while training the model", fontsize=25)
        plt.xlabel("Number of Epochs", fontsize=20)
        plt.ylabel("MSE Loss", fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        # plt.legend()
        plt.legend(fontsize=15)
        plt.savefig(self.dirName + 'trainTestLoss.png')
        plt.close()

        plt.rcParams["figure.figsize"] = (18, 10)
        N = np.arange(0, len(prnrL))
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, prnrL, 'o-', markersize=3, alpha=0.6, label='Training PSNR Accuracy')
        # plt.plot(N, lossesT,'*--',markersize=1, alpha=0.6, label='Testing Loss')
        plt.title("PSNR value while training the model", fontsize=25)
        plt.xlabel("Number of Epochs", fontsize=20)
        plt.ylabel("Accuracy in PSNR (dB)", fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        # plt.legend()
        plt.legend(fontsize=15)
        plt.savefig(self.dirName + 'trainTestPSNR.png')
        plt.close()
        return losses

    def predictionLosses(self):
        errors = []
        no_train = 0
        test_itr = self.test_dataset_obj.load_landsat_images(all_black_clouds=True, batch_size=10)
        while no_train < 30:
            try:
                no_train += 10
                (cloudy_img_land, target_cloudfree_land, inp_prev_sent, inp_prev_landsat, _)  = next(test_itr)
                history = self.generator.evaluate([cloudy_img_land,inp_prev_landsat, inp_prev_sent], [target_cloudfree_land[:,:,:,:8]], verbose=False)
                errors.append(history)
            except StopIteration:
                break
        errorPerEpoch = np.mean(np.array(errors))
        print("Total testing samples: ", no_train)
        return errorPerEpoch

    def getLayerWeightChange(self, model1, model2=None):
        path = "/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/"
        json_file = open(path + model1 + "GeneratorModel.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model1 = model_from_json(loaded_model_json)
        loaded_model1.load_weights(path + model1 + "GeneratorModel.h5")

        if model2 is not None:
            json_file = open(path + model2 + "GeneratorModel.json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model2 = model_from_json(loaded_model_json)
            loaded_model2.load_weights(path + model2 + "GeneratorModel.h5")


        for layer in loaded_model1.layers:
            print(layer.get_config(), layer.get_weights())
            we = layer.get_weights()
            if len(we) != 0:
                print("Shape: ")
                print(np.array(we[0]).shape)

    def display_training_images(self, epoch,
                                landsat_batch_x_cloudy, land_cloudy_with_clouds_batch, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch,  geo_batch, cc_batch, allI=False):
        if self.isTrain and allI:
            output_dir = self.dirName + 'trainALL/'
        elif self.isTrain and not allI:
            output_dir = self.dirName + 'train/'
        else:
            output_dir = self.dirName + 'test/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        r, c = 2, 4

        fakeImg = (self.dataloader.denormalize11(
            self.generator.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(np.uint8)

        # fakeImg = (self.dataloader.denormalize11(
        #     self.generator.predict(landsat_batch_x_cloudy)[-1])).astype(np.uint8)

        cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)

        for i in range(0, self.targetH):
            for j in range(0, self.targetW):
                if not all(cloudy_img_land[i, j] == 0):
                    fakeImg[i, j] = cloudy_img_land[i, j, :]

        target_cloudfree_land = self.dataloader.denormalize11(landsat_batch_y_cloud_free[-1][:,:,:8]).astype(np.uint8)


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

        _,rmse = self.psnrAndRmse(fakeImg[:,:,:3], target_cloudfree_land[:,:,:3])

        psnr = tf.image.psnr(fakeImg[:,:,:3], target_cloudfree_land[:,:,:3], 255).numpy()
        psnrAll = tf.image.psnr(fakeImg, target_cloudfree_land, 255).numpy()

        plt.suptitle("Geo: {} Cloud: {}% Target MSE: {} PSNR RGB: {} ALL: {} RMSE: {}".format( geo_batch[-1],  str(round(float(cc_batch[-1]) * 100, 3)),
            round(self.mse(self.normal01(fakeImg[:,:,:3]), self.normal01(target_cloudfree_land[:,:,:3])), 4), np.round(psnr,3), np.round(psnrAll,3), round(rmse, 4)), fontsize=20)

        fig.savefig(output_dir + "%s.png" % (epoch))
        plt.close()

        return psnr, psnrAll

    # def weightedLoss(self, y_true, y_pred):
    #     return K.mean(K.square(K.abs(y_pred - y_true[:,:,:,:8]) * y_true[:,:,:,-8:])) * 100 + K.mean(K.square(K.abs(y_pred - y_pred[:,:,:,:8])))
    def weightedLoss(self, y_true, y_pred):
        cloud_cloudshadow_mask = y_true[:, :, :, -1:]
        clearmask = K.ones_like(y_true[:, :, :, -1:]) - y_true[:, :, :, -1:]
        predicted = y_pred[:, :, :, 0:8]
        target = y_true[:, :, :, 0:8]
        cscmae = K.mean(clearmask * K.abs(predicted - target) + cloud_cloudshadow_mask * K.abs(
            predicted - target)) + 1.0 * K.mean(K.abs(predicted - target))
        return cscmae

    class LossAndErrorPrintingCallback(keras.callbacks.Callback):
        def __init__(self, sample_interval, saveLoss, loadplotlosses):
            self.global_loss,self.global_psnr = [],[]
            self.sample_interval = sample_interval
            self.saveLoss = saveLoss
            self.loadplotlosses = loadplotlosses

        def on_epoch_end(self, epoch, logs=None):
            self.global_loss.append(logs["loss"])
            self.global_psnr.append(logs["psnrLossMetric"])
            if epoch % self.sample_interval == 0:
                self.saveLoss(self.global_loss,self.global_psnr, istesting=False)
                self.loadplotlosses()

        def on_train_end(self, logs=None):
            self.saveLoss(self.global_loss, self.global_psnr, istesting=False)
            self.loadplotlosses()


    def trainOnlyGen(self, epoch_end, epoch_start=0, sample_interval=10, batchS=1000, geohashes=None):
        callbacks1 = hvd.callbacks.BroadcastGlobalVariablesCallback(0)
        callbacks1.set_model(self.generator)
        callbacks1.on_train_begin()
        callbacks2 = hvd.callbacks.MetricAverageCallback()
        callbacks2.set_model(self.generator)


        # callbacks2 = hvd.callbacks.LearningRateWarmupCallback(initial_lr=0.0001 * hvd.size(), warmup_epochs=5, verbose=1)
        # callbacks2.set_model(self.generator)
        # callbacks2.on_train_begin()

        if self.loadModel is False:
            global_loss, global_psnr = [], []
        elif os.path.exists(self.dirName + 'trainLoss.txt'):
            with open(self.dirName + 'trainLoss.txt', 'rb') as fp:
                global_loss = pickle.load(fp)

                if len(global_loss) == 0:
                    epoch_start = 0
                else:
                    epoch_start = len(global_loss) + 1


            with open(self.dirName + 'trainPSNR.txt', 'rb') as fp:
                global_psnr = pickle.load(fp)
        else:
            global_loss = []
            global_psnr = []

        test_itr = self.dataloader.load_iterator_from_paths( batch_size=1, geohashes=geohashes)

        for _ in range(4):
            landsat_batch_x_cloudy1, land_cloudy_with_clouds_batch1, landsat_batch_y_cloud_free1, sent_batch1, landsat_prev_batch1, geo_batch1, cc_batch1  = test_itr.__next__()

        train_itr = self.dataloader.load_iterator_from_paths_for_training(resize_image=True, batch_size=self.batch_size, geohashes=geohashes)

        for epoch in range(epoch_start, epoch_end):

            callbacks2.on_epoch_end(epoch=epoch)
            count=0
            batch_losses, batch_psnrs = [],[]
            if hvd.rank()==0:
                print("------------------------Training on epoch", epoch, "-------------------------------------")
            batchC = 0
            start_time = datetime.datetime.now()
            while count < batchS:
                try:
                    count += self.batch_size
                    batchC += 1
                    (inputs, landsat_batch_y_cloud_free) = next(train_itr)
                    if self.lossMethod=='mse':
                        landsat_batch_y_cloud_free = landsat_batch_y_cloud_free[:,:,:,:8]
                    g_loss = self.generator.train_on_batch(inputs, landsat_batch_y_cloud_free)

                    batch_losses.append(g_loss[0])
                    batch_psnrs.append(g_loss[1])
                    if hvd.rank() == 0:
                        print("\nEpoch {}/{} Batch {}/{} PSNR: {}".format(epoch, epoch_end, batchC, int(batchS/self.batch_size), round(g_loss[1], 3)), end='\r')

                except StopIteration:
                    break

            g_epoch_loss = np.mean(np.array(batch_losses))
            g_epoch_psnr = np.mean(np.array(batch_psnrs))

            timeE = ((datetime.datetime.now() - start_time).microseconds) * 0.001
            finalEpochRes = "------- Epoch {}/{} | Time: {}ms |  loss: {} | PSNR: {} --------".format(
                epoch, epoch_end, timeE, g_epoch_loss, g_epoch_psnr)

            if hvd.rank() == 0:
                print(finalEpochRes)

            global_psnr.append(g_epoch_psnr)
            global_loss.append(g_epoch_loss)

            if epoch % sample_interval == 0 :
                        self.display_training_images(str(epoch), landsat_batch_x_cloudy1, land_cloudy_with_clouds_batch1, landsat_batch_y_cloud_free1,
                                                                                                  sent_batch1, landsat_prev_batch1, geo_batch1, cc_batch1)
                        self.saveModel(self.generator)
                        self.saveLoss(global_loss, global_psnr, istesting=False)
                        self.loadplotlosses()

        self.saveModel(self.generator)
        self.saveLoss(global_loss,global_psnr, istesting= False)
        self.loadplotlosses()

    def trainOnlyGen1(self, epoch_end, epoch_start=0, sample_interval=10):
        test_itr = self.dataloader.load_iterator_from_paths(batch_size=1)
        for _ in range(4):
            landsat_batch_x_cloudy1, land_cloudy_with_clouds_batch1, landsat_batch_y_cloud_free1, sent_batch1, landsat_prev_batch1, geo_batch1, cc_batch1 = test_itr.__next__()

        save_weights_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: self.saveModel(self.generator) if epoch%sample_interval==0 else None)
        displayImage_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: self.display_training_images(epoch, landsat_batch_x_cloudy1, land_cloudy_with_clouds_batch1, landsat_batch_y_cloud_free1,
                                                                                                  sent_batch1, landsat_prev_batch1, geo_batch1, cc_batch1) if epoch%sample_interval==0 else None)
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            hvd.callbacks.LearningRateWarmupCallback(initial_lr= 0.0001 * hvd.size(),warmup_epochs=5,verbose=1),
            self.LossAndErrorPrintingCallback(sample_interval, self.saveLoss, self.loadplotlosses),
            save_weights_callback, displayImage_callback
            ]

        file1 = open("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/TotalCleanImages.txt", "r")
        total_samples = int(file1.read())
        if hvd.rank()==0:
            verbose=1
        else:
            verbose=0
        # total_samples // self.batch_size
        train_itr = self.dataloader.load_iterator_from_paths_for_training(resize_image = True, batch_size=self.batch_size)
        history = self.generator.fit(train_itr,
                            steps_per_epoch= 1000 // self.batch_size,
                            callbacks=callbacks,
                            epochs=epoch_end,
                            verbose=verbose,
                            initial_epoch=epoch_start
                           )
        if verbose:
            with open(self.dirName + 'trainHistoryDict', 'wb') as file:
                pickle.dump(history.history, file)

        # test_itr = self.dataloader2.load_iterator_from_paths(resize_image = True, batch_size=1)
        # score = hvd.allreduce(self.generator.evaluate_generator(test_iter, len(test_iter), workers=4))
        # print("Score: ", score)
        # if verbose:
        #     print('Test loss:', score[0])
        #     print('Test accuracy:', score[1])
        self.saveModel(self.generator)
        self.loadplotlosses()

    def printim(self):
        trian_itr = self.dataloader.load_landsat_images(all_black_clouds=True, batch_size=1)
        for i in range(28):
            cloudy_img_land1, target_cloudfree_land1, inp_sent_img1, inp_prev_landsat1, geo1 = trian_itr.__next__()
            self.display_training_images(str(i),
                                         cloudy_img_land1, target_cloudfree_land1, inp_sent_img1, inp_prev_landsat1,
                                         geo1)

    def getErrorsNDVI(self):
            if self.isTrain:
                startT = '2020-05-01'
                endT = '2020-11-01'
            else:
                startT = '2020-04-01'
                endT = '2020-05-01'
            dataloader = data_loader_single_mask.DatasetHandling(128, 128, folderI=self.folderI, startT=startT, endT=endT,
                                                        album='iowa-2015-2020-spa', no_of_timesteps=self.timestamps,
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

                    target_cloudfree_land = dataloader.denormalize11(target_cloudfree_land[-1][:,:,:8]).astype(
                        np.uint8)  # Returns image pixels between (0, 255)
                    target_cloudfree_land = self.normal01(target_cloudfree_land)


                    ndviFake = (fakeImg[:, :,3] - fakeImg[:, :, 0]) / (fakeImg[:, :, 3] + fakeImg[:, :,0])
                    ndviReal = (target_cloudfree_land[:, :,3] - target_cloudfree_land[:, :, 0]) / (target_cloudfree_land[:, :, 3] + target_cloudfree_land[:, :,0])
                    dataP.append(np.mean(ndviFake))
                    dataA.append(np.mean(ndviReal))


                except StopIteration:
                    break

            with open(self.dirName + 'ndviATesting_' + str(socket.gethostname()[-3:])  + '.txt', 'wb') as f:
                pickle.dump(dataA, f)

            with open(self.dirName + 'ndviPTesting_' + str(socket.gethostname()[-3:])  + '.txt', 'wb') as f:
                pickle.dump(dataP, f)

            coer,_ = pearsonr(dataA,dataP)
            plt.scatter(dataA, dataP, label = 'Pearson CorrCoef:' + str(round(coer, 4)), cmap='Greens')
            plt.style.use("ggplot")
            plt.title('Correlation plot between NDVI index for\npredicted and actual cloud free image')
            plt.ylabel('Predicted Average NDVI index')
            plt.xlabel('Actual Average NDVI index')
            plt.plot(np.unique(dataA), np.poly1d(np.polyfit(dataA, dataP, 1))(np.unique(dataA)), color='yellow')
            plt.legend()
            plt.savefig(self.dirName + 'NDVIComparisonTesting_' + str(socket.gethostname()[-3:]) +'.png')
            plt.close()
            return


    def getErrorsNDVIGlobal(self):
            dataP = []
            dataA = []

            for i in range(176, 216):
                if i == 177 or i==215:
                    continue
                with open('./plots/ndvia/ndviA_' + str(i) + '.txt', 'rb') as f:
                    ac = pickle.load(f)
                with open('./plots/ndvip/ndviP_' + str(i) + '.txt', 'rb') as f2:
                    pr = pickle.load(f2)
                if len(ac) == len(pr):
                    dataP.extend(pr)
                    dataA.extend(ac)

            coer,_ = pearsonr(dataA,dataP)
            plt.scatter(dataA, dataP, s = 4, label = '--- Pearson CorrCoef: ' + str(round(coer, 4)) + ' ---', alpha=0.4,c='b', marker='.' )
            plt.title('Correlation plot between NDVI index for\npredicted and actual cloud free image')
            plt.ylabel('Predicted Average NDVI index')
            plt.xlabel('Actual Average NDVI index')
            plt.plot(np.unique(dataA), np.poly1d(np.polyfit(dataA, dataP, 1))(np.unique(dataA)), color='yellow')
            plt.legend()
            plt.savefig(self.dirName + 'NDVIComparisonAllMachines_' + str(socket.gethostname()[-3:]) +'.png')
            plt.close()
            return

    def PrintErrors(self, clip_image = True, tirb = -2):
        # YYMMDD
        dataloader = data_loader_single_mask.DatasetHandling(128, 128,folderI=self.folderI, startT='2020-04-01', endT='2020-05-01',
                                           album='iowa-2015-2020-spa', no_of_timesteps=self.timestamps, batch_size=1, istrain=self.isTrain)

        psnrs =  []
        test_itr = dataloader.load_landsat_images(batch_size=1, all_black_clouds=True)
        count = 0
        output_dir = "./allIm2/"
        while True:
            try:
                (cloudy_img_land, target_cloudfree_land, inp_prev_sent, inp_prev_landsat, geo) = next(test_itr)
                fakeImg = (dataloader.denormalize11(
                    self.generator.predict([cloudy_img_land, inp_prev_landsat, inp_prev_sent])[-1])).astype(np.uint8)

                target_cloudfree_land = dataloader.denormalize11(target_cloudfree_land[-1][:,:,:8]).astype(np.uint8)    # Returns image pixels between (0, 255)

                cloudy_img_land = self.dataloader.denormalize11(cloudy_img_land[-1]).astype(np.uint8)
                inp_prev_sent = inp_prev_sent[-1]
                inp_prev_landsat = np.array(
                    [self.dataloader.denormalize11(img).astype(np.uint8) for img in inp_prev_landsat])
                psnr, _ = self.psnrAndRmse(fakeImg, target_cloudfree_land)
                if clip_image is True:
                    psnrTIR, _ = self.psnrAndRmse(np.clip(fakeImg[:, :, tirb], 190, None), target_cloudfree_land[:, :, tirb])
                else:
                    psnrTIR, _ = self.psnrAndRmse(fakeImg[:, :, tirb], target_cloudfree_land[:, :, tirb])

                titles1 = ['Landsat Y1', 'Sentinel-2-RGB', 'Sen2 NDVI', 'Sen2 Edge', 'Input cloudy Landsat', 'Target',
                           'Predicted', 'TIR P', 'TIR A']
                titles = titles1
                r = 2
                fig, axarr = plt.subplots(r, 5, figsize=(15, 12))
                np.vectorize(lambda axarr: axarr.axis('off'))(axarr)

                for col in range(1):
                    axarr[0, col].imshow(inp_prev_landsat[col][:, :, :3])
                    axarr[0, col].set_title(titles[col], fontdict={'fontsize': 15})

                axarr[0, 1].imshow(inp_prev_sent[:, :, :3])
                axarr[0, 1].set_title(titles[1], fontdict={'fontsize': 15})

                axarr[0, 2].imshow(inp_prev_sent[:, :, 3], cmap=plt.cm.summer)
                axarr[0, 2].set_title(titles[2], fontdict={'fontsize': 15})

                axarr[0, 3].imshow(inp_prev_sent[:, :, -1])
                axarr[0, 3].set_title(titles[3], fontdict={'fontsize': 15})

                axarr[r - 1, 0].imshow(cloudy_img_land[:, :, :3])
                axarr[r - 1, 0].set_title(titles[4], fontdict={'fontsize': 15})

                axarr[r - 1, 1].imshow(target_cloudfree_land[:, :, :3])
                axarr[r - 1, 1].set_title(titles[5], fontdict={'fontsize': 15})

                axarr[r - 1, 2].imshow(fakeImg[:, :, :3])
                axarr[r - 1, 2].set_title(titles[6], fontdict={'fontsize': 15})

                if clip_image is True:
                    vminv = min(np.min(np.clip(fakeImg[:, :,tirb], 190, None)), np.min(target_cloudfree_land[:,:,tirb]))
                    vmaxv = max(np.max(np.clip(fakeImg[:, :, tirb], 190, None)), np.max(target_cloudfree_land[:,:,tirb]))
                else:
                    vminv = min(np.min(fakeImg[:, :, tirb]), np.min(target_cloudfree_land[:, :, tirb]))
                    vmaxv = max(np.max(fakeImg[:, :, tirb]), np.max(target_cloudfree_land[:, :, tirb]))

                img2 = axarr[r - 1, 3].imshow(np.clip(fakeImg[:, :, tirb], 190, None), cmap='hot' ,interpolation='nearest',
                                               vmin = vminv, vmax= vmaxv)
                axarr[r - 1, 3].set_title(titles[7], fontdict={'fontsize': 15})
                # fig.colorbar(img2, ax=axarr[r - 1, 3], shrink=0.5)

                img2 = axarr[r - 1, 4].imshow(target_cloudfree_land[:, :, tirb], cmap='hot', interpolation='nearest',
                                               vmin = vminv, vmax= vmaxv)
                axarr[r - 1, 4].set_title(titles[8], fontdict={'fontsize': 15})
                fig.colorbar(img2, ax=axarr[r - 1, [3,4]], shrink=0.5)

                # Print Predicted

                psnr, rmse = self.psnrAndRmse(fakeImg, target_cloudfree_land)

                plt.suptitle("Geohash: {} Target MSE: {} PSNR tir: {} RMSE: {}".format(geo,round(self.mse(self.normal01(fakeImg),self.normal01(target_cloudfree_land)), 3), round(psnrTIR, 3),
                                                                                   round(rmse, 3)), fontsize=20)

                fig.savefig(output_dir + "%s.png" % (count))
                plt.close()



                print("Count : ", count)
                psnrs.append(psnr)
                count +=1
            except StopIteration:
                break

        psnrs = np.array(psnrs)
        print("Total Testing sample: {} Average Testing PSNR: {}".format(count, np.mean(psnrs)))

        return



    def PrintErrorsPerband(self, geohashes=None):
        psnrsRed,psnrsBlue,psnrsGreen,psnrsNIR,psnrsSWIR1,psnrsSWIR2, psnrsTIR1, psnrsTIR2 = [], [], [], [], [], [], [], []
        test_itr = self.dataloader.load_iterator_from_paths(batch_size=1, geohashes=geohashes)
        count = 0
        while True:
            try:
                (landsat_batch_x_cloudy, land_cloudy_with_clouds_batch, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, geo_batch, cc_batch) = next(test_itr)
                fakeImg = (self.dataloader.denormalize11(
                    self.generator.predict([landsat_batch_x_cloudy, landsat_prev_batch, sent_batch])[-1])).astype(np.uint8)

                target_cloudfree_land = self.dataloader.denormalize11(landsat_batch_y_cloud_free[-1][:,:,:8]).astype(np.uint8)    # Returns image pixels between (0, 255)
                cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)
                bandDic = {0:psnrsRed, 1:psnrsBlue, 2:psnrsGreen, 3:psnrsNIR, 4:psnrsSWIR1, 5:psnrsSWIR2, 6:psnrsTIR1, 7:psnrsTIR2}
                #
                # r = 2
                # fig, axarr = plt.subplots(r, 2)
                # np.vectorize(lambda axarr: axarr.axis('off'))(axarr)
                # titles = ['NIR Predicted', 'NIR Actual', 'SWIR 1 Predicted', 'Actual swir 1']
                # psnr1 = tf.image.psnr(np.reshape(fakeImg[:, :, 3], (128, 128, 1)),
                #                      np.reshape(target_cloudfree_land[:, :, 3], (128, 128, 1)), 255).numpy()
                #
                # psnr2 = tf.image.psnr(np.reshape(fakeImg[:, :, 4], (128, 128, 1)),
                #                       np.reshape(target_cloudfree_land[:, :, 4], (128, 128, 1)), 255).numpy()
                # im1 = axarr[0, 0].imshow(fakeImg[:, :, 3], cmap=plt.cm.summer, vmin =0 , vmax =255)
                # axarr[0, 0].set_title(titles[0] + str(psnr1), fontdict={'fontsize': 15})
                # fig.colorbar(im1, ax=axarr[0,0])
                #
                # im2 = axarr[0, 1].imshow(target_cloudfree_land[:, :, 3], cmap=plt.cm.summer, vmin =0 , vmax =255)
                # axarr[0, 1].set_title(titles[1], fontdict={'fontsize': 15})
                # fig.colorbar(im2, ax=axarr[0, 1])
                #
                # im3 = axarr[1, 0].imshow(fakeImg[:, :, 4], cmap=plt.cm.summer, vmin =0 , vmax =255)
                # axarr[1, 0].set_title(titles[2]  + str(psnr2), fontdict={'fontsize': 15})
                # fig.colorbar(im3, ax=axarr[1, 0])
                #
                # im4 = axarr[1, 1].imshow(target_cloudfree_land[:, :, 4], cmap=plt.cm.summer, vmin =0 , vmax =255)
                # axarr[1, 1].set_title(titles[3], fontdict={'fontsize': 15})
                # fig.colorbar(im4, ax=axarr[1, 1] )
                #
                # fig.savefig(self.dirName + "/random/%s.png" % (count))
                # plt.close()
                for i in range(0, self.targetH):
                    for j in range(0, self.targetW):
                        if not all(cloudy_img_land[i, j] == 0):
                            fakeImg[i, j] = cloudy_img_land[i, j, :]

                for i in range(0, len(bandDic)):
                    psnr= tf.image.psnr(np.reshape(fakeImg[:, :, i], (128,128,1)), np.reshape(target_cloudfree_land[:, :, i],(128,128,1)), 255).numpy()
                    if np.isinf(psnr):
                        psnr = 48.0
                    oldpsnr = bandDic.get(i)
                    oldpsnr.append(psnr)
                    bandDic[i] = oldpsnr

                count +=1
            except StopIteration:
                break

        print("Total Testing sample: ", count)
        allPsnr = [psnrsRed, psnrsBlue, psnrsGreen, psnrsNIR, psnrsSWIR1, psnrsSWIR2, psnrsTIR1, psnrsTIR2]
        bandsN = ['Red', 'Blue', 'Green', 'NIR', 'SWIR1', 'SWIR2', 'TIR1', 'TIR2']

        # with open(self.dirName + 'psnrPerBandTesting_' + str(socket.gethostname()[-3:]) + '.txt', 'wb') as f:
        #     pickle.dump(allPsnr, f)

        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        ax.boxplot(allPsnr,  notch=True, patch_artist=True,  showfliers=False)
        plt.xticks(np.arange(1, len(bandsN) + 1), bandsN)
        plt.title("PSNR Training accuracy per band")
        plt.xlabel("Spectral Bands")
        plt.ylabel("PSNR Measure")
        plt.savefig(self.dirName + 'LossPerBandTesting_'+  str(socket.gethostname()[-3:]) + '.png')
        plt.close()

        return

    def PrintErrorsPerbandGlobal(self):
        psnrsRed, psnrsBlue, psnrsGreen, psnrsNIR, psnrsSWIR1, psnrsSWIR2, psnrsTIR1, psnrsTIR2 = [], [], [], [], [], [], [], []
        psnrsRedM, psnrsBlueM, psnrsGreenM, psnrsNIRM, psnrsSWIR1M, psnrsSWIR2M, psnrsTIR1M, psnrsTIR2M = [], [], [], [], [], [], [], []
        bandDic = {0: psnrsRed, 1: psnrsBlue, 2: psnrsGreen, 3: psnrsNIR, 4: psnrsSWIR1, 5: psnrsSWIR2,
                   6: psnrsTIR1, 7: psnrsTIR2}
        bandDicM = {0: psnrsRedM, 1: psnrsBlueM, 2: psnrsGreenM, 3: psnrsNIRM, 4: psnrsSWIR1M, 5: psnrsSWIR2M,
                    6: psnrsTIR1M, 7: psnrsTIR2M}
        bandsN = ['Red', 'Blue', 'Green', 'NIR', 'SWIR1', 'SWIR2', 'TIR1', 'TIR2']
        machps = []
        for i in range(176, 216):

                with open('./plots/psnr/psnrPerBand_' + str(i) + '.txt', 'rb') as f:
                    psnrallpermachine = pickle.load(f)

                allb = []
                for j in range(0, len(bandDic)):
                     oldpsnr = bandDic.get(j)
                     print("hahah")
                     oldpsnr.extend(psnrallpermachine[j])
                     bandDic[j] = oldpsnr
                     print("Band Name: {} Machine: {} Mean psnr: {}".format(bandsN[j],i , np.mean(psnrallpermachine[j])))
                     allb.append(np.mean(psnrallpermachine[j]))
                machps.append(sum(allb)/8)


        print("Global PSNR value: ", np.mean(np.array(machps)))
        allPsnr = [psnrsRed, psnrsBlue, psnrsGreen, psnrsNIR, psnrsSWIR1, psnrsSWIR2, psnrsTIR1, psnrsTIR2]

        # with open(self.dirName + 'psnrPerBandAllMachines_' + str(socket.gethostname()[-3:]) + '.txt', 'wb') as f:
        #     pickle.dump(allPsnr, f)
        #
        fig, ax = plt.subplots()
        plt.style.use("ggplot")
        ax.boxplot(allPsnr, notch=True, patch_artist=True)
        plt.xticks(np.arange(1, len(bandsN) + 1), bandsN)
        plt.title("PSNR Training accuracy per band")
        plt.xlabel("Spectral Bands")
        plt.ylabel("PSNR Measure")
        plt.savefig(self.dirName + 'PsnrPerBandAllMachines.png')
        plt.close()
        return

    def checkForMoreTraining(self, folderI, psnrThres = 30, varianceThres = 3, countT=800, geohashes=None):
            dirName = "/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/" + str(folderI)  +"/"
            json_file = open(dirName + "GeneratorModel.json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            generator = model_from_json(loaded_model_json)
            generator.load_weights(dirName + "GeneratorModel.h5")
            if self.lossMethod == 'w':
                generator.compile(loss=self.weightedLoss, optimizer=self.gan_optimizer,
                                       experimental_run_tf_function=False,  metrics=[self.psnrLossMetric, 'mse'])
            elif self.lossMethod == 'mse':
                generator.compile(loss='mse', optimizer=self.gan_optimizer, experimental_run_tf_function=False, metrics=[self.psnrLossMetric, 'mse'])
            # psnrPerGeo = {}
            errors = []
            test_dataset_obj = data_loader_single_mask.DatasetHandling(128,128,folderI=self.folderI,
                                               album='iowa-2015-2020-spa', no_of_timesteps=1,
                                               batch_size=1, istrain=False, all_black_clouds = True,saveInputMetaData = False)
            test_itr = test_dataset_obj.load_iterator_from_paths_for_training( resize_image = True, batch_size=1, geohashes=geohashes)

            if self.isTrain :
                file1 = open("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/TotalCleanImages.txt", "r")
                clean_image_count = int(file1.read())
                file1.close()
            else:
                file1 = open("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/TotalCleanImagesValidation.txt","r")
                clean_image_count = int(file1.read())
                file1.close()
            imageCo = 0
            # clean_image_count = 10
            while imageCo <= clean_image_count:
                imageCo += 1
                try:

                    inp, landsat_batch_y_cloud_free = next(test_itr)
                    inp_cloudy_mask ,_,_= inp
                    inp_cloudy_mask = self.dataloader.denormalize11(inp_cloudy_mask[-1]).astype(np.uint8)

                    fakeImg = (self.dataloader.denormalize11(
                        generator.predict(inp)[-1])).astype(np.uint8)

                    for i in range(0, self.targetH):
                        for j in range(0, self.targetW):
                            if not all(inp_cloudy_mask[i, j] == 0):
                                fakeImg[i, j] = inp_cloudy_mask[i, j, :]

                    target_cloudfree_land = self.dataloader.denormalize11(landsat_batch_y_cloud_free[-1][:, :, :8]).astype(
                        np.uint8)
                    psnr = tf.image.psnr(fakeImg,target_cloudfree_land, max_val=255).numpy()
                    print("Count =" ,  imageCo, " psnr: ", psnr)
                    errors.append(psnr)
                except StopIteration:
                    break

            avgPSNR = np.mean(np.array(errors))
            stdPSNR = np.std(errors)
            print("Average PSNR: ", avgPSNR, " Average STD: ", stdPSNR)
            # test_itr = test_dataset_obj.load_iterator_from_paths_for_training(resize_image=True, batch_size=1)
            # score = hvd.allreduce(generator.evaluate(test_itr,steps= 150, verbose=1))
            # print("Final average PSNR score: ", round(score[1].numpy(), 3))

    def justSaveImage(self, geohashes=None, allI=None):
        psnrsRGB, psnrsAll = [],[]
        test_itr = self.dataloader.load_iterator_from_paths(batch_size=1, geohashes=geohashes)
        count = 1
        while True:
            try:
                print("Displaying count: ", count)
                landsat_batch_x_cloudy, land_cloudy_with_clouds_batch, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch, geo_batch, cc_batch  = test_itr.__next__()
                self.display_training_images(count, landsat_batch_x_cloudy, land_cloudy_with_clouds_batch, landsat_batch_y_cloud_free,
                                             sent_batch, landsat_prev_batch, geo_batch, cc_batch, allI=allI)

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

                psnrRGB = tf.image.psnr(fakeImg[:, :, :3], target_cloudfree_land[:, :, :3], 255).numpy()
                psnrALL = tf.image.psnr(fakeImg, target_cloudfree_land, 255).numpy()

                if np.isinf(psnrALL):
                    psnrALL = 48.0
                if np.isinf(psnrRGB):
                    psnrRGB = 48.0
                # print(psnrRGB, psnrALL)
                count += 1
                psnrsRGB.append(psnrRGB)
                psnrsAll.append(psnrALL)
            except StopIteration:
                break

        if len(psnrsRGB) is not 0:
            print("Average psnr RGB band: ", np.mean(np.array(psnrsRGB)))
            print("Average psnr All bands: ", np.mean(np.array(psnrsAll)))
        else:
            print("No image found")


class RefinementTree():
    def __init__(self):
        self.geohash = None
        self.neighbors = None
        self.childrens = None
        self.currentPSNR = None
        self.weightDIR = None
        self.machineStored = None
        self.areaCode = None

    def getgeohash(self, geohash):
        self.geohash = geohash

    def setgeohash(self):
        return self.geohash

    def getneighbors(self, neighbors):
        self.neighbors = neighbors

    def setneighbors(self):
        return self.neighbors

    def getchildrens(self, childrens):
        self.childrens = childrens

    def setchildrens(self):
        return self.childrens

    def getcurrentPSNR(self, currentPSNR):
        self.currentPSNR = currentPSNR

    def setcurrentPSNR(self):
        return self.currentPSNR

    def getweightDIR(self, weightDIR):
        self.weightDIR = weightDIR

    def setweightDIR(self):
        return self.weightDIR

    def getmachineStored(self, machineStored):
        self.machineStored = machineStored

    def setmachineStored(self):
        return self.machineStored

    def getareaCode(self, areaCode):
        self.areaCode = areaCode

    def setareaCode(self):
        return self.areaCode

class NeighboorhoodTree:
    def __init__(self):
        self.neighbours = None
        self.PSNR = None
        self.machineStored = None
        self.areaCode = None



if __name__ == '__main__':

    train_helpers=train_helpers(folderI = 1008,timestamps = 1, batch_size = 3, lossMethod='mse', onlyPredict=False,
                                istrain=True, tranferLear=1002
                                , loadModel=True)
    # train_helpers.checkForMoreTraining(folderI=10, countT=400)
    # train_helpers.display_training_images(epoch = 1, is_train=True)
    #
    # trian_itr = train_helpers.dataloader.load_landsat_images(all_black_clouds=True, batch_size=1)
    # test_itr = train_helpers.test_dataset_obj.load_landsat_images(all_black_clouds=True, batch_size=1)
    # countI, countTest = 0, 0
    # while True:
    #     try:
    #          cloudy_img_land, target_cloudfree_land, cloud_cov_perc, inp_prev_sent, inp_prev_landsat = trian_itr.__next__()
    #          # train_helpers.display_training_images(countI, is_train=True)
    #          # print("cloudy_img_land : {} \n target_cloudfree_land : {} \n cloud_cov_perc : {} \n inp_prev_sent : {} \n inp_prev_landsat : {}".format(cloudy_img_land.shape, target_cloudfree_land.shape, cloud_cov_perc.shape, inp_prev_sent.shape, inp_prev_landsat.shape))
    #          countI += 1
    #          print("Current train: ", countI)
    #     except StopIteration:
    #         break
    # while True:
    #     try:
    #          cloudy_img_land, target_cloudfree_land, cloud_cov_perc, inp_prev_sent, inp_prev_landsat = test_itr.__next__()
    #          # train_helpers.display_training_images(countI, is_train=True)
    #          # print("cloudy_img_land : {} \n target_cloudfree_land : {} \n cloud_cov_perc : {} \n inp_prev_sent : {} \n inp_prev_landsat : {}".format(cloudy_img_land.shape, target_cloudfree_land.shape, cloud_cov_perc.shape, inp_prev_sent.shape, inp_prev_landsat.shape))
    #          countTest += 1
    #          print("Current test : ", countTest)
    #     except StopIteration:
    #         break
    # print("\n\nTotal Training images founds: ", countI)
    # print("Total Testing images founds: ", countTest)
    # print("Total images found: {} ".format(countI))
    # train_helpers.checkForMoreTraining(1002)

    geohashes1 = ['5', '7', '6', '4', '1', '3', '0', '2','p','r','n','q', 'j', 'm', 'h', 'k']
    geohashes2 = ['8', 'e','g','d','f','9','c','b','x','z','w','y','t','v','s','u']
    train_helpers.trainOnlyGen(epoch_end=701, sample_interval=5, batchS=1000, geohashes=geohashes1)
    #
    train_helpers.justSaveImage(geohashes2, allI=False)
    # train_helpers.justSaveImage(geohashes1, all=True)

    # train_helpers.PrintErrorsPerbandGlobal()
    # train_helpers.getErrorsNDVIGlobal()
    # train_helpers.PrintErrorsPerband(geohashes1)
    # train_helpers.getErrorsNDVI()
    # train_helpers.PrintErrors(clip_image=True, tirb=-2)