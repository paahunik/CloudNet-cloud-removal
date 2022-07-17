import math
import tensorflow as tf
import horovod.tensorflow.keras as hvd
hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
from cloud_removal import data_loader_clouds, model_helpers
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, model_from_json,Model
from tensorflow.keras.layers import Dense,Lambda, Dropout, Flatten,Conv2D, MaxPooling2D, ReLU,\
    Activation,Dropout, RepeatVector, Input,\
    Conv2DTranspose, LeakyReLU,Reshape, BatchNormalization, Concatenate, Add
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras import backend as K
from scipy.stats.stats import pearsonr
import os
import numpy as np
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[ '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499'])

import socket
import gdal
import datetime
import pickle

class train_helpers():
    def __init__(self, folderI=6, timestamps=1, batch_size=1, istrain=True, h=128, w=128, lossMethod='mse', onlyPredict=False, modelN='own',
                 loadModel = False):
        self.dataloader = data_loader_clouds.DatasetHandling(h, w,
                         album='iowa-2015-2020-spa', no_of_timesteps=timestamps, batch_size=batch_size, istrain=istrain)
        self.model_helper = model_helpers.cloud_removal_models(timeStep=timestamps, batch_size=batch_size, w=128,h=128)
        self.timestamps = timestamps
        self.isTrain = istrain
        self.loadModel = loadModel
        self.dirName = "/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/" + str(folderI)  +"/"
        if not os.path.isdir("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/" ):
            os.mkdir("/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/")
        if not os.path.isdir(self.dirName):
            os.mkdir(self.dirName)
        self.dataloader2 = data_loader_clouds.DatasetHandling(h, w,
                                                              album='iowa-2015-2020-spa', no_of_timesteps=timestamps,
                                                              batch_size=1, istrain=istrain)

        self.test_dataset_obj = data_loader_clouds.DatasetHandling(h, w,
                                                             album='iowa-2015-2020-spa', no_of_timesteps=timestamps,
                                                             batch_size=5, istrain=False)
        self.lossMethod = lossMethod
        self.targetH = h
        self.targetW = w
        self.targetShape = (self.targetW, self.targetH, 8)
        self.inputShape = (self.targetW, self.targetH, 8)
        self.sentShape = (self.targetH, self.targetW, 8)
        self.gan_optimizer = hvd.DistributedOptimizer(tf.keras.optimizers.Adam(0.0001 * hvd.size()))
        self.onlyPredict = onlyPredict
        if self.onlyPredict:
            self.generator = self.load_model()
        elif self.onlyPredict is False and modelN == 'own':
            if loadModel is False:
                self.generator = self.getGeneratorModel2()
            else:
               self.generator = self.load_model()
        elif self.onlyPredict is False and modelN == 'res':
            if loadModel is True:
                self.generator = self.resNet_LandSat_model()
            else:
                self.generator = self.load_model()



    def carl_error(self, y_true, y_pred):
        cloud_cloudshadow_mask = y_true[:, :, :, -1:]
        clearmask = K.ones_like(y_true[:, :, :, -1:]) - y_true[:, :, :, -1:]
        predicted = y_pred[:, :, :, 0:8]
        target = y_true[:, :, :, 0:8]
        cscmae = K.mean(clearmask * K.abs(predicted - target) + cloud_cloudshadow_mask * K.abs(
            predicted - target)) + 1.0 * K.mean(K.abs(predicted - target))
        return cscmae

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

        optimizer = hvd.DistributedOptimizer(Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
                                             backward_passes_per_step=1,
                                             average_aggregated_gradients=True)
        if self.lossMethod == 'mse':
            model.compile(optimizer=optimizer, loss='mse')
        else:
            model.compile(optimizer=optimizer, loss=self.carl_error)
        return model

    def getGeneratorModel2(self):
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
            genmodel.compile(loss=self.weightedLoss, optimizer=self.gan_optimizer, experimental_run_tf_function=False)
        elif self.lossMethod == 'mse':
            genmodel.compile(loss='mse', optimizer=self.gan_optimizer, experimental_run_tf_function=False)

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

    def load_model(self):
        json_file = open(self.dirName + "GeneratorModel.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.dirName + "GeneratorModel.h5")
        self.generator = loaded_model
        optimizer = hvd.DistributedOptimizer(Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
                                             backward_passes_per_step=1,
                                             average_aggregated_gradients=True)
        if self.lossMethod == 'w':
            self.generator.compile(loss=self.weightedLoss, optimizer=self.gan_optimizer, experimental_run_tf_function=False)
        elif self.lossMethod == 'mse':
            self.generator.compile(loss='mse', optimizer=optimizer, experimental_run_tf_function=False)
        else:
            self.generator.compile(loss=self.carl_error, optimizer=optimizer, experimental_run_tf_function=False)
        return loaded_model

    def rmseM(self, image1, image2):
        return np.sqrt(np.mean((image2.astype(np.float64) - image1.astype(np.float64)) ** 2))

    def psnrAndRmse(self, target, ref):
            rmseVU = self.rmseM(self.normal01(target), self.normal01(ref))
            rmseV = self.rmseM(target, ref)
            return round(20 * math.log10(255. / rmseV), 1), round(rmseVU,5)

    def normal01(self, img):
        return img/255

    def saveLoss(self, losses, istesting=False):
        if istesting:
            with open(self.dirName + 'testLoss.txt', 'wb') as f:
                    pickle.dump(losses, f)
        else:
            with open(self.dirName + 'trainLoss.txt', 'wb') as f:
                    pickle.dump(losses, f)

    def loadplotlosses(self):
        with open(self.dirName + 'trainLoss.txt', 'rb') as fp:
            losses = pickle.load(fp)
        with open(self.dirName + 'testLoss.txt', 'rb') as fp:
            lossesT = pickle.load(fp)

        plt.rcParams["figure.figsize"] = (18, 10)
        N = np.arange(0, len(losses))
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, losses, 'o-',  markersize=0.8, alpha=0.6, label='Training Loss')
        plt.plot(N, lossesT,'*--',markersize=0.8, alpha=0.6, label='Testing Loss')
        plt.title("Training and Testing Loss while training the Resnet model", fontsize=25)
        plt.xlabel("Number of Epochs", fontsize=20)
        plt.ylabel("MSE Loss", fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend()
        plt.legend(fontsize=15)
        plt.savefig(self.dirName + 'trainTestLoss.png')
        return losses

    def display_training_images_res(self, epoch,
                                cloudy_img_land, target_cloudfree_land, geo, is_train=True):


        if is_train:
            output_dir = self.dirName + 'train/'
        else:
            output_dir = self.dirName + 'test/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)


        fakeImg = (self.dataloader.denormalize11(self.generator.predict(cloudy_img_land)[-1])).astype(np.uint8)
        cloudy_img_land = self.dataloader.denormalize11(cloudy_img_land[-1]).astype(np.uint8)
        target_cloudfree_land = self.dataloader.denormalize11(target_cloudfree_land[-1][:,:,:8]).astype(np.uint8)
        titles = ['Input cloudy Landsat', 'Target', 'Predicted']

        fig, axarr = plt.subplots(1, 3, figsize=(15, 12))
        np.vectorize(lambda axarr: axarr.axis('off'))(axarr)

        axarr[0].imshow(cloudy_img_land[:, :, :3])
        axarr[0].set_title(titles[0], fontdict={'fontsize': 15})

        axarr[1].imshow(target_cloudfree_land[:, :, :3])
        axarr[1].set_title(titles[1], fontdict={'fontsize': 15})

        axarr[2].imshow(fakeImg[:, :, :3])
        axarr[2].set_title(titles[2], fontdict={'fontsize': 15})

        psnr,rmse = self.psnrAndRmse(fakeImg[:,:,:3], target_cloudfree_land[:,:,:3])

        plt.suptitle("Geohash: {} Target MSE: {} PSNR: {} RMSE: {}".format( geo,
            round(self.mse(self.normal01(fakeImg[:,:,:3]), self.normal01(target_cloudfree_land[:,:,:3])), 3), round(psnr,3), round(rmse, 3)), fontsize=20)

        fig.savefig(output_dir + "%s.png" % (epoch))
        plt.close()


    def weightedLoss(self, y_true, y_pred):
        return K.mean(K.square(K.abs(y_pred - y_true[:,:,:,:8]) * y_true[:,:,:,-8:])) * 100 + K.mean(K.square(K.abs(y_pred - y_pred[:,:,:,:8])))

    def predictionLosses(self):
        errors = []
        no_train = 0
        test_itr = self.test_dataset_obj.load_landsat_images(all_black_clouds=True, batch_size=10)
        while no_train < 300:
            try:
                no_train += 10
                (cloudy_img_land, target_cloudfree_land, _, _, _) = next(test_itr)
                history = self.generator.evaluate(cloudy_img_land, target_cloudfree_land[:,:,:,:8], verbose=False)
                errors.append(history)
            except StopIteration:
                break
        errorPerEpoch = np.mean(np.array(errors))
        return errorPerEpoch


    def trainOnlyGen(self, epoch_end, epoch_start=0, sample_interval=10):
        callbacks1 = hvd.callbacks.BroadcastGlobalVariablesCallback(0)
        callbacks1.set_model(self.generator)
        callbacks1.on_train_begin()
        test_itr = self.dataloader2.load_landsat_images(all_black_clouds=True, batch_size=1)

        for _ in range(2):
            cloudy_img_land1, target_cloudfree_land1, _, _, geo1 = test_itr.__next__()

        if self.loadModel is False:
            global_loss, testing_loss = [],[]
        elif os.path.exists(self.dirName + 'trainLoss.txt'):
            with open(self.dirName + 'trainLoss.txt', 'rb') as fp:
                global_loss = pickle.load(fp)
                if len(global_loss) == 0:
                    epoch_start = 0
                else:
                    epoch_start = len(global_loss) + 1
            if os.path.exists(self.dirName + 'testLoss.txt'):
                with open(self.dirName + 'testLoss.txt', 'rb') as fp:
                    testing_loss = pickle.load(fp)
            else:
                testing_loss = []
        else:
            testing_loss = []
            global_loss = []

        for epoch in range(epoch_start, epoch_end):
            trian_itr = self.dataloader.load_landsat_images(all_black_clouds=True)
            losses = []
            print("-------------Training on epoch", epoch, "-----------------")
            batchC = 0
            start_time = datetime.datetime.now()
            while True:
                try:
                    batchC += 1
                    (cloudy_img_land, target_cloudfree_land, _, _, _) = next(trian_itr)

                    g_loss = self.generator.train_on_batch(cloudy_img_land, target_cloudfree_land)
                    print_losses = {"G": []}
                    print_losses['G'].append(g_loss)
                    g_avg_loss = np.array(print_losses['G']).mean(axis=0)

                    losses.append(g_avg_loss)
                    print("Batch : ", batchC)

                except StopIteration:
                    break

            timeE = ((datetime.datetime.now() - start_time).microseconds) * 0.001

            finalEpochRes = "\nEpoch {}/{} | Time: {}ms |  Generator: {}".format(
                epoch, epoch_end, timeE, g_avg_loss)

            if hvd.rank() == 0:
                print(finalEpochRes)

            # print("testing loss: ", testing_loss)
            # print("training loss: ", global_loss)
            testing_loss.append(self.predictionLosses())
            global_loss.append(np.mean(losses))

            if epoch % sample_interval == 0 :
                        self.display_training_images(str(epoch) ,
                                                      cloudy_img_land1, target_cloudfree_land1, geo1)
                        self.saveModel(self.generator)
                        if hvd.rank() == 0:
                            self.saveLoss(global_loss, istesting=False)
                            self.saveLoss(testing_loss, istesting= True)
                            self.loadplotlosses()


        self.saveModel(self.generator)
        if hvd.rank() == 0:
            self.saveLoss(global_loss, istesting= False)
            self.saveLoss(testing_loss, istesting=True)
            self.loadplotlosses()

    def printim(self):
        trian_itr = self.dataloader.load_landsat_images(all_black_clouds=True, batch_size = 1)
        test_itr = self.test_dataset_obj.load_landsat_images(all_black_clouds=True, batch_size=1)
        i = 0
        while True:
            try:
                # cloudy_img_land1, target_cloudfree_land1, _,_, geo1 = trian_itr.__next__()
                #
                # self.display_training_images(str(i),
                #                              cloudy_img_land1, target_cloudfree_land1, geo1, is_train=True)

                # except StopIteration:
                #     print("No more taining data ")
                #     break
                #
                # try:
                #

                cloudy_img_land, target_cloudfree_land, _,_, geo = next(test_itr)

                self.display_training_images(str(i),
                                                 cloudy_img_land, target_cloudfree_land, geo, is_train=False)
            except StopIteration:

                    print("no more testing data")
                    break

            i += 1

    def getErrorsNDVI(self):
            if self.isTrain:
                startT = '2020-05-01'
                endT = '2020-11-01'
            else:
                startT = '2020-04-01'
                endT = '2020-05-01'
            dataloader = data_loader_clouds.DatasetHandling(128, 128, startT=startT, endT=endT,
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
            plt.plot(np.unique(dataA), np.poly1d(np.polyfit(dataA, dataP, 1))(np.unique(dataA)))
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
            plt.plot(np.unique(dataA), np.poly1d(np.polyfit(dataA, dataP, 1))(np.unique(dataA)))
            plt.legend()
            plt.savefig(self.dirName + 'NDVIComparisonAllMachines_' + str(socket.gethostname()[-3:]) +'.png')
            plt.close()
            return

    def PrintErrors(self, clip_image = True, tirb = -2):
        # YYMMDD
        dataloader = data_loader_clouds.DatasetHandling(128, 128, startT='2020-04-01', endT='2020-05-01',
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



    def PrintErrorsPerband(self):
        # YYMMDD
        if self.isTrain:
            startT = '2020-05-01'
            endT = '2020-11-01'
        else:
            startT = '2020-04-01'
            endT = '2020-05-01'
        dataloader = data_loader_clouds.DatasetHandling(128, 128, startT=startT, endT=endT,
                                           album='iowa-2015-2020-spa', no_of_timesteps=self.timestamps, batch_size=1, istrain=self.isTrain)

        psnrsRed,psnrsBlue,psnrsGreen,psnrsNIR,psnrsSWIR1,psnrsSWIR2, psnrsTIR1, psnrsTIR2 = [], [], [], [], [], [], [], []
        test_itr = dataloader.load_landsat_images(batch_size=1, all_black_clouds=True)
        count = 0
        while True:
            try:
                (cloudy_img_land, target_cloudfree_land, inp_prev_sent, inp_prev_landsat, _) = next(test_itr)
                fakeImg = (dataloader.denormalize11(
                    self.generator.predict([cloudy_img_land, inp_prev_landsat, inp_prev_sent])[-1])).astype(np.uint8)

                target_cloudfree_land = dataloader.denormalize11(target_cloudfree_land[-1][:,:,:8]).astype(np.uint8)    # Returns image pixels between (0, 255)

                bandDic = {0:psnrsRed, 1:psnrsBlue, 2:psnrsGreen, 3:psnrsNIR, 4:psnrsSWIR1, 5:psnrsSWIR2, 6:psnrsTIR1, 7:psnrsTIR2}

                for i in range(0, len(bandDic)):
                    if i == 6:
                        psnr, _ = self.psnrAndRmse(np.clip(fakeImg[:, :, i], 190, None), target_cloudfree_land[:, :, i])
                    psnr, _ = self.psnrAndRmse(fakeImg[:,:,i], target_cloudfree_land[:,:,i])
                    oldpsnr = bandDic.get(i)
                    oldpsnr.append(psnr)
                    bandDic[i] = oldpsnr
                count +=1
            except StopIteration:
                break

        print("Total Testing sample: ", count)
        allPsnr = [psnrsRed, psnrsBlue, psnrsGreen, psnrsNIR, psnrsSWIR1, psnrsSWIR2, psnrsTIR1, psnrsTIR2]
        bandsN = ['Red', 'Blue', 'Green', 'NIR', 'SWIR1', 'SWIR2', 'TIR1', 'TIR2']

        with open(self.dirName + 'psnrPerBandTesting_' + str(socket.gethostname()[-3:]) + '.txt', 'wb') as f:
            pickle.dump(allPsnr, f)

        fig, ax = plt.subplots()
        plt.style.use("ggplot")
        ax.boxplot(allPsnr,  notch=True, patch_artist=True)
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

if __name__ == '__main__':

    train_helpers=train_helpers(folderI = 100,timestamps = 1, batch_size = 5,  lossMethod='carl', onlyPredict=False,
                                istrain=True, modelN='res', loadModel=True)

    # train_helpers.display_training_images(epoch = 1, is_train=True)
    # countI = 0
    # while True:
    #     try:
    #         cloudy_img_land, target_cloudfree_land, cloud_cov_perc, inp_prev_sent, inp_prev_landsat = train_helpers.test_itr.__next__()
    #         countI += 1
    #     except StopIteration:
    #         print("total umage e:", countI)
    #        train_helpers.display_training_images(countI, is_train=True)
    #        print("cloudy_img_land : {} \n target_cloudfree_land : {} \n cloud_cov_perc : {} \n inp_prev_sent : {} \n inp_prev_landsat : {}".format(cloudy_img_land.shape, target_cloudfree_land.shape, cloud_cov_perc.shape, inp_prev_sent.shape, inp_prev_landsat.shape))

    # print("Total images found: {} ".format(countI))
    # train_helpers.predictionLosses()
    # train_helpers.PrintErrorsPerbandGlobal()
    # train_helpers.getErrorsNDVIGlobal()
    # train_helpers.PrintErrorsPerband()
    # train_helpers.getErrorsNDVI()
    # train_helpers.PrintErrors(clip_image=True, tirb=-2)
    train_helpers.trainOnlyGen(epoch_end=501, sample_interval=5)
