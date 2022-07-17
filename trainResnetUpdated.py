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
    Activation,Dropout, RepeatVector, Input,\
    Conv2DTranspose, LeakyReLU,Reshape, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from scipy.stats.stats import pearsonr
import os
import numpy as np
np.seterr(divide='ignore', invalid='ignore', over= 'ignore')
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

    def getGeneratorModel2(self):
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

    def saveModel(self, model=None):
        model_json = self.generator.to_json()
        self.generator.save_weights(self.dirName + "GeneratorModel.h5")
        with open(self.dirName + "GeneratorModel.json", "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk")
        return

    def mse(self, img1, img2):
        return np.mean((img2.astype(np.float64) - img1.astype(np.float64)) ** 2)

    def load_model(self, dir=None):
        if dir is not None:
            dirname = "/s/" + socket.gethostname() + "/a/nobackup/galileo/paahuni/cloud_removal/" + str(dir)  +"/"
        else:
            dirname = self.dirName
        json_file = open( dirname + "GeneratorModel.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
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

    def display_training_images(self, epoch,
                                landsat_batch_x_cloudy, land_cloudy_with_clouds_batch, landsat_batch_y_cloud_free, sent_batch, landsat_prev_batch,  geo_batch, cc_batch, is_train=True):
        if is_train:
            output_dir = self.dirName + 'train/'
        else:
            output_dir = self.dirName + 'test/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        r, c = 2, 4

        fakeImg = (self.dataloader.denormalize11(self.generator.predict([landsat_batch_x_cloudy,landsat_prev_batch, sent_batch])[-1])).astype(np.uint8)
        cloudy_img_land = self.dataloader.denormalize11(landsat_batch_x_cloudy[-1]).astype(np.uint8)
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

        psnr,rmse = self.psnrAndRmse(fakeImg[:,:,:3], target_cloudfree_land[:,:,:3])

        plt.suptitle("Geohash: {} CloudCov: {}% Target MSE: {} PSNR: {} RMSE: {}".format( geo_batch[-1],  str(round(float(cc_batch[-1]) * 100, 3)),
            round(self.mse(self.normal01(fakeImg[:,:,:3]), self.normal01(target_cloudfree_land[:,:,:3])), 3), round(psnr,3), round(rmse, 3)), fontsize=20)

        fig.savefig(output_dir + "%s.png" % (epoch))
        plt.close()

    def trainOnlyGen(self, epoch_end, epoch_start=0, sample_interval=10, batchS=1000):
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

        test_itr = self.dataloader.load_iterator_from_paths( batch_size=1)

        for _ in range(4):
            landsat_batch_x_cloudy1, land_cloudy_with_clouds_batch1, landsat_batch_y_cloud_free1, sent_batch1, landsat_prev_batch1, geo_batch1, cc_batch1  = test_itr.__next__()

        train_itr = self.dataloader.load_iterator_from_paths_for_training(resize_image=True, batch_size=self.batch_size)

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
                                                                                                  sent_batch1, landsat_prev_batch1, geo_batch1, cc_batch1, is_train=True)
                        self.saveModel(self.generator)
                        self.saveLoss(global_loss, global_psnr, istesting=False)
                        self.loadplotlosses()

        self.saveModel(self.generator)
        self.saveLoss(global_loss,global_psnr, istesting= False)
        self.loadplotlosses()

if __name__ == '__main__':

    train_helpers=train_helpers(folderI = 1,timestamps = 1, batch_size = 10, lossMethod='carl', onlyPredict=False,
                                istrain=True, tranferLear=None, loadModel=True)
    train_helpers.trainOnlyGen(epoch_end=1002, sample_interval=10, batchS=1000)