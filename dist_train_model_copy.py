import argparse
import random
import tensorflow as tf
import socket
import tensorflow.keras.backend as K
import numpy as np
from resNetModel import resNet_LandSat_model
from data_loader_clouds_new import DatasetHandling
import image_met_landsat as img_met
from tensorflow.keras.optimizers import Nadam
import horovod.tensorflow.keras as hvd
import os
import socket
import gdal
import datetime

K.set_image_data_format('channels_last')
machine_dict={ 'char':1, 'anchovy':2  , 'bullhead':3, 'dorado':4, 'flounder':5,  'grouper':6, 'halibut':7,   'herring':8,   'mackerel':9,
                'marlin':10, 'perch':11,  'pollock':12,  'sardine':13,  'swordfish':14,   'wahoo':0}

def run_landsat_resNet(epochs=10):
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    
    scaled_lr = 0.0001
    callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=10, verbose=1)]

    verbose = 1 if hvd.rank() == 0 else 0
    hostName = socket.gethostname()
    output_path = '/s/chopin/f/proj/fineET/outputs2/machine-' + str(machine_dict[hostName])
    if hvd.rank() == 0:
       filePath = output_path+'/checkpoint-{epoch:02d}.h5'
       callbacks.append(tf.keras.callbacks.ModelCheckpoint(filePath))

    logPath = '/s/chopin/f/proj/fineET/outputs2/logs/'+'machine-'+str(machine_dict[hostName])+'-training-'+str(hvd.rank())
    callbacks.append(tf.keras.callbacks.CSVLogger(logPath))
    num_layers = 16
    feature_size = 256
    crop_size = 128
    loss = img_met.carl_error #cloud_root_mean_squared_error
    metrics = [img_met.psnr, img_met.cloud_mean_squared_error, img_met.cloud_mean_absolute_error]

    lr = 0.0001
    optimizer = Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)#, schedule_decay=0.004)
    optimizer = hvd.DistributedOptimizer(optimizer, backward_passes_per_step=1, average_aggregated_gradients=True)
    in_channels = 8
    input_shape = (crop_size, crop_size,in_channels)
    random_seed_general = 42
    random.seed(random_seed_general)
    np.random.seed(random_seed_general)
    tf.random.set_seed(random_seed_general)
    model = resNet_LandSat_model(input_shape,
                                       num_layers=num_layers,
                                       layer_output_feature_size=feature_size,
                                       )
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    w= input_shape[0]
    h=input_shape[1]
    train_itr = DatasetHandling(w, h, batch_size=1,cluster_size=15,node_rank=machine_dict[hostName])
    model.fit_generator(train_itr.load_landsat_images(all_black_clouds=True),epochs=epochs, steps_per_epoch=1300,callbacks=callbacks, verbose=verbose)

if __name__ == '__main__':
    run_landsat_resNet(epochs=200)
