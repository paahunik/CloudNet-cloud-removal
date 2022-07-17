import math

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.keras.backend as K
import numpy as np


def cloud_mean_absolute_error(y_true, y_pred):
    """Computes the MAE over the full image."""
    return K.mean(K.abs(y_pred[:, :, :, 0:8] - y_true[:, :, :, 0:8]))


def cloud_mean_squared_error(y_true, y_pred):
    """Computes the MSE over the full image."""
    return K.mean(K.square(y_pred[:, :, :, 0:8] - y_true[:, :, :, 0:8]))


def cloud_root_mean_squared_error(y_true, y_pred):
    """Computes the RMSE over the full image."""
    return K.sqrt(K.mean(K.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :])))


def cloud_bandwise_root_mean_squared_error(y_true, y_pred):
    return K.mean(K.sqrt(K.mean(K.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]), axis=[2, 3])))


def cloud_ssim(y_true, y_pred):
    """Computes the SSIM over the full image."""
    y_true = y_true[:, 0:13, :, :]
    y_pred = y_pred[:, 0:13, :, :]

    y_true *= 2000
    y_pred *= 2000

    y_true = tf.transpose(y_true, [0, 2, 3, 1])
    y_pred = tf.transpose(y_pred, [0, 2, 3, 1])

    ssim = tf.image.ssim(y_true, y_pred, max_val=10000.0)
    ssim = tf.reduce_mean(ssim)

    return ssim


def get_sam(y_true, y_predict):
    """Computes the SAM array."""
    mat = tf.math.multiply(y_true, y_predict)
    mat = tf.math.reduce_sum(mat, 1)
    mat = tf.math.divide(mat, K.sqrt(tf.reduce_sum(tf.multiply(y_true, y_true), 1)))
    mat = tf.math.divide(mat, K.sqrt(tf.reduce_sum(tf.multiply(y_predict, y_predict), 1)))
    mat = tf.math.acos(K.clip(mat, -1, 1))

    return mat


def cloud_mean_sam(y_true, y_predict):
    """Computes the SAM over the full image."""
    mat = get_sam(y_true[:, 0:13, :, :], y_predict[:, 0:13, :, :])

    return tf.reduce_mean(mat)


    
def cloud_mean_sam_covered(y_true, y_pred):
    """Computes the SAM over the covered image parts."""
    cloud_cloudshadow_mask = y_true[:, -1:, :, :]
    target = y_true[:, 0:13, :, :]
    predicted = y_pred[:, 0:13, :, :]

    if K.sum(cloud_cloudshadow_mask) == 0:
        return 0.0

    sam = get_sam(target, predicted)
    sam = tf.expand_dims(sam, 1)
    sam = K.sum(cloud_cloudshadow_mask * sam) / K.sum(cloud_cloudshadow_mask)

    return sam


def cloud_mean_sam_clear(y_true, y_pred):
    """Computes the SAM over the clear image parts."""
    clearmask = K.ones_like(y_true[:, -1:, :, :]) - y_true[:, -1:, :, :]
    predicted = y_pred[:, 0:13, :, :]
    input_cloudy = y_pred[:, -14:-1, :, :]

    if K.sum(clearmask) == 0:
        return 0.0

    sam = get_sam(input_cloudy, predicted)
    sam = tf.expand_dims(sam, 1)
    sam = K.sum(clearmask * sam) / K.sum(clearmask)

    return sam


def cloud_psnr(y_true, y_predict):
    """Computes the PSNR over the full image."""
    y_true *= 2000
    y_predict *= 2000
    #rmse = K.sqrt(K.mean(K.square(y_predict[:, 0:13, :, :] - y_true[:, 0:13, :, :])))
    rmse = K.sqrt(K.mean(K.square(y_predict[:, :,:,0:8] - y_true[:,:,:, 0:8])))

    return 20.0 * (K.log(10000.0 / rmse) / K.log(10.0))


def cloud_mean_absolute_error_clear(y_true, y_pred):
    """Computes the SAM over the clear image parts."""
    clearmask = K.ones_like(y_true[:, -1:, :, :]) - y_true[:, -1:, :, :]
    predicted = y_pred[:, 0:13, :, :]
    input_cloudy = y_pred[:, -14:-1, :, :]

   
    if K.sum(clearmask) == 0:
        return 0.0

    clti = clearmask * K.abs(predicted - input_cloudy)
    clti = K.sum(clti) / (K.sum(clearmask) * 13)

    return clti


def cloud_mean_absolute_error_covered(y_true, y_pred):
    """Computes the SAM over the covered image parts."""
    cloud_cloudshadow_mask = y_true[:, -1:, :, :]
    predicted = y_pred[:, 0:13, :, :]
    target = y_true[:, 0:13, :, :]

    if K.sum(cloud_cloudshadow_mask) == 0:
        return 0.0

    ccmaec = cloud_cloudshadow_mask * K.abs(predicted - target)
    ccmaec = K.sum(ccmaec) / (K.sum(cloud_cloudshadow_mask) * 13)

    return ccmaec


def carl_error(y_true, y_pred):
    """Computes the Cloud-Adaptive Regularized Loss (CARL)"""
    cloud_cloudshadow_mask = y_true[:, :, :, -1:]
    clearmask = K.ones_like(y_true[:, :, :, -1:]) - y_true[:, :, :, -1:]
    predicted = y_pred[:, :, :, 0:8]
    input_cloudy = y_pred[:, :, :, 0:8]
    target = y_true[:, :, :, 0:8]

    cscmae = K.mean(clearmask * K.abs(predicted - target) + cloud_cloudshadow_mask * K.abs(
        predicted - target)) + 1.0 * K.mean(K.abs(predicted - target))

    return cscmae

def denormalize11(imgs):  # converting -1 to 1 values to 0-255
    return (imgs + 1.) * 127.5

def rmseM(image1, image2):
    return K.sqrt(K.mean((image2 - image1) ** 2))

def psnr(target, ref):
    target = denormalize11(target[:,:,:,0:8])
    ref = denormalize11(ref[:,:,:,0:8])
    rmseV = rmseM(target, ref)
    return 20*(K.log(255.0 / rmseV) / K.log(10.0))