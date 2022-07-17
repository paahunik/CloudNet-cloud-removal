import tensorflow.keras.backend as K
import tensorflow as tf
import horovod.tensorflow.keras as hvd
hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
from tensorflow.keras.layers import Conv2D, Concatenate, Activation, Lambda, Add
from tensorflow.keras import Model, Input
K.set_image_data_format('channels_last')


def resBlock(input_l, layer_output_feature_size, kernel_size, scale=0.1):
    tmp = Conv2D(layer_output_feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(input_l)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(layer_output_feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(tmp)
    tmp = Lambda(lambda x: x * scale)(tmp)
    return Add()([input_l, tmp])

def resNet_LandSat_model(input_shape,num_layers=32,layer_output_feature_size=256):
    input_data = Input(shape=input_shape)
    x = input_data
    x = Conv2D(layer_output_feature_size, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
    x = Activation('relu')(x)
    for i in range(num_layers):
        x = resBlock(x, layer_output_feature_size, kernel_size=[3, 3])
    x = Conv2D(input_shape[2], (3, 3), kernel_initializer='he_uniform', padding='same')(x)
    x = Add()([x, input_data])
    model = Model(inputs=input_data, outputs=x)
    return model

def carl_error(y_true, y_pred):
        cloud_cloudshadow_mask = y_true[:, :, :, -1:]
        clearmask = K.ones_like(y_true[:, :, :, -1:]) - y_true[:, :, :, -1:]
        predicted = y_pred[:, :, :, 0:8]
        target = y_true[:, :, :, 0:8]
        cscmae = K.mean(clearmask * K.abs(predicted - target) + cloud_cloudshadow_mask * K.abs(
            predicted - target)) + 1.0 * K.mean(K.abs(predicted - target))
        return cscmae

if __name__ == '__main__':
    num_layers = 16
    feature_size = 256
    crop_size = 128
    in_channels = 8
    input_shape = (crop_size, crop_size, in_channels)
    lr = 0.0001

    model = resNet_LandSat_model(input_shape,num_layers=num_layers,layer_output_feature_size=feature_size)
    optimizer = Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    optimizer = hvd.DistributedOptimizer(optimizer, backward_passes_per_step=1, average_aggregated_gradients=True)

    model.compile(optimizer=optimizer, loss=carl_error)
