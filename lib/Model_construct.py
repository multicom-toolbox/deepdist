# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:41:28 2019

@author: Zhiye
"""

from six.moves import range
import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import Input, Dense, Reshape, Activation, Flatten, Dropout, Lambda, add, concatenate, Concatenate, average, multiply
from keras.layers import GlobalAveragePooling2D, Permute
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.activations import tanh, softmax
from keras import metrics, initializers, utils, regularizers
import numpy as np

import tensorflow as tf
import sys
sys.setrecursionlimit(10000)

# Helper to build a conv -> BN -> relu block
def _bn_relu(input):
    norm = BatchNormalization(axis=-1)(input)
    return Activation("relu")(norm)

def _in_relu(input):
    norm = InstanceNormalization(axis=-1)(input)
    return Activation("relu")(norm)

def _in_elu(input):
    norm = InstanceNormalization(axis=-1)(input)
    return Activation("elu")(norm)

def _in_elu_conv2D(filters,  nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal",  kernel_regularizer=None, dilation_rate=(1,1)):
    def f(input):
        act = _in_elu(input)
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", kernel_regularizer=kernel_regularizer, dilation_rate = dilation_rate)(act)
        return conv
    return f

def _conv_bn_relu2D(filters,  nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal",  kernel_regularizer=None, dilation_rate=(1,1)):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", kernel_regularizer=kernel_regularizer, dilation_rate = dilation_rate)(input)
        # norm = BatchNormalization(axis=-1)(conv)
        norm = BatchNormalization(axis=-1)(conv)
        return Activation("relu")(norm)
    return f

def _conv_in_relu2D(filters,  nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal",  kernel_regularizer=None, dilation_rate=(1,1)):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", kernel_regularizer=kernel_regularizer, dilation_rate = dilation_rate)(input)
        norm = InstanceNormalization(axis=-1)(conv)
        return Activation("relu")(norm)
    return f

def _in_sigmoid(input):
    norm = InstanceNormalization(axis=-1)(input)
    return Activation("sigmoid")(norm)

def _conv_in_sigmoid2D(filters, nb_row, nb_col, strides=(1, 1), use_bias=True, kernel_initializer = "he_normal", dilation_rate=(1, 1)):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=strides,use_bias=use_bias,
                             kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(input)
        norm = InstanceNormalization(axis=-1)(conv)
        return Activation("sigmoid")(conv)
    
    return f

def _weighted_mean_squared_error(weight):

    def loss(y_true, y_pred):
        #set 20A as thresold
        # y_bool = Lambda(lambda x: x <= 20.0)(y_pred)
        y_bool = K.cast((y_true <= 10.0), dtype='float32') # 16.0
        y_bool_invert = K.cast((y_true > 10.0), dtype='float32')
        y_mean = K.mean(y_true)
        y_pred_below = y_pred * y_bool 
        y_pred_upper = y_pred * y_bool_invert 
        y_true_below = y_true * y_bool 
        y_true_upper = y_true * y_bool_invert 
        weights1 = weight
        # weights2 = 0# here need confirm whether use mean or constant
        weights2 = 1/(1 + K.square(y_pred_upper/y_mean))
        return K.mean(K.square((y_pred_below-y_true_below))*weights1) + K.mean(K.square((y_pred_upper-y_true_upper))*weights2)
        # return add([K.mean(K.square((y_pred_below-y_true_below))*weights1), K.mean(K.square((y_pred_upper-y_true_upper))*weights2)], axis= -1)
    return loss

def MaxoutAct(input, filters, kernel_size, output_dim, padding='same', activation = "relu"):
    output = None
    for _ in range(output_dim):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(input)
        activa = Activation(activation)(conv)
        maxout_out = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(activa)
        if output is not None:
            output = concatenate([output, maxout_out], axis=-1)
        else:
            output = maxout_out
    return output

class InstanceNormalization(Layer):
    def __init__(self, axis=-1, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis '+str(self.axis)+' of input tensor should have a defined dimension but the layer received an input with shape '+str(input_shape)+ '.')
        shape = (dim,)

        self.gamma = self.add_weight(shape=shape, name='gamma', initializer=initializers.random_normal(1.0, 0.02))
        self.beta = self.add_weight(shape=shape, name='beta', initializer='zeros')
        self.built = True

    def call(self, inputs, training=None):
        mean, var = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)
        return K.batch_normalization(inputs, mean, var, self.beta, self.gamma, self.epsilon)

class RowNormalization(Layer):
    def __init__(self, axis=-1, epsilon=1e-5, **kwargs):
        super(RowNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis '+str(self.axis)+' of input tensor should have a defined dimension but the layer received an input with shape '+str(input_shape)+ '.')
        shape = (dim,)

        self.gamma = self.add_weight(shape=shape, name='gamma', initializer=initializers.random_normal(1.0, 0.02))
        self.beta = self.add_weight(shape=shape, name='beta', initializer='zeros')
        self.built = True

    def call(self, inputs, training=None):
        mean, var = tf.nn.moments(inputs, axes=[1], keep_dims=True)
        return K.batch_normalization(inputs, mean, var, self.beta, self.gamma, self.epsilon)

class ColumNormalization(Layer):
    def __init__(self, axis=-1, epsilon=1e-5, **kwargs):
        super(ColumNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis '+str(self.axis)+' of input tensor should have a defined dimension but the layer received an input with shape '+str(input_shape)+ '.')
        shape = (dim,)

        self.gamma = self.add_weight(shape=shape, name='gamma', initializer=initializers.random_normal(1.0, 0.02))
        self.beta = self.add_weight(shape=shape, name='beta', initializer='zeros')
        self.built = True

    def call(self, inputs, training=None):
        mean, var = tf.nn.moments(inputs, axes=[2], keep_dims=True)
        return K.batch_normalization(inputs, mean, var, self.beta, self.gamma, self.epsilon)


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3

def _shortcut(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = 1
    stride_height = 1
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    if not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal")(input)
    return add([shortcut, residual])


def squeeze_excite_block(input, ratio=16):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)
    x = multiply([init, se])
    return x

def _in_relu_K(x, bn_name=None, relu_name=None):
    # norm = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name)(x)
    norm = InstanceNormalization(axis=-1, name=bn_name)(x)
    return Activation("relu", name=relu_name)(norm)

def _rcin_relu_K(x, bn_name=None, relu_name=None):
    norm1 = InstanceNormalization(axis=-1, name=bn_name)(x)
    norm2 = RowNormalization(axis=-1, name=bn_name)(x)
    norm3 = ColumNormalization(axis=-1, name=bn_name)(x)
    norm  = concatenate([norm1, norm2, norm3])
    return Activation("relu", name=relu_name)(norm)

def _dilated_residual_block(block_function, filters, repetitions, is_first_layer=False, dilation_rate=(1,1), use_SE = False):
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                # init_strides = (2, 2)
                init_strides = (1, 1)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0),  dilation_rate=dilation_rate[i], use_SE = use_SE)(input)
        return input

    return f
    
def dilated_bottleneck_rc(filters, init_strides=(1, 1), is_first_block_of_first_layer=False, dilation_rate=(1,1), use_SE = False):
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=regularizers.l2(1e-4))(input)
        else:
            # conv_1_1 = _rcin_relu_K(input) 
            conv_1_1 = _rcin_relu_K(input) 
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1), strides=init_strides,padding="same",kernel_initializer="he_normal")(conv_1_1)

        conv_3_3 = _rcin_relu_K(conv_1_1) 
        conv_3_3 = Conv2D(filters=filters, kernel_size=(3, 3), strides=init_strides, padding="same", kernel_initializer="he_normal")(conv_3_3)
        conv_7_1 = Conv2D(filters=filters, kernel_size=(7, 1), strides=init_strides, padding="same", kernel_initializer="he_normal")(conv_3_3)
        conv_1_7 = Conv2D(filters=filters, kernel_size=(1, 7), strides=init_strides, padding="same", kernel_initializer="he_normal")(conv_3_3)
        conv_3_3 = concatenate([conv_3_3, conv_7_1, conv_1_7])
        # conv_3_3 = _in_elu_conv2D(filters=filters, nb_row=3, nb_col=3, dilation_rate=dilation_rate)(conv_1_1)
        # residual = _in_elu_conv2D(filters=filters * 2, nb_row=1, nb_col=1)(conv_3_3)
        residual = _rcin_relu_K(conv_3_3) 
        residual = Conv2D(filters=filters, kernel_size=(1, 1), strides=init_strides, padding="same", kernel_initializer="he_normal")(residual)
        if use_SE == True:
            residual = squeeze_excite_block(residual)
        return _shortcut(input, residual)
    return f

def DeepDistRes_with_paras_2D(kernel_size,feature_2D_num, filters,nb_layers,opt, initializer = "he_normal", loss_function = "categorical_crossentropy", weight_p=1.0, weight_n=1.0):
    _handle_dim_ordering()
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(None,None,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)
    
    DeepDist_2D_input = contact_input
    DeepDist_2D_conv = DeepDist_2D_input
    DeepDist_2D_conv = InstanceNormalization(axis=-1)(DeepDist_2D_conv)
    DeepDist_2D_conv = Conv2D(128, 1, padding = 'same')(DeepDist_2D_conv)
    # DeepDist_2D_conv = Dense(64)(DeepDist_2D_conv)
    DeepDist_2D_conv = MaxoutAct(DeepDist_2D_conv, filters=4, kernel_size=(1,1), output_dim=64, padding='same', activation = "elu")

    # ######This is original residual
    # DeepDist_2D_conv = _conv_in_relu2D(filters=64, nb_row=7, nb_col=7, strides=(1, 1))(DeepDist_2D_conv)

    # DeepDist_2D_conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1,1),use_bias=True, kernel_initializer=initializer, padding="same")(DeepDist_2D_conv)
    block = DeepDist_2D_conv
    dilated_num = [1, 2, 4, 8, 1] * 4
    # repetitions = [3, 4, 6, 3]
    repetitions = [20]
    for i, r in enumerate(repetitions):
        block = _dilated_residual_block(dilated_bottleneck_rc, filters=filters, repetitions=r, is_first_layer=(i == 0), dilation_rate =dilated_num, use_SE = True)(block)
        block = Dropout(0.2)(block)
    # Last activation
    block = _rcin_relu_K(block)
    DeepDist_2D_conv = block
    if loss_function == 'mul_class_and_real_dist_G':
        DeepDist_2D_conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),use_bias=True,
                         kernel_initializer=initializer, padding="same", kernel_regularizer=None)(DeepDist_2D_conv)
        DeepDist_2D_conv = InstanceNormalization(axis=-1)(DeepDist_2D_conv)
        mul_class = Dense(42, activation='softmax', name='mul_class')(DeepDist_2D_conv) #3.5-19 21 3.5-16 18 3.5-24 26
        mul_class_loss= 'categorical_crossentropy'
        
        real_dist = Conv2D(filters=1, kernel_size=1, strides=1,use_bias=False, padding="same", activation='relu', name='real_dist')(DeepDist_2D_conv)
        real_dist_loss = 'mean_squared_error'
        loss={'mul_class':'categorical_crossentropy', 'real_dist':'mean_squared_error'}

        DeepDist_RES = Model(inputs=contact_input, outputs=[mul_class, real_dist])
    else:
        if loss_function == 'binary_crossentropy':
            DeepDist_2D_conv = _conv_in_sigmoid2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DeepDist_2D_conv)
            loss = loss_function
        elif loss_function == 'categorical_crossentropy_D':
            DeepDist_2D_conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),use_bias=True,
                                 kernel_initializer=initializer, padding="same", kernel_regularizer=None)(DeepDist_2D_conv)
            DeepDist_2D_conv = InstanceNormalization(axis=-1)(DeepDist_2D_conv)
            DeepDist_2D_conv = Dense(33, activation='softmax')(DeepDist_2D_conv) 
            loss= 'categorical_crossentropy'
        elif loss_function == 'categorical_crossentropy_G':
            DeepDist_2D_conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),use_bias=True,
                                 kernel_initializer=initializer, padding="same", kernel_regularizer=None)(DeepDist_2D_conv)
            DeepDist_2D_conv = InstanceNormalization(axis=-1)(DeepDist_2D_conv)
            DeepDist_2D_conv = Dense(42, activation='softmax')(DeepDist_2D_conv) 
            loss= 'categorical_crossentropy'
        elif loss_function == 'categorical_crossentropy_T':
            DeepDist_2D_conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),use_bias=True,
                                 kernel_initializer=initializer, padding="same", kernel_regularizer=None)(DeepDist_2D_conv)
            DeepDist_2D_conv = InstanceNormalization(axis=-1)(DeepDist_2D_conv)
            DeepDist_2D_conv = Dense(38, activation='softmax')(DeepDist_2D_conv) 
            loss= 'categorical_crossentropy'
        elif loss_function == 'categorical_crossentropy_C':
            DeepDist_2D_conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1),use_bias=True,
                                 kernel_initializer=initializer, padding="same", kernel_regularizer=None)(DeepDist_2D_conv)
            DeepDist_2D_conv = InstanceNormalization(axis=-1)(DeepDist_2D_conv)
            DeepDist_2D_conv = Dense(10, activation='softmax')(DeepDist_2D_conv) 
            loss= 'categorical_crossentropy'
        elif loss_function == 'real_dist':
            DeepDist_2D_conv = _conv_bn_relu2D(filters=1, nb_row=1, nb_col=1, strides=(1, 1), kernel_initializer=initializer)(DeepDist_2D_conv)
            loss = 'mean_squared_error'    
        DeepDist_2D_out = DeepDist_2D_conv
        DeepDist_RES = Model(inputs=contact_input, outputs=DeepDist_2D_out)
    DeepDist_RES.compile(loss=loss, metrics=['accuracy'], optimizer=opt)
    DeepDist_RES.summary()
    return DeepDist_RES

