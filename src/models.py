import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras import backend as K

import numpy as np
import pandas as pd
    #system
import os
import sys
import csv

import math
import random
import matplotlib
import matplotlib.pyplot as plt

def squash(s, axis=-1, name="squash"):
    """
    non-linear squashing function to manipulate the length of the capsule vectors
    :param s: input tensor containing capsule vectors
    :param axis: If `axis` is a Python integer, the input is considered a batch of vectors,
      and `axis` determines the axis in `tensor` over which to compute squash.
    :return: a Tensor with same shape as input vectors
    """
    with tf.name_scope(name):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                         keepdims=True)
        safe_norm = tf.sqrt(squared_norm + keras.backend.epsilon())
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector
        
def safe_norm(s, axis=-1, keepdims=False, name="safe_norm"):
    """
    Safe computation of vector 2-norm
    :param s: input tensor
    :param axis: If `axis` is a Python integer, the input is considered a batch 
      of vectors, and `axis` determines the axis in `tensor` over which to 
      compute vector norms.
    :param keepdims: If True, the axis indicated in `axis` are kept with size 1.
      Otherwise, the dimensions in `axis` are removed from the output shape.
    :param name: The name of the op.
    :return: A `Tensor` of the same type as tensor, containing the vector norms. 
      If `keepdims` is True then the rank of output is equal to
      the rank of `tensor`. If `axis` is an integer, the rank of `output` is 
      one less than the rank of `tensor`.
    """
    with tf.name_scope(name):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                 keepdims=keepdims)
        return tf.sqrt(squared_norm + keras.backend.epsilon())    
        
class SecondaryCapsule(keras.layers.Layer):
    """
    The Secondary Capsule layer With Dynamic Routing Algorithm. 
    input shape = [None, input_num_capsule, input_dim_capsule] 
    output shape = [None, num_capsule, dim_capsule]
    :param n_caps: number of capsules in this layer
    :param n_dims: dimension of the output vectors of the capsules in this layer
    """
    def __init__(self, n_caps, n_dims, routings=2, **kwargs):
        super().__init__(**kwargs)
        self.n_caps = n_caps
        self.n_dims = n_dims
        self.routings = routings
    def build(self, batch_input_shape):
        # transformation matrix
        self.W = self.add_weight(
            name="W", 
            shape=(1, batch_input_shape[1], self.n_caps, self.n_dims, batch_input_shape[2]),
            initializer=keras.initializers.RandomNormal(stddev=0.1))
        super().build(batch_input_shape)
    def call(self, X):
        # predict output vector
        batch_size = tf.shape(X)[0]
        caps1_n_caps = tf.shape(X)[1] 
        W_tiled = tf.tile(self.W, [batch_size, 1, 1, 1, 1])
        caps1_output_expanded = tf.expand_dims(X, -1, name="caps1_output_expanded")
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2, name="caps1_output_tile")
        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, self.n_caps, 1, 1], name="caps1_output_tiled")
        caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")
        
        # rounting by agreement
        # routing weights
        raw_weights = tf.zeros([batch_size, caps1_n_caps, self.n_caps, 1, 1],
                       dtype=np.float32, name="raw_weights")
        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            routing_weights = tf.nn.softmax(raw_weights, axis=2, name="routing_weights")
            weighted_predictions = tf.multiply(routing_weights, caps2_predicted, name="weighted_predictions")
            weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True, name="weighted_sum")
            caps2_output_round_1 = squash(weighted_sum, axis=-2, name="caps2_output_round_")
            caps2_output_squeezed = tf.squeeze(caps2_output_round_1, axis=[1,4], name="caps2_output_squeezed")
            if i < self.routings - 1:
                caps2_output_round_1_tiled = tf.tile(
                                        caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
                                        name="caps2_output_tiled_round_")
                agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                      transpose_a=True, name="agreement")
                raw_weights_round_2 = tf.add(raw_weights, agreement,
                             name="raw_weights_round_")
                raw_weights = raw_weights_round_2
        return caps2_output_squeezed
    def compute_output_shape(self, batch_input_shape):
        return (batch_input_shape[0], self.n_caps, self.n_dims)
    def get_config(self): ### custom layers to be serializable as part of a Functional model
        base_config = super().get_config()
        return {**base_config, 
                "n_caps": self.n_caps, 
                "n_dims": self.n_dims,
                "routings": self.routings}
                
class LengthLayer(keras.layers.Layer):
    """
    Compute the length of capsule vectors.
    inputs: shape=[None, num_capsule, dim_vector]
    output: shape=[None, num_capsule]
    """
    def call(self, X):
        y_proba = safe_norm(X, axis=-1, name="y_proba")
        # return tf.math.divide(y_proba,tf.reshape(tf.reduce_sum(y_proba,-1),(-1,1),name='reshape'),name='Normalising_Probability')
        return y_proba
    def compute_output_shape(self, batch_input_shape): # in case the layer modifies the shape of its input, 
                                                        #you should specify here the shape transformation logic.
                                                        #This allows Keras to do automatic shape inference.
        return (batch_input_shape[0], batch_input_shape[1])
          
class MarginLoss(keras.losses.Loss):
    """
    Compute margin loss.
    y_true shape [None, n_classes] 
    y_pred shape [None, num_capsule] = [None, n_classes]
    """
    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_=0.5, **kwargs):
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_
        super().__init__(**kwargs)
        
    def call(self, y_true, y_proba):
        present_error_raw = tf.square(tf.maximum(0., self.m_plus - y_proba), name="present_error_raw")
        absent_error_raw = tf.square(tf.maximum(0., y_proba - self.m_minus), name="absent_error_raw")
        L = tf.add(y_true * present_error_raw, self.lambda_ * (1.0 - y_true) * absent_error_raw,
           name="L")
        total_marginloss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
        return total_marginloss
    
    def get_config(self): ### custom layers to be serializable as part of a Functional model
        base_config = super().get_config()
        return {**base_config, 
                "m_plus": self.m_plus,
                "m_minus": self.m_minus,
                "lambda_": self.lambda_}
                

def HD_CapsNet(input_shape, input_shape_yc, input_shape_ym, input_shape_yf,
                no_coarse_class = 2, no_medium_class = 7, no_fine_class = 10,
                PCap_n_dims = 8, SCap_f_dims = 16, SCap_m_dims = 16, SCap_c_dims = 16, 
                model_name = 'HD-CapsNet'):
                
    ### Layer-1: Inputs ###

    # Input image
    x_input = keras.layers.Input(shape=input_shape, name="Input_Image")
    
    # Input True Labels
    y_c = keras.layers.Input(shape=input_shape_yc, name='input_yc')
    y_m = keras.layers.Input(shape=input_shape_ym, name='input_ym')
    y_f = keras.layers.Input(shape=input_shape_yf, name='input_yf')
    
    ### Layer-2: Feature Extraction Blocks ###
    #--- block 1 ---
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    #--- block 2 ---
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    #--- block 3 ---
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    #--- block 4 ---
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)


    ### Layer-3: Reshape to 8D primary capsules ###
    
    reshape = keras.layers.Reshape((int((tf.reduce_prod(x.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims), name="reshape_layer")(x)
    p_caps = keras.layers.Lambda(squash, name='p_caps')(reshape)

    ### Layer-4: Secondary Capsules ### 
    #--- Secondary Capsule for fine level ---#
    s_caps_f = SecondaryCapsule(n_caps=no_fine_class, n_dims=SCap_f_dims, 
                        name="s_caps_fine")(p_caps)

    #--- Secondary Capsule for medium level ---#
    s_caps_m = SecondaryCapsule(n_caps=no_medium_class, n_dims=SCap_f_dims, 
                        name="s_caps_medium")(s_caps_f)

    #--- Secondary Capsule for coarse level ---#
    s_caps_c = SecondaryCapsule(n_caps=no_coarse_class, n_dims=SCap_c_dims, 
                        name="s_caps_coarse")(s_caps_m)
                        
    ### Layer-5: Output Layers ### 
    #--- Length for coarse level ---#
    pred_c = LengthLayer(name='prediction_coarse')(s_caps_c)
    #--- Length for medium level ---#
    pred_m = LengthLayer(name='prediction_medium')(s_caps_m)
    #--- Length for fine level ---#
    pred_f = LengthLayer(name='prediction_fine')(s_caps_f)
    
    ### Building Keras Model ###
    model = keras.Model(inputs= [x_input, y_c, y_m, y_f],
                        outputs= [pred_c, pred_m, pred_f],
                        name=model_name)    
    return model
    
def HD_CapsNet_Mod_3_3(input_shape, input_shape_yc, input_shape_ym, 
                        input_shape_yf, no_coarse_class, no_medium_class, no_fine_class,
                        PCap_n_dims = 8, SCap_f_dims = 16, SCap_m_dims = 32, SCap_c_dims = 64):

    # Input image
    x_input = keras.layers.Input(shape=input_shape, name="Input_Image")

    # Input True Labels
    y_c = keras.layers.Input(shape=input_shape_yc, name='input_yc')
    y_m = keras.layers.Input(shape=input_shape_ym, name='input_ym')
    y_f = keras.layers.Input(shape=input_shape_yf, name='input_yf')

    #--- block 1 ---
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    #--- block 2 ---
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    #--- block 3 ---
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    #--- block 4 ---
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)


    # Layer 3: Reshape to 8D primary capsules 
    reshape = keras.layers.Reshape((int((tf.reduce_prod(x.shape[1:]).numpy())/PCap_n_dims),
                                     PCap_n_dims), name="reshape_layer")(x)
    p_caps = keras.layers.Lambda(squash, name='p_caps')(reshape)

    ## Layer Secondary Capsule: For coarse level
    s_caps_c = SecondaryCapsule(n_caps=no_coarse_class, n_dims=SCap_c_dims, 
                        name="s_caps_coarse")(p_caps)

    ## Skip Connection: For Medium Level
    p_caps_m = keras.layers.Reshape((int((tf.reduce_prod(p_caps.shape[1:]).numpy())/s_caps_c.shape[-1]),
                                     s_caps_c.shape[-1]), name="primary_skip_m")(p_caps)
    skip_m = keras.layers.Concatenate(axis=1, name="skip_connection_m")([p_caps_m, s_caps_c])

    ## Layer Secondary Capsule: For medium level
    s_caps_m = SecondaryCapsule(n_caps=no_medium_class, n_dims=SCap_m_dims, 
                        name="s_caps_medium")(skip_m)

    ## Skip Connection: For Fine Level
    p_caps_f = keras.layers.Reshape((int((tf.reduce_prod(p_caps.shape[1:]).numpy())/s_caps_m.shape[-1]),
                                     s_caps_m.shape[-1]), name="primary_skip_f")(p_caps)
    skip_f = keras.layers.Concatenate(axis=1, name="skip_connection_f")([p_caps_f, s_caps_m])

    ## Layer Secondary Capsule: For fine level
    s_caps_f = SecondaryCapsule(n_caps=no_fine_class, n_dims=SCap_f_dims, 
                        name="s_caps_fine")(skip_f)

    pred_c = LengthLayer(name='prediction_coarse')(s_caps_c)

    pred_m = LengthLayer(name='prediction_medium')(s_caps_m)

    pred_f = LengthLayer(name='prediction_fine')(s_caps_f)

    model = keras.Model(inputs= [x_input, y_c, y_m, y_f],
                        outputs= [pred_c, pred_m, pred_f],
                        name='HD-CapsNet')
    ## Return Model
    return model

   
def HD_CapsNet_Mod_3_2(input_shape, input_shape_yc, input_shape_yf, 
                        no_coarse_class, no_fine_class,
                        PCap_n_dims = 8, SCap_f_dims = 16, SCap_c_dims = 32):

    # Input image
    x_input = keras.layers.Input(shape=input_shape, name="Input_Image")

    # Input True Labels
    y_c = keras.layers.Input(shape=input_shape_yc, name='input_yc')
    y_f = keras.layers.Input(shape=input_shape_yf, name='input_yf')

    #--- block 1 ---
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    #--- block 2 ---
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    #--- block 3 ---
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    #--- block 4 ---
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)


    # Layer 3: Reshape to 8D primary capsules 
    reshape = keras.layers.Reshape((int((tf.reduce_prod(x.shape[1:]).numpy())/PCap_n_dims),
                                     PCap_n_dims), name="reshape_layer")(x)
    p_caps = keras.layers.Lambda(squash, name='p_caps')(reshape)

    ## Layer Secondary Capsule: For coarse level
    s_caps_c = SecondaryCapsule(n_caps=no_coarse_class, n_dims=SCap_c_dims, 
                        name="s_caps_coarse")(p_caps)

    ## Skip Connection: For Fine Level
    p_caps_f = keras.layers.Reshape((int((tf.reduce_prod(p_caps.shape[1:]).numpy())/s_caps_c.shape[-1]),
                                     s_caps_c.shape[-1]), name="primary_skip_f")(p_caps)
    skip_f = keras.layers.Concatenate(axis=1, name="skip_connection_f")([p_caps_f, s_caps_c])

    ## Layer Secondary Capsule: For fine level
    s_caps_f = SecondaryCapsule(n_caps=no_fine_class, n_dims=SCap_f_dims, 
                        name="s_caps_fine")(skip_f)

    pred_c = LengthLayer(name='prediction_coarse')(s_caps_c)

    pred_f = LengthLayer(name='prediction_fine')(s_caps_f)

    model = keras.Model(inputs= [x_input, y_c, y_f],
                        outputs= [pred_c, pred_f],
                        name='HD-CapsNet')
    ## Return Model
    return model

def B_CNN_Model_B(input_shape, num_class_c, num_class_m, num_class_f, 
                  model_name:str='B_CNN_Model_B'):
    
    img_input = keras.layers.Input(shape=input_shape, name='input')

    #--- block 1 ---
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    #--- block 2 ---
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    #--- coarse 1 branch ---
    c_bch = keras.layers.Flatten(name='c1_flatten')(x)
    c_bch = keras.layers.Dense(256, activation='relu', name='c_fc_1')(c_bch)
    c_bch = keras.layers.BatchNormalization()(c_bch)
    c_bch = keras.layers.Dropout(0.5)(c_bch)
    c_bch = keras.layers.Dense(256, activation='relu', name='c_fc_2')(c_bch)
    c_bch = keras.layers.BatchNormalization()(c_bch)
    c_bch = keras.layers.Dropout(0.5)(c_bch)
    c_pred = keras.layers.Dense(num_class_c, activation='softmax', name='c_predictions')(c_bch)

    #--- block 3 ---
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    #--- coarse 2 branch ---
    m_bch = keras.layers.Flatten(name='c2_flatten')(x)
    m_bch = keras.layers.Dense(512, activation='relu', name='m_fc_1')(m_bch)
    m_bch = keras.layers.BatchNormalization()(m_bch)
    m_bch = keras.layers.Dropout(0.5)(m_bch)
    m_bch = keras.layers.Dense(512, activation='relu', name='m_fc_2')(m_bch)
    m_bch = keras.layers.BatchNormalization()(m_bch)
    m_bch = keras.layers.Dropout(0.5)(m_bch)
    m_pred = keras.layers.Dense(num_class_m, activation='softmax', name='m_predictions')(m_bch)

    #--- block 4 ---
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    #--- fine block ---
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(1024, activation='relu', name='f_fc_1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1024, activation='relu', name='f_fc2_2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    f_pred = keras.layers.Dense(num_class_f, activation='softmax', name='f_predictions')(x)
    model = keras.Model(img_input, [c_pred, m_pred, f_pred], name=model_name)
    
    return model

def initial_lw(class_labels: dict):
    """
    Give dictionary input for hierarchical levels.
    Where the values for the input keys needs to be total number of classes in the levels.
    Example for 3 levels hierarchy with 2, 7, 10 class will be in following format:
    class_labels = {"coarse": coarse_class, "medium": medium_class,"fine": fine_class}
    :c:The Function will return initial loss weight values in a dictionary, based on number of classes in levels
    """

    q = {}
    for k, v in class_labels.items():
        q[k] = (1 - (v / sum(class_labels.values())))

    c = {}
    for k, v in class_labels.items():
        c[k] = (q[k] / sum(q.values()))
        
    return c
    
class LossWeightsModifier(keras.callbacks.Callback):
    
    def __init__(self, lossweight : dict,initial_lw : dict, directory : str):

        self.lossweight = lossweight
        self.directory = directory
        self.reconstruction_loss = lossweight['decoder_lw']
        
        if 'coarse_lw' in self.lossweight and 'medium_lw' in self.lossweight and 'fine_lw' in self.lossweight:
            self.coarse_lw = lossweight['coarse_lw']
            self.medium_lw = lossweight['medium_lw']
            self.fine_lw = lossweight['fine_lw']
        
            self.c1 = initial_lw['coarse'] # Initial LW for Coarse class
            self.c2 = initial_lw['medium'] # Initial LW for Fine class
            self.c3 = initial_lw['fine'] # Initial LW for Fine class
        
            self.header = ['Epoch',
                            'Coarse_Accuracy', 'Coarse_LossWeight',
                            'Medium_Accuracy', 'Medium_LossWeight',
                            'Fine_Accuracy', 'Fine_LossWeight']
            
        elif 'coarse_lw' in self.lossweight and 'fine_lw' in self.lossweight:

            self.coarse_lw = lossweight['coarse_lw']
            self.fine_lw = lossweight['fine_lw']
        
            self.c1 = initial_lw['coarse'] # Initial LW for Coarse class
            self.c2 = initial_lw['fine'] # Initial LW for Fine class
        
            self.header = ['Epoch',
                            'Coarse_Accuracy', 'Coarse_LossWeight',
                            'Fine_Accuracy', 'Fine_LossWeight']
          
        csv_file = open(self.directory+'/LossWeight.csv', 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(self.header)
        csv_file.close()
    
    def on_epoch_end(self, epoch, logs={}):
    
        if 'coarse_lw' in self.lossweight and 'medium_lw' in self.lossweight and 'fine_lw' in self.lossweight:
            # Taking Training Accuracy for Calculation
            ACC1 = logs.get('prediction_coarse_accuracy')
            ACC2 = logs.get('prediction_medium_accuracy')
            ACC3 = logs.get('prediction_fine_accuracy')
            
            # Taking Validation Accuracy just for printing
            VACC1 = logs.get('val_prediction_coarse_accuracy')
            VACC2 = logs.get('val_prediction_medium_accuracy')
            VACC3 = logs.get('val_prediction_fine_accuracy')
            
            #Calculating Tau Values for each classes
            tau1 = (1.0-ACC1) * self.c1
            tau2 = (1.0-ACC2) * self.c2
            tau3 = (1.0-ACC3) * self.c3
            
            # Updated Loss for each classes
            L1 = float((1-self.reconstruction_loss) * (tau1 / (tau1 + tau2 + tau3)))
            L2 = float((1-self.reconstruction_loss) * (tau2 / (tau1 + tau2 + tau3)))
            L3 = float((1-self.reconstruction_loss) * (tau3 / (tau1 + tau2 + tau3)))
            
            print('\n\033[91m','\033[1m',"\u2022",
                  "Coarse Accuracy =",'{:.2f}%'.format(ACC1*100),"|",
                  "Val_Accuracy =",'{:.2f}%'.format(VACC1*100),"|",
                  "LossWeight =",'{:.2f}'.format(L1),
                  '\033[0m')
            print('\033[91m','\033[1m',"\u2022",
                  "Medium Accuracy =",'{:.2f}%'.format(ACC2*100),"|",
                  "Val_Accuracy =",'{:.2f}%'.format(VACC2*100),"|",
                  "LossWeight =",'{:.2f}'.format(L2),
                  '\033[0m')
            print('\033[91m','\033[1m',"\u2022",
                  "Fine   Accuracy =",'{:.2f}%'.format(ACC3*100),"|",
                  "Val_Accuracy =",'{:.2f}%'.format(VACC3*100),"|",
                  "LossWeight =",'{:.2f}'.format(L3),
                  '\033[0m')
            
            ## Saving Data to CSV FILE##
            data = {'Epoch': epoch,
                    'Coarse_Accuracy': ACC1,
                    'Coarse_LossWeight': L1,
                    'Medium_Accuracy': ACC2,
                    'Medium_LossWeight': L2,
                    'Fine_Accuracy': ACC3,
                    'Fine_LossWeight': L3}
            
            with open(self.directory+'/LossWeight.csv', mode='a', newline='') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames = self.header)
                csv_writer.writerow(data)
                
            #Setting Loss weight Values
            K.set_value(self.coarse_lw, L1)
            K.set_value(self.medium_lw, L2)
            K.set_value(self.fine_lw, L3)
            
        elif 'coarse_lw' in self.lossweight and 'fine_lw' in self.lossweight:

            # Taking Training Accuracy for Calculation
            ACC1 = logs.get('prediction_coarse_accuracy')
            ACC2 = logs.get('prediction_fine_accuracy')
            
            # Taking Validation Accuracy just for printing
            VACC1 = logs.get('val_prediction_coarse_accuracy')
            VACC2 = logs.get('val_prediction_fine_accuracy')
            
            #Calculating Tau Values for each classes
            tau1 = (1.0-ACC1) * self.c1
            tau2 = (1.0-ACC2) * self.c2
            
            # Updated Loss for each classes
            L1 = float((1-self.reconstruction_loss) * (tau1 / (tau1 + tau2)))
            L2 = float((1-self.reconstruction_loss) * (tau2 / (tau1 + tau2)))
            
            print('\n\033[91m','\033[1m',"\u2022",
                  "Coarse Accuracy =",'{:.2f}%'.format(ACC1*100),"|",
                  "Val_Accuracy =",'{:.2f}%'.format(VACC1*100),"|",
                  "LossWeight =",'{:.2f}'.format(L1),
                  '\033[0m')
            print('\033[91m','\033[1m',"\u2022",
                  "Fine   Accuracy =",'{:.2f}%'.format(ACC2*100),"|",
                  "Val_Accuracy =",'{:.2f}%'.format(VACC2*100),"|",
                  "LossWeight =",'{:.2f}'.format(L2),
                  '\033[0m')
            
            ## Saving Data to CSV FILE##
            data = {'Epoch': epoch,
                    'Coarse_Accuracy': ACC1,
                    'Coarse_LossWeight': L1,
                    'Fine_Accuracy': ACC2,
                    'Fine_LossWeight': L2}
            
            with open(self.directory+'/LossWeight.csv', mode='a', newline='') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames = self.header)
                csv_writer.writerow(data)
            
            ## Setting Loss weight Values
            K.set_value(self.coarse_lw, L1)
            K.set_value(self.fine_lw, L2)
             
        
class model_analysis():
        def __init__(self, model: keras.Model, dataset: dict):
            self.dataset = dataset
            self.model = model
        
        def evaluate(self):
                       
            if 'y_test_coarse' in self.dataset and 'y_test_medium' in self.dataset and 'y_test_fine' in self.dataset:
                results = self.model.evaluate([self.dataset['x_test']],[self.dataset['y_test_coarse'],self.dataset['y_test_medium'], self.dataset['y_test_fine']],verbose=1)
                
                for n in range(len(results)):
                    print(str(n+1)+'.',self.model.metrics_names[n], '==>', results[n])
          
            elif 'y_test_coarse' in self.dataset and 'y_test_fine' in self.dataset:
                results = self.model.evaluate([self.dataset['x_test']],[self.dataset['y_test_coarse'],self.dataset['y_test_fine']],verbose=1)
                
                for n in range(len(results)):
                    print(str(n+1)+'.',self.model.metrics_names[n], '==>', results[n])
          
            return results
            
        def prediction(self):
            if 'y_test_coarse' in self.dataset and 'y_test_medium' in self.dataset and 'y_test_fine' in self.dataset:
                predictions = self.model.predict([self.dataset['x_test']],verbose=1)
                
                plot_sample_image(predictions,x_input = self.dataset['x_test'],y_labels = {'coarse':self.dataset['y_test_coarse'], 'medium':self.dataset['y_test_medium'],'fine':self.dataset['y_test_fine']})
                
            elif 'y_test_coarse' in self.dataset and 'y_test_fine' in self.dataset:

                predictions = self.model.predict([self.dataset['x_test']],verbose=1)
                
                plot_sample_image(predictions,x_input = self.dataset['x_test'],y_labels = {'coarse':self.dataset['y_test_coarse'], 'fine':self.dataset['y_test_fine']})
                
            return predictions
            
def plot_sample_image(predictions, x_input, y_labels : dict, no_sample : int = 10, no_column : int = 10):

            input_shape = x_input.shape[1:] # input shape
            
            plot_row= math.ceil(no_sample/no_column)
            random_number = random.sample(range(0, len(x_input)), no_sample)
            fig, axs = plt.subplots(plot_row,no_column, #### Row and column number for no_sample
                                figsize=(20, 20), facecolor='w', edgecolor='k')
                                
            fig.subplots_adjust(hspace = .5, wspace=.001)
            axs = axs.ravel()
            
            for i in range(no_sample):
                sample_image = x_input[random_number[i]].reshape(input_shape)
                axs[i].imshow(sample_image)
                
                if len(y_labels) == 2:
                    axs[i].set_title('(True | Pred)\n Coarse=({0}|{1}),\nFine=({2}|{3})'.format(str(np.argmax(y_labels['coarse'][random_number[i]])),
                    str(np.argmax(predictions[0][random_number[i]])),
                    str(np.argmax(y_labels['fine'][random_number[i]])),
                    str(np.argmax(predictions[1][random_number[i]]))
                    ))
                elif len(y_labels) == 3:
                    axs[i].set_title('(True | Pred)\n Coarse=({0}|{1}),\nMedium=({2}|{3}),\nFine =({4}|{5})'.format(str(np.argmax(y_labels['coarse'][random_number[i]])),
                    str(np.argmax(predictions[0][random_number[i]])),
                    str(np.argmax(y_labels['medium'][random_number[i]])),
                    str(np.argmax(predictions[1][random_number[i]])),
                    str(np.argmax(y_labels['fine'][random_number[i]])),
                    str(np.argmax(predictions[2][random_number[i]]))))
            
def multi_gpu_select(os_system : str = 'windows'):
    if os_system == 'windows':
        #Multiple GPU ------ Windows
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    elif os_system == 'linux':
        #Multiple GPU ------ LINUX
        strategy = tf.distribute.MirroredStrategy()
    return strategy
