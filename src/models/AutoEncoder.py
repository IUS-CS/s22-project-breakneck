# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 00:29:09 2022
"""

from Wrapper import ModelWrapper
from utils.ImageLoader import ImageLoader
from utils.ImageLoader import xToXY
from tensorflow import keras
import matplotlib.pyplot as plt

class AutoEncoderModel(ModelWrapper):
    
    def __resnetBlock(self, size : int, in_depth : int, out_depth : int):
        #single resnet block, for details see https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
        
        inX = keras.layers.Input(shape = (size, size, in_depth))
        
        X = keras.layers.Conv2D(out_depth, (3, 3), padding = 'same')(inX)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.Activation('relu')(X)
        X = keras.layers.Conv2D(out_depth, (2, 2), padding = 'same')(X)
        X = keras.layers.BatchNormalization()(X)
        
        X1 = keras.layers.Conv2D(out_depth, (1, 1))(inX)
        Y = keras.layers.Add()([X, X1])
        Y = keras.layers.Activation('relu')(Y)
        
        return keras.Model(inX, Y)
        
    def __resnetBlockInverse(self, size : int, in_depth : int, out_depth : int):
         #single inverse resnet block
        inX = keras.layers.Input(shape = (size, size, in_depth))
        
        X = keras.layers.Conv2D(out_depth, (3, 3), padding = 'same')(inX)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.Activation('relu')(X)
        X = keras.layers.Conv2D(out_depth, (2, 2), padding = 'same')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.Activation('relu')(X)
        
        #up scale
        X = keras.layers.Conv2DTranspose(out_depth, (2, 2), strides = (2, 2))(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.Activation('relu')(X)
        
        X1 = keras.layers.Conv2D(out_depth, (1, 1))(inX)
        X1 = keras.layers.UpSampling2D()(X1)
        Y = keras.layers.Add()([X, X1])
        Y = keras.layers.Activation('relu')(Y)
        
        return keras.Model(inX, Y)
    
    def __encoder_model(self):
        
        #take (256, 256, 3) as input
        inX = keras.layers.Input(shape = (256, 256, 3))
        X = keras.layers.BatchNormalization()(inX)
        
        #output 128 * 128 * 24
        X = self.__resnetBlock(256, 3, 24)(X)
        X = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(X)
        
        #output 64 * 64 * 48
        X = self.__resnetBlock(128, 24, 48)(X)
        X = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(X)
        
        #output 32 * 32 * 96
        X = self.__resnetBlock(64, 48, 96)(X)
        X = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(X)
        
        #output 16 * 16 * 192
        X = self.__resnetBlock(32, 96, 192)(X)
        X = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(X)
        
        #output 8 * 8 * 250
        X = self.__resnetBlock(16, 192, 250)(X)
        X = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(X)
        
        #output 4 * 4 * 300
        X = self.__resnetBlock(8, 250, 300)(X)
        X = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(X)
        
        #output 2 * 2 * 350
        X = self.__resnetBlock(4, 300, 350)(X)
        X = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(X)
        
        #output 1 * 1 * 400
        X = self.__resnetBlock(2, 350, 400)(X)
        X = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(X)
        
        #output 400
        X = keras.layers.Flatten()(X)
        X = keras.layers.Dense(400)(X)
        X = keras.layers.Activation('tanh')(X)
        
        return keras.Model(inX, X)
        
    def __decoder_model(self):
        
        inX = keras.layers.Input(shape = (400,))
        
        #output 1 * 1 * 400
        X = keras.layers.Dense(400)(inX)
        X = keras.layers.Activation('sigmoid')(X)
        X = keras.layers.Reshape((1, 1, 400))(X)
        
        #output 2 * 2 * 350
        X = self.__resnetBlockInverse(1, 400, 350)(X)
        
        #output 4 * 4 * 300
        X = self.__resnetBlockInverse(2, 350, 300)(X)
        
        #output 8 * 8 * 250
        X = self.__resnetBlockInverse(4, 300, 250)(X)
        
        #output 16 * 16 * 192
        X = self.__resnetBlockInverse(8, 250, 192)(X)
        
        #output 32 * 32 * 96
        X = self.__resnetBlockInverse(16, 192, 96)(X)
        
        #output 64 * 64 * 48
        X = self.__resnetBlockInverse(32, 96, 48)(X)
        
        #output 128 * 128 * 24
        X = self.__resnetBlockInverse(64, 48, 24)(X)
        
        #output 256 * 256 * 16
        X = self.__resnetBlockInverse(128, 24, 16)(X)
        
        #output 256 * 256 * 3
        X = keras.layers.Conv2D(3, (1, 1))(X)
        X = keras.layers.Activation('sigmoid')(X)
        
        return keras.Model(inX, X)
        
    
    def _define_model(self):
        encoder = self.__encoder_model()
        decoder = self.__decoder_model()
        
        X = keras.Input(shape = (256, 256, 3))
        Y = encoder(X)
        Y = decoder(Y)
        
        return keras.Model(X, Y), 'AutoEncoder'
        
        
    def _compile_model(self, model):
        model.compile(optimizer = keras.optimizers.SGD(0.0001, 0.9), loss = 'mse')


def auto_encoder_train(batchSize : int, epoch : int):
    ae = AutoEncoderModel()
    imgLoader = ImageLoader('../data/dataset/')
    imageGenerator = xToXY(imgLoader.getGenerator(batchSize))
    
    ae.train(x = imageGenerator, steps_per_epoch = 200, batch_size = batchSize, epochs = epoch)
    ae.save()
            