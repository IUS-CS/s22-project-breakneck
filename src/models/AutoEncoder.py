# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 00:29:09 2022
"""

from Wrapper import ModelWrapper
from utils.ImageLoader import ImageLoader
from utils.ImageLoader import xToXY
from tensorflow import keras

class AutoEncoderModel(ModelWrapper):
    
    def __resnetBlock(self, size : int, in_depth : int, out_depth : int):
        #single resnet block, for details see https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
        
        inX = keras.layers.Input(shape = (size, size, in_depth))
        
        X = keras.layers.Conv2D(out_depth, (3, 3), padding = 'same')(inX)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.Activation('relu')(X)
        X = keras.layers.Conv2D(out_depth, (3, 3), padding = 'same')(X)
        X = keras.layers.BatchNormalization()(X)
        
        X1 = keras.layers.Conv2D(out_depth, (1, 1))(inX)
        Y = keras.layers.Add()([X, X1])
        Y = keras.layers.Activation('relu')(Y)
        
        return keras.Model(inX, Y)
        
        
    def __encoder_model(self):
        
        #take (256, 256, 3) as input
        inX = keras.layers.Input(shape = (256, 256, 3))
        X = keras.layers.BatchNormalization()(inX)
        
        #output 128 * 128 * 6
        X = self.__resnetBlock(256, 3, 6)(X)
        X = self.__resnetBlock(256, 6, 6)(X)
        X = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(X)
        
        #output 64 * 64 * 12
        X = self.__resnetBlock(128, 6, 12)(X)
        X = self.__resnetBlock(128, 12, 12)(X)
        X = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(X)
        
        #output 32 * 32 * 24
        X = self.__resnetBlock(64, 12, 24)(X)
        X = self.__resnetBlock(64, 24, 24)(X)
        X = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(X)
        
        #output 16 * 16 * 48
        X = self.__resnetBlock(32, 24, 48)(X)
        X = self.__resnetBlock(32, 48, 48)(X)
        X = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(X)
        
        #output 8 * 8 * 96
        X = self.__resnetBlock(16, 48, 96)(X)
        X = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(X)
        
        #output 4 * 4 * 192
        X = self.__resnetBlock(8, 96, 192)(X)
        X = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(X)
        
        #output 2 * 2 * 384
        X = self.__resnetBlock(4, 192, 384)(X)
        X = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(X)
        
        #output 1 * 1 * 768
        X = self.__resnetBlock(2, 384, 768)(X)
        X = keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))(X)
        
        #output 256
        X = keras.layers.Flatten()(X)
        X = keras.layers.Dense(256)(X)
        X = keras.layers.Activation('tanh')(X)
        
        return keras.Model(inX, X)
        
    def __decoder_model(self):
        
        inX = keras.layers.Input(shape = (256,))
        
        #output 2 * 2 * 384
        X = keras.layers.Dense(1536)(inX)
        X = keras.layers.Activation('sigmoid')(X)
        X = keras.layers.Reshape((2, 2, 384))(X)
        
        #output 4 * 4 * 192
        X = keras.layers.Conv2DTranspose(384, (2, 2), (2, 2))(X)
        X = keras.layers.Activation('relu')(X)
        X = self.__resnetBlock(4, 384, 192)(X)
        
        #output 8 * 8 * 96
        X = keras.layers.Conv2DTranspose(192, (2, 2), (2, 2))(X)
        X = keras.layers.Activation('relu')(X)
        X = self.__resnetBlock(8, 192, 96)(X)
        
        #output 16 * 16 * 48
        X = keras.layers.Conv2DTranspose(96, (2, 2), (2, 2))(X)
        X = keras.layers.Activation('relu')(X)
        X = self.__resnetBlock(16, 96, 96)(X)
        X = self.__resnetBlock(16, 96, 48)(X)
        
        #output 32 * 32 * 24
        X = keras.layers.Conv2DTranspose(48, (2, 2), (2, 2))(X)
        X = keras.layers.Activation('relu')(X)
        X = self.__resnetBlock(32, 48, 48)(X)
        X = self.__resnetBlock(32, 48, 24)(X)
        
        #output 64 * 64 * 12
        X = keras.layers.Conv2DTranspose(24, (2, 2), (2, 2))(X)
        X = keras.layers.Activation('relu')(X)
        X = self.__resnetBlock(64, 24, 24)(X)
        X = self.__resnetBlock(64, 24, 12)(X)
        
        #output 128 * 128 * 6
        X = keras.layers.Conv2DTranspose(12, (2, 2), (2, 2))(X)
        X = keras.layers.Activation('relu')(X)
        X = self.__resnetBlock(128, 12, 12)(X)
        X = self.__resnetBlock(128, 12, 6)(X)
        
        #output 256 * 256 * 6
        X = keras.layers.Conv2DTranspose(6, (2, 2), (2, 2))(X)
        X = keras.layers.Activation('relu')(X)
        X = self.__resnetBlock(256, 6, 6)(X)
        X = self.__resnetBlock(256, 6, 6)(X)
        
        #output 256 * 256 * 3
        X = keras.layers.Conv2DTranspose(3, (2, 2), padding = 'same')(X)
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
        model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.00001), loss = 'mse')


def auto_encoder_train(batchSize : int, epoch : int):
    ae = AutoEncoderModel()
    imgLoader = ImageLoader('../data/dataset/')
    imageGenerator = xToXY(imgLoader.getGenerator(batchSize))
    
    ae.train(x = imageGenerator, steps_per_epoch = 10, batch_size = batchSize, epochs = epoch)
    ae.save()
            