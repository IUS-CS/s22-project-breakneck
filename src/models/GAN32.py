# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 22:44:32 2022
"""

from Wrapper import ModelWrapper
import numpy as np
from utils.ImageLoader import ImageLoader
from tensorflow import keras
from matplotlib import pyplot as plt

'''
Generator part for GAN32 model
'''
class Generator32(ModelWrapper):
    
    def __block(self, size : int, in_depth : int, out_depth : int, kSize):
        
        inX = keras.layers.Input(shape = (size, size, in_depth))
        Y = keras.layers.Conv2D(out_depth, kernel_size = kSize, padding = 'same')(inX)
        Y = keras.layers.BatchNormalization(momentum = 0.8)(Y)
        Y = keras.layers.ReLU()(Y)
        return keras.Model(inX, Y)
    
    def __blockInverse(self, size : int, in_depth : int, out_depth : int, kSize):
        inX = keras.layers.Input(shape = (size, size, in_depth))
        Y = keras.layers.UpSampling2D(size = (2, 2))(inX)
        Y = keras.layers.Conv2D(out_depth, kernel_size = kSize, padding = 'same')(Y)
        Y = keras.layers.BatchNormalization(momentum = 0.8)(Y)
        Y = keras.layers.ReLU()(Y)
        
        return keras.Model(inX, Y)
    
    def _define_model(self):
        
        # input 32 * 32 image-----------------------
        X0 = keras.layers.Input(shape = (32, 32, 3))
        Y0 = self.__block(32, 3, 512, 3)(X0)
        Y0 = self.__block(32, 512, 512, 3)(Y0)
        #----------------------------------------------
        
        #parameter vector-----------------------------
        X1 = keras.layers.Input(shape = (400,))
        Y1 = keras.layers.Dense(32768)(X1)
        Y1 = keras.layers.Reshape((8, 8, 512))(Y1)
        Y1 = keras.layers.BatchNormalization(momentum = 0.8)(Y1)
        Y1 = keras.layers.ReLU()(Y1)
        #---------------------------------------------
        
        #up-scale to 16 * 16 * 512
        Y1 = self.__blockInverse(8, 512, 512, 2)(Y1)
        Y1 = self.__block(16, 512, 512, 2)(Y1)
        Y1 = self.__block(16, 512, 512, 2)(Y1)
        
        #up-scale to 32 * 32 * 512
        Y1 = self.__blockInverse(16, 512, 512, 3)(Y1)
        Y1 = self.__block(32, 512, 512, 3)(Y1)
        Y1 = self.__block(32, 512, 512, 3)(Y1)
        
        #concat Y0 and Y1, output 32 * 32 * 1024
        Y2 = keras.layers.Concatenate()([Y0, Y1])
        
        #up-scale to 64 * 64 * 700
        Y2 = self.__blockInverse(32, 1024, 700, 3)(Y2)
        Y2 = self.__block(64, 700, 700, 3)(Y2)
        Y2 = self.__block(64, 700, 700, 3)(Y2)
        
        #output 64 * 64 * 3
        Y2 = keras.layers.Conv2D(3, (3, 3), padding = 'same')(Y2)
        Y2 = keras.layers.Activation('sigmoid')(Y2)
        
        return keras.Model([X0, X1], Y2), 'Generator32'
        
        
    def _compile_model(self, model):
        pass

'''
Discriminator part for GAN32 model
'''
class Discriminator32(ModelWrapper):
    
    def __block(self, size : int, in_depth : int, out_depth : int, kSize):
        
        inX = keras.layers.Input(shape = (size, size, in_depth))
        Y = keras.layers.Conv2D(out_depth, kernel_size = kSize, padding = 'same')(inX)
        Y = keras.layers.LeakyReLU(alpha = 0.2)(Y)
        
        return keras.Model(inX, Y)
    
    def _define_model(self):
        
        #take 64 * 64 * 32 image
        X0 = keras.layers.Input(shape = (64, 64, 3))
        Y0 = self.__block(64, 3, 32, 3)(X0)

        
        #output 32 * 32 * 64
        Y0 = self.__block(64, 32, 64, 3)(Y0)      
        Y0 = self.__block(64, 64, 64, 3)(Y0)
        Y0 = self.__block(64, 64, 64, 3)(Y0)
        Y0 = keras.layers.MaxPooling2D((2, 2), strides = (2, 2))(Y0)
        
        #output 16 * 16 * 128
        Y0 = self.__block(32, 64, 128, 3)(Y0)      
        Y0 = self.__block(32, 128, 128, 2)(Y0)
        Y0 = self.__block(32, 128, 128, 2)(Y0)
        Y0 = keras.layers.MaxPooling2D((2, 2), strides = (2, 2))(Y0)
        
        #output 8 * 8 * 256
        Y0 = self.__block(16, 128, 256, 3)(Y0)      
        Y0 = self.__block(16, 256, 256, 2)(Y0)
        Y0 = self.__block(16, 256, 256, 2)(Y0)
        Y0 = keras.layers.MaxPooling2D((2, 2), strides = (2, 2))(Y0)
        
        #output 4 * 4 * 512
        Y0 = self.__block(8, 256, 512, 2)(Y0)      
        Y0 = self.__block(8, 512, 512, 2)(Y0)
        Y0 = self.__block(8, 512, 512, 2)(Y0)
        Y0 = keras.layers.MaxPooling2D((2, 2), strides = (2, 2))(Y0)
        
        #output 1 * 1 * 1024
        Y0 = self.__block(4, 512, 1024, 2)(Y0)      
        Y0 = self.__block(4, 1024, 1024, 2)(Y0)
        Y0 = self.__block(4, 1024, 1024, 2)(Y0)
        Y0 = keras.layers.MaxPooling2D((4, 4), strides = (4, 4))(Y0)
        
        # output 2
        Y0 = keras.layers.Flatten()(Y0)
        Y0 = keras.layers.Dense(2)(Y0)
        Y0 = keras.layers.Activation('softmax')(Y0)
        
        return keras.Model(X0, Y0), 'Discriminator32'
        
        
    def _compile_model(self, model):
        model.compile(optimizer = keras.optimizers.RMSprop(learning_rate = 0.00001), loss = keras.losses.CategoricalCrossentropy())

'''
First layer GAN model that outputs 64x64 image
'''
class GAN32Model(ModelWrapper):
    
    __generator = None
    __generatorWrapper = None
    __discriminator = None
    __discriminatorWrapper = None
    
    def __init__(self):
        
        #load generator and discriminator model first
        self.__generatorWrapper = Generator32()
        self.__discriminatorWrapper = Discriminator32()
        
        self.__generator = self.__generatorWrapper.getModel()
        self.__discriminator = self.__discriminatorWrapper.getModel()
        print('Please ignore "unable to load GAN32 model, GAN model load generator and discriminator separately."')
        super().__init__()
    
    def _define_model(self):
        #construct combined model, this combined model only train the generator, so set discriminator untrainable
        #two inputs for generator one for feature vector, one for feature vector and one for parameter vector
        X0 = keras.layers.Input(shape = (32, 32, 3))
        X1 = keras.layers.Input(shape = (400,))
        
        #pass X0, X1 to generator
        Y0 = self.__generator([X0, X1])
        
        #set discriminator untrainable
        self.__discriminator.trainable = False
        
        #connect the generator to the discriminator
        Y0 = self.__discriminator(Y0)
        
        return keras.Model([X0, X1], Y0), 'GAN32'
        
        
    def _compile_model(self, model):
        model.compile(optimizer = keras.optimizers.RMSprop(learning_rate = 0.00001), loss = keras.losses.CategoricalCrossentropy())
        
    def save(self):
        self.__generatorWrapper.save()
        self.__discriminator.trainable = True
        self.__discriminatorWrapper.save()
        self.__discriminator.trainable = False


    def train(self, x, steps_per_epoch, batch_size, epochs, autoSave = 5, **kwargs):
        
        #losses for generator and discriminator
        disLoss = 0
        genLoss = 0
        
        autoSaveCounter = 0
        
        #true and fake labels
        trueLabel = np.ones((batch_size, 2))
        falseLabel = np.ones((batch_size, 2))
        
        trueLabel[:, 1] = 0
        falseLabel[:, 0] = 0
        
        #simple model to shrink image
        def shrinkModel():
            x = keras.layers.Input(shape = (64, 64, 3))
            y = keras.layers.Resizing(32, 32)(x)
            return keras.Model(x, y)
        
        
        shrink = shrinkModel()
        
        for ep in range(epochs):
            
            
            for step in range(steps_per_epoch):
                
                #parameters for generator
                parameterVector = np.random.normal(0, 1, (batch_size, 400))
                upScaleImg = shrink.predict(next(x))
                
                #generate images
                fakeX = self.__generator.predict(x = [upScaleImg ,parameterVector])
                realX = next(x)
                
                
                #train discriminator
                disLoss = 0.5 * self.__discriminator.train_on_batch(x = fakeX, y = falseLabel)
                disLoss += 0.5 * self.__discriminator.train_on_batch(x = realX, y = trueLabel)
                
                #train generator
                genLoss = self._model.train_on_batch(x = [upScaleImg, parameterVector], y = trueLabel)
                
                print('epoch', ep, 'generator loss:', genLoss, 'discriminator loss:', disLoss)
                
                if step == 0:
                    plt.imshow(upScaleImg[0])
                    plt.show()
                    plt.imshow(fakeX[0])
                    plt.show()
                 
            #autosave
            autoSaveCounter+=1
            
            if autoSaveCounter >= autoSave:
                self.save()
                print('generator and discriminator auto saved')
                autoSaveCounter = 0
        

def gan32_train(batchSize : int, epoch : int):
    gan = GAN32Model()
    imgLoader = ImageLoader('../data/dataset/')
    
    imageGenerator = imgLoader.getGenerator(batchSize, imageSize = 64)
    
    gan.train(x = imageGenerator, steps_per_epoch = 20, batch_size = batchSize, epochs = epoch)
    gan.save()

gan32_train(20, 10)
