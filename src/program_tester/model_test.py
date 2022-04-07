
import os
os.chdir('../models/')

from matplotlib import pyplot as plt
from GAN32 import GAN32Model
from GAN64 import GAN64Model
from GAN128 import GAN128Model
from AutoEncoder import AutoEncoderModel
from utils.ImageLoader import ImageLoader
from tensorflow import keras
import numpy as np
from keras.applications.vgg16 import VGG16
from tensorflow import keras

# load prediction model
model = VGG16()

imgLoader = ImageLoader('../data/dataset/')
gen = imgLoader.getGenerator(2)

#models
ae =  AutoEncoderModel()
encoder = ae.getModel().layers[1]
decoder = ae.getModel().layers[2]
gan32 = GAN32Model().getModel().layers[2]
gan64 = GAN64Model().getModel().layers[2]
gan128 = GAN128Model().getModel().layers[2]

def genImg(img, paraVector):
    output = keras.layers.Resizing(32, 32)(img)
    output = gan32.predict([output, paraVector])
    output = gan64.predict([output, paraVector])
    output = gan128.predict([output, paraVector])
    
    return output

def getScore(img):
    res = model.predict(keras.layers.Resizing(224, 224)(img))
    
    #gini impurity
    res = res**2
    return res.sum()
    

#input image
for i in range(5):
    inputImage = next(gen)

    paraVector = np.random.normal(0, 1, (2, 400))
    output = genImg(inputImage, paraVector)
    
    #show score
    print("score:", getScore(output))
    

    for i, j in zip(output, inputImage):
        plt.imshow(i)
        plt.show()
        plt.imshow(j)
        plt.show()

