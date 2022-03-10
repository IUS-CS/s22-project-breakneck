# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 19:50:36 2022
"""

from matplotlib import pyplot as plt
from AutoEncoder import AutoEncoderModel
from utils.ImageLoader import ImageLoader

imgLoader = ImageLoader('../data/dataset/')
#load saved model
ae = AutoEncoderModel()

gen = imgLoader.getGenerator(10)
inputXimg = next(gen)

#get predicted image see how close the original images are to the generated images
outputImg = ae.predict(inputXimg)

#draw images
for i, j in zip(outputImg, inputXimg):
    plt.imshow(i)
    plt.show()
    plt.imshow(j)
    plt.show()
