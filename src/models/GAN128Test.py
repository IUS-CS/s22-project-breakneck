# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 03:20:57 2022
"""

from matplotlib import pyplot as plt
from GAN128 import GAN128Model
from utils.ImageLoader import ImageLoader
import numpy as np

imgLoader = ImageLoader('../data/dataset/')
gan = GAN128Model()

#get input images
gen= imgLoader.getGenerator(2, imageSize = 128)
inputXimg = next(gen)

#random generate parameters
parameterVec = np.random.normal(0, 1, (2, 400))

#get image generator
generator = gan.getModel().layers[2]


#get output and draw
out = generator.predict([inputXimg, parameterVec])

for i, j in zip(out, inputXimg):
    plt.imshow(i)
    plt.show()
    plt.imshow(j)
    plt.show()



