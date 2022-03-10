# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 03:20:57 2022
"""

from matplotlib import pyplot as plt
from GAN64 import GAN64Model
from utils.ImageLoader import ImageLoader
import numpy as np

imgLoader = ImageLoader('../data/dataset/')
gan = GAN64Model()

#get input images
gen= imgLoader.getGenerator(10, imageSize = 64)
inputXimg = next(gen)

#random generate parameters
parameterVec = np.random.normal(0, 1, (10, 400))

#get image generator
generator = gan.getModel().layers[2]


#get output and draw
out = generator.predict([inputXimg, parameterVec])

for i, j in zip(out, inputXimg):
    plt.imshow(i)
    plt.show()
    plt.imshow(j)
    plt.show()



