"""
Using RESNET-50 pretrained model as the basis for our discriminator model;

The job of this model is simple: binary classification on the pictures given to it by the
GAN model...

The discriminator cannot be trained before being used in conjunction to the GAN model
this creates a couple questions, how can I determine that it works correctly without training it?

We need just the foundation, no way to truly train it without using the GAN model


input: image
output: 0 or 1

First step? Accept input, what's the best way to give input to the discriminator?

"""
import numpy as np
import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Flatten
import matplotlib.image as img



# adjusting the image to the format we need
# 
img_file = np.load("/mnt/c/Users/edwar/Pictures/darkness.png", allow_pickle=True)#img.imread("/mnt/c/Users/edwar/Pictures")
image = img_file # converting,
print(image) 