# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 17:48:10 2022
"""

from AutoEncoder import auto_encoder_train
from GAN32 import gan32_train
from tensorflow.keras import backend as K

K.clear_session()

#train auto encoder
#auto_encoder_train(20, 3000)

#train GAN32
gan32_train(20, 3000)