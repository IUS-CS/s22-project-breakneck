# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 22:43:46 2022


"""

import os
from PIL import Image
import numpy as np

class ImageLoader:
    '''
        This image loader returns a generator that always load a portion of images from a image directory
    '''

    __img_dir = None
    __img_list =[]
    
    def __init__(self, img_dir : str):
        '''
        init image generator

        Parameters
        ----------
        img_dir : str
            image dataset directory

        Returns
        -------
        None.

        '''
        self.__img_dir = img_dir
        
        #read filenames to list
        self.__img_list = os.listdir(img_dir)
        
    def getGenerator(self, batchSize: int):
        '''
        returns a generator that will load (batchSize) images everytime
        this generator will load images sequentially, and repeat after
        the last image is loaded

        Parameters
        ----------
        batchSize : int
            load how many images

        Returns
        -------
        (batchSize, 256, 256, 3) numpy array

        '''
        
        #batch size check
        if batchSize < 1:
            raise RuntimeError('batch size must larger than 0')
        
        counter = 0
        #allocate space for images
        X = np.zeros((batchSize, 256, 256, 3), np.float32)
        
        #never ends, always load more images
        while True:
            
            for i in range(batchSize):
                
                #load img
                img = np.asarray(Image.open(self.__img_dir + '/' + self.__img_list[counter]).convert('RGB').resize((256, 256)))/255
                X[i, :] = img
   
                counter+=1
                # if counter > batchSize, repeat again
                counter %= len(self.__img_list)
            
            #return generator
            yield X

def xToXY(gen):
    '''
    This function duplicates the return data from a generator
    and warps it into a tuple

    Parameters
    ----------
    gen : TYPE
        generator

    Returns
    -------
    a generator that returns a tuple

    '''
    try:
        while True:
            X = next(gen)
            yield (X, X)
            
    except:
        pass