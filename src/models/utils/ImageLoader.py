# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 22:43:46 2022


"""

import os
from PIL import Image
from PIL import ImageEnhance
import numpy as np
import random
import threading

class ImageLoader:
    '''
        This image loader returns a generator that always load a portion of images from a image directory
    '''

    __img_dir = None
    __img_list =[]
    __imageCounter = 0
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
        
        
        
        #shuffle
        random.seed()
        random.shuffle(self.__img_list)
    
    def __loadPreloadBatch(self, buffer, batchSize, batchPreloadSize):
        '''
        load (batchPreloadSize) batches of images into buffer

        Parameters
        ----------
        buffer : TYPE
            target buffer

        batchPreloadSize : int
            preload how many batches

        Returns
        -------
        None.

        '''
        for i in range(batchSize * batchPreloadSize):
            #load img
            PILImage = Image.open(self.__img_dir + '/' + self.__img_list[self.__imageCounter])
            
            #random rotation 
            PILImage = PILImage.rotate(-15 + random.randint(0, 39))
            
            #random crop
            width, height = PILImage.size
            PILImage = PILImage.crop((random.randint(0, 15), random.randint(0, 15), width - random.randint(0, 15), height - random.randint(0, 15)))
            
            #random color tweak
            enhance_range = 0.3
            PILImage = ImageEnhance.Brightness(PILImage).enhance(1 + np.random.uniform(0, enhance_range) - (enhance_range/2))
            PILImage = ImageEnhance.Color(PILImage).enhance(1 + np.random.uniform(0, enhance_range) - (enhance_range/2))
            PILImage = ImageEnhance.Contrast(PILImage).enhance(1 + np.random.uniform(0, enhance_range) - (enhance_range/2))
            
            #convert to numpy array
            img = np.asarray(PILImage.convert('RGB').resize((256, 256)))/255
            buffer[i, :] = img

            self.__imageCounter += 1
            # if counter > batchSize, repeat again
            self.__imageCounter %= len(self.__img_list)
        
        
    
    def getGenerator(self, batchSize: int, batchPreloadSize = 4, batchReuseSize = 2):
        '''
        returns a generator that will load (batchSize) images everytime
        this generator will load images sequentially, and repeat after
        the last image is loaded

        Parameters
        ----------
        batchSize : int
            load how many images

        batchPreloadSize : int
            preload how many batch of images into memory
            
        batchReuseSize
            How many times the generator will reuse images from preloaded data before loading new data
        
        Returns
        -------
        (batchSize, 256, 256, 3) numpy array

        '''
        
        #batch size check
        if batchSize < 1:
            raise RuntimeError('batch size must larger than 0')
        
        #allocate space for images
        X = np.zeros((batchSize, 256, 256, 3), np.float32)
        
        #avoid negative numbers
        if batchPreloadSize < 1:
            batchPreloadSize = 1
        
        if batchReuseSize < 1:
            batchReuseSize = 1
        
        #preload buffer
        buffer = [np.zeros((batchSize * batchPreloadSize, 256, 256, 3), np.float32()), np.zeros((batchSize * batchPreloadSize, 256, 256, 3), np.float32())]
        currentBuffer = 0
        task = None
        
        
        #load first buffer
        self.__loadPreloadBatch(buffer[currentBuffer], batchSize, batchPreloadSize)
        
        #never ends, always load more images
        while True:
            
            #load next buffer in different thread
            task = threading.Thread(target = self.__loadPreloadBatch, args = (buffer[not currentBuffer], batchSize, batchPreloadSize))
            task.start()
            
            #use image data from buffer
            for j in range(batchReuseSize):
                for i in range(batchPreloadSize):
                    
                    X[:] = buffer[currentBuffer][i * batchSize : (i + 1) * batchSize]
                    
                    #return generator
                    yield X
            
            #swap buffer
            currentBuffer = not currentBuffer
            
            #wait until loading thread finish it's work
            task.join()
            

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