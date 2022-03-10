# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 00:29:39 2022

@author: otz55
"""

import os
from tensorflow import keras
from tensorflow.keras.callbacks import LambdaCallback

class ModelWrapper:
    
    _model = None
    __name = None
    __fileDir = os.path.dirname(__file__)
    def __init__(self):
        
        #load init model
        self.reset()
        
        #auto load weights
        self.load()
    
    def _define_model(self):
        '''
        This function must be implemented in the child class
        Please override and build the keras model in this function.
        
        Raises
        ------
        NotImplementedError
            raised when this function is not override by child class

        Returns
        -------
        returns a tuple where the first element is the keras model
        and the second element is the model name
        (model, name)

        '''
        raise NotImplementedError('Please implement this function')
    
    def _compile_model(self, model):
        '''
        This function must be implemented in the child class
        Please override and compile the keras model here.
        
        Parameters
        ----------
        model : TYPE
            keras model

        Raises
        ------
        NotImplementedError
            raised when this function is not override by child class

        Returns
        -------
        None.

        '''
        raise NotImplementedError('Please implement this function')
    
    
    def train(self, autoSave = 5, **kwargs):
        '''
        wrap the model.fit() function with auto save mechanism added

        Parameters
        ----------
        autoSave : TYPE, optional
            auto save model every X epoch
            disable auto save if X < 1
            this default value is 5
        **kwargs : TYPE
            parameters for model.fit()

        Returns
        -------
        None.

        '''
        #auto save callback
        def auto_save(epoch, logs):
            
            #save model
            if epoch % autoSave == 0:
                self.save()
                print(self.__name ,'model auto saved')
        
        #add auto save callback while training
        if autoSave >= 1:
            kwargs['callbacks'] = [LambdaCallback(on_epoch_end = auto_save)]
        
        #train the model
        self._model.fit(**kwargs)
        
    def predict(self, X):
        '''
        wrap the model.predict()

        Parameters
        ----------
        X : TYPE
            input X

        Returns
        -------
        model predict result

        '''
        return self._model.predict(X)
    
    def save(self):
        '''
        save model weights

        Returns
        -------
        None.

        '''
        try:
            self._model.save_weights(self.__fileDir + '/saved_models/' + self.__name + '.h5')
        except:
            print('unable to save the model', self.__name)
    
    def load(self):
        '''
        load saved weights

        Returns
        -------
        None.

        '''
        try:
            self._model.load_weights(self.__fileDir + '/saved_models/' + self.__name + '.h5')
        except:
            print('unable to load the model', self.__name)
        
    
    def reset(self):
        '''
        reset all weights

        Returns
        -------
        None.

        '''
        #get defined model
        self._model, self.__name = self._define_model()
        
        #compile model
        self._compile_model(self._model)

    def getModel(self):
        return self._model
