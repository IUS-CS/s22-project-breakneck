# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 13:32:10 2022

This file build&run the project
"""

import os

PROJECT_NAME = 'program.py'
os.system("pip install -r lib_requirements.txt")
#run program
print(os.getcwd())
os.chdir('models/')
print('launching program...')
os.system('python ' + PROJECT_NAME)
