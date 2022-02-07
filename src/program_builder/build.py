# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 13:32:10 2022

This file build&run the project
"""

import environment_check as env
import os
import subprocess
PROJECT_NAME = 'example.py'


os.chdir('program_builder')
#check environment
st_code = env.check()

#if st_code is 3(unkown error) or 1(python version outdated)
if st_code == 1 or st_code == 3:
    print('build failed...')
    exit()

#download missing packages
print('downloading missing packages...')
for l in open('missing_libs.txt'):
    command = l.replace('\r', '').replace('\n', '')
    ret_code = 0
  
    result = subprocess.check_output(command.split(' '), ret_code)
    if ret_code !=0:
        print('error, return code:', ret_code)
    
#run program
os.chdir('../')
print('launching program...')
os.system('python ' + PROJECT_NAME)

