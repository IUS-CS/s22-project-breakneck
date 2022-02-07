# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 13:15:32 2022

This file check all necessary requirements that are needed to build and run the program
"""

import pkg_resources
import traceback
import sys




def check():
    """
    this check function will be called directly by build.py
    return 0 if everything is satisfied
    return 1 if python version incorrect
    return 2 if missing libraries
    return 3 unkonwn errors
    """
    
    #inner function that converts version from string type to tuple type
    def version2tuple(v : str):
        arr = []
        for i in v.replace('\n', '').replace('\r', '').split('.'):
            if i.isdigit():
                arr.append(int(i))
        
        return tuple(arr)
   
    
    #check requirements
    try:
        
        #load library requirements file
        req_list = []
        installed_dict = {}
        read = open('../lib_requirements.txt', 'r')
        write = open('./missing_libs.txt', 'w')
        
        for line in read.readlines():
            #load all required libraries to req_list
            sp = line.split(' ')
            req_list.append((sp[0], version2tuple(sp[1])))
            
        read.close()
        
        
        #python version check
        if sys.version_info < (3, 5):
            print('Your python version is outdated, 3.5 or higher version is required')
            return 1
        print('Your python version:', sys.version_info)
        
        #check libraries
        missing_lib = False
        for lib in pkg_resources.working_set:
            #to check libraries effectively
            #load all installed libraries to installed_dict dictionary as the format (name : str, version : tuple)
            installed_dict[lib.project_name] = version2tuple(lib.version)
            print(lib.project_name, lib.version)
        
        for l, v in req_list:
            try:
                version = installed_dict[l]
                if v > version :
                    #need to update
                    write.write('pip install ' + l + ' --upgrade\n')
                    print('your', l, 'is outdated, version:', v , 'is required')
                else:
                    print('package:', l, 'version:', version, '..OK')
            except:
                missing_lib = True
                #missing library
                write.write('pip install ' + l + '\n')
                print('missing package:', l)
        
        write.close()
        if missing_lib:
            return 2
        
        return 0
    except:
        print(traceback.format_exc())
        return 3
    