from fileinput import filename
import os
import sys
import glob
from google.cloud import storage
from pathlib import Path

### Script for uploading and downloading models from google cloud storage

BUCKET_NAME='project_fake_image_bucket'

storage_client = storage.Client.create_anonymous_client()
bucket = storage_client.bucket(BUCKET_NAME)

# blob: binary large object, a collection of binary data stored as one... blob
def upload_to_bucket(blob_name: str, file_path: str):
    try:
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
        return True
    except Exception as e:
        print(e)
        return False
        
def download_files_from_bucket(blob_name: str, file_path: str):
    try:
       blob = bucket.blob(blob_name)
       with open(file_path, 'wb') as f:
           storage_client.download_blob_to_file(blob, f)
       return True
    except Exception as e:
       print(e)
       return False
    
    # Returns a list of files (their full paths) in gcs 
def list_files_in_gcs():
    blob_list = bucket.list_blobs()
    files = []
 
    for blob in blob_list:
        str_blob = str(blob)
        str_blob = str_blob.split(",")
        filepath = str_blob[1].replace(" ", "")
        
        files.append(filepath)
        
    return files

if __name__ == '__main__':
        
    path = os.getcwd()
    pardir = os.path.join(path, os.pardir)
    src_dir = os.path.abspath(pardir)
    print(src_dir)
    args =  sys.argv
    blobs = []
    
    run = True
    while(run):       
        print(  "1: download all models (will be saved in models/saved_models\n"
                "2: upload a model to GCS\n"
                "l: list files in GCS\n"
                )
        
        choice = input()
        if (choice == '1'):
            files= list_files_in_gcs()
            path = src_dir+"/saved_models/"
            
            i = 0
            while i< len(files):
                download_files_from_bucket(files[i], path+files[i])
                i+=1
            
        elif (choice == '2'):
            print("\n Make sure the model is stored in the saved_models directory, Input the name of the model you want to upload \n")
            file = input()
            path = src_dir+"/saved_models/"+file
            print(path)
            upload_to_bucket(file, path)
            print("Be patient, depending on the size of the model upload could take a while")
            
        elif (choice == 'l'):
            list = list_files_in_gcs()
            print(list)
            
        print("Are you done? y/n")
        if(input() == 'y'):
            run = False
            