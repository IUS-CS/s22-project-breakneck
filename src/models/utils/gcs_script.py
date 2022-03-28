from fileinput import filename
import os
import sys
import glob
from google.cloud import storage
from pathlib import Path

### Script for accessing google cloud storage currently only provides the ability to upload files  

BUCKET_NAME='project_fake_image_bucket'

# storage module looks at this environ path and locates service account file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'Service_Key_GCS.json'
storage_client = storage.Client()

# blob: binary large object, a collection of binary data stored as one... blob
def upload_to_bucket(blob_name: str, file_path: str):
    try:
        bucket = storage_client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
        return True
    except Exception as e:
        print(e)
        return False
        
def download_files_from_bucket(blob_name: str, file_path: str):
    try:
       bucket = storage_client.bucket(BUCKET_NAME)
       blob = bucket.blob(blob_name)
       with open(file_path, 'wb') as f:
           storage_client.download_blob_to_file(blob, f)
       return True
    except Exception as e:
       print(e)
       return False
    
    # Return a list of files (their full paths) in gcs 
def list_files_in_gcs():
    bucket = storage_client.get_bucket(BUCKET_NAME)
    blob_list = bucket.list_blobs()
    files = []
 
    for blob in blob_list:
        str_blob = str(blob)
        str_blob = str_blob.split(",")
        filepath = str_blob[1].replace(" ", "")
        
        files.append(filepath)
        
    return files

# downloads all files in bucket to specified address
def download_all(files_list: list, filepath: str):
    bucket = storage_client.get_bucket(BUCKET_NAME)
    for file in files_list:
        gcs_path = file
        filename = file.split("/")
        filename = file[1]
        download_files_from_bucket(gcs_path, filepath+"/"+filename)
        
def upload_dir(blob_name: str, dir_path: str):
    bucket = storage_client.get_bucket(BUCKET_NAME)
    files = glob.glob(dir_path+"/*")
    for file in files:
        file=file.split(",")
        #upload_to_bucket(blaob_name, dir_path+file)
        


 # want to give them the ability to download all files from the model folder to a specific directory 
 # want to give them th eabiltiy to upload files to the model folder 
if __name__ == '__main__':
        
    path = os.getcwd()
    pardir = os.path.join(path, os.pardir)
    src_dir = os.path.abspath(pardir)
    print(src_dir)
    args =  sys.argv
    blobs = []
    
    run = True
    while(run):       
        print("1: download all files in GCS model folder\n"
                "2: upload a file to GCS model folder\n"
                "3: delete a file from GCS model folder\n"
                "l: list files in GCS\n"
                )
        choice = input()
        if (choice == '1'):
            files=input()
            print(files)
            print("Input the path where you would like to store the files, assume you're already in the src folder\n"
                "Example: models/saved_models")
            path = input()
            download_files_from_bucket(files, src_dir+"/"+path)
            
        elif (choice == '2'):
            print("Input name of the file you'll be uploading")
            file_name= input()
            print("Input the path to acquire this file from, assume you're already in the src folder\n"
                "Example: models/saved_models")
            path = input()
            upload_to_bucket("models/"+file_name,src_dir+"/"+path+"/"+file_name)
            
        elif (choice == 'l'):
            list = list_files_in_gcs()
            print(list)
            
        print("Are you done? y/n")
        if(input() == 'y'):
            run = False
            
    
    