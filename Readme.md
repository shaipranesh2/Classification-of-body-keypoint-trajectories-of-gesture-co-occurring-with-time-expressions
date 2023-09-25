This project aims to develop a Neural network architecture to classify gestures with body-keypose data extracted using STGCN on ellen dataset
(This code was developed as part of Google Summer of Code'23. Read more on: https://medium.com/@f20200731/google-summer-of-code-23-red-hen-labs-6bc683056ebf)

Usage:
1. ellen_npy.py is used to extract the openpose keypose points from the json format, and it also extracts the labels and creates the data using
   the custom dataset class.
   It takes in 2 arguments:- npz file data path and label path order, respectively in that order strictly.
   For ex:
     python ellen_npy.py /mnt/rds/redhen/gallina/home/ixb164/gsoc2023_ellen2014_npzs /mnt/rds/redhen/gallina/home/ixb164/labels

2. Run ellen_model.py as a python script and the model trains and validates for every epoch
   For ex:
     python ellem_model.py

The model's performance in losses and accuracy graphs are also uploaded, along with some pretrained weights and the dataset as a .pth file.

