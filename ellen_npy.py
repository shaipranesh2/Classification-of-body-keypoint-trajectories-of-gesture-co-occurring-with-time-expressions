import numpy as np
import sys
import os
from ellen_dataset import skeletalData
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split

fps=29.97
frames_taken = 75

keypose_path = sys.argv[1]
labels_path = sys.argv[2]
frame = None
sample_number = 400
test_sample = 300
frame=0
frame_in_sec = 0

data = []
label_data = []

def is_value_in_range(row):
    return float(row['Begin Time - msec']) <= frame_in_sec*1000 <= float(row['End Time - msec'])

keypose_files = sorted(os.listdir(keypose_path))
test = keypose_files[int(len(keypose_files)*0.8):]
keypose_files = keypose_files[:int(len(keypose_files)*0.8)]

for i in keypose_files:
    dataset = np.load(keypose_path+'/'+i)["arr_0"]
    label=None
    df = pd.read_csv(labels_path+'/'+i[:-4]+'.csv')
    for _ in range(sample_number):
        frame = random.randint(0,np.shape(dataset)[0]-1-frames_taken)
        frame_in_sec = (frame+frames_taken)/fps
        if(df.apply(is_value_in_range, axis=1).any()):
            label=1
        else:
            label=0
        label_data.append(label)
        data.append(dataset[frame:frame+frames_taken,:,:,:])
        #if(len(data)==16):
        #    break
    #break

train = skeletalData(data, label_data)

torch.save(train, './train_dataloader_ellen.pth')
data = []
label_data = []

for i in test:
    dataset = np.load(keypose_path+'/'+i)["arr_0"]
    label=None
    df = pd.read_csv(labels_path+'/'+i[:-4]+'.csv')
    for _ in range(test_sample):
        frame = random.randint(0,np.shape(dataset)[0]-1-frames_taken)
        frame_in_sec = (frame+frames_taken)/fps
        if(df.apply(is_value_in_range, axis=1).any()):
            label=float(1.0)
        else:
            label=float(0.0)
        label_data.append(label)
        data.append(dataset[frame:frame+frames_taken,:,:,:])

test = skeletalData(data, label_data)

torch.save(test, './test_dataloader_ellen.pth')
