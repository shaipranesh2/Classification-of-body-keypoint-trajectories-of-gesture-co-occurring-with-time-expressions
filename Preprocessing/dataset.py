import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class skeletalData(Dataset):
    def __init__(self, data,label):
        self.data=data
        self.label=label
    def __len__(self):
        return len(self.data)
    def __get_item__(self, index):
        array_data_path = self.data[index]
        skeletal_array = np.load(array_data_path)
        class_label = self.labels[array_data_path]
        reutrn skeletal_array, class_label


data=[]
label={}
for file1 in os.listdir('~/keyposes'):
    for file2 in os.listdir('~/keyposes/'+file1):
        for file3 in os.listdir('.~/keyposes/'+file1+'/'+file2):
            if(file3[:-4]=='.npy'):
                data.append('.~/keyposes/'+file1+'/'+file2+'/'+file3)
                label['.~/keyposes/'+file1+'/'+file2+'/'+file3]=file2

complete_dataset=skeletalData(data,label)
print(len(complete_dataset))
