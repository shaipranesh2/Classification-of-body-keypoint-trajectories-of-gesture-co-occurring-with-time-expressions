import os
import torch
from torch import nn
import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import skeletalData
import numpy as np
import random
from model import *
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

class_types=[]
data=[]
label={}
index_label={}
f='/home/sxp1428/sxp1428/numpy_arr'
count=0
distribution = {}
total=0

def oversample(data_sampled, tmp_data_sampled):
    while(len(tmp_data_sampled)<136):
        sampled=random.choice(tmp_data_sampled)
        tmp_data_sampled.append(sampled)
        data_sampled.append(sampled)
    return data_sampled
        

for file1 in os.listdir(f):
    if(file1!='thank_god'):
        index_label[file1] = count
        count+=1
        class_types.append(file1)
    i=0
    tmp_data=[]
    for file2 in os.listdir(f+'/'+file1):
        i+=1
        data.append(f+'/'+file1+'/'+file2)
        label[f+'/'+file1+'/'+file2]=file1
        tmp_data.append(f+'/'+file1+'/'+file2)
    if(i!=0):
        data=oversample(data, tmp_data)
        total+=136
        distribution[file1]=136

print(distribution)
#print(len(distribution))
#print(data)
weights=[]
weight_per_class=[]
for label_class in distribution.keys():
    tmp=[distribution[label_class]/total]*distribution[label_class]
    weights.extend(tmp)
    weight_per_class.append(distribution[label_class]/total)

weight_per_class_tensor = torch.tensor(weight_per_class)
weighted_tensor = torch.tensor(weights)

complete_dataset=skeletalData(data,label)
one_hot_vector=torch.eye(len(index_label))

train_size = int(0.7 * len(complete_dataset))  # 70% for training
test_size = len(complete_dataset) - train_size  # Remaining 30% for testing

train_dataset, test_dataset = torch.utils.data.random_split(complete_dataset, [train_size, test_size])


# Create data loaders for the training and testing sets
train_loader = DataLoader(train_dataset ,batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

"""
for batch in train_loader:
    tensors, labels = batch
    # Do something with the tensors and labels
    print("Tensors:", tensors.size())
    print("Labels:", labels[0])
"""
#print(index_label)
model = Model(3, 41, {'layout':'openpose_25', 'strategy':'spatial'}, True)

pretrained_dict = torch.load('/home/sxp1428/sxp1428/weights/29.pt')

model.load_state_dict(pretrained_dict)

"""
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
"""

optimizer = torch.optim.Adam(model.parameters(), lr = 5e-3)
loss_fn = nn.CrossEntropyLoss()
epochs = 100

best_acc = - np.inf   # init to negative infinity
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

#testing
y_pred=[]
y_actual=[]
epoch_loss=[]
epoch_acc=[]
print_los=0.0
print_acc=0.0
with tqdm.trange(len(test_dataset), unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch 1")
        for batch in test_loader:
            count+=1
            inputs, actual_label = batch
            output = model.forward(inputs)
            #print(inputs.size())
            #print(one_hot_vector[index_label[actual_label[0]]].unsqueeze(0).size())
            loss= loss_fn(output, one_hot_vector[index_label[actual_label[0]]].unsqueeze(0))
            y_pred.append(class_types[torch.argmax(output)])
            y_actual.append(actual_label[0])
            #print(output)
            #print(one_hot_vector[index_label[actual_label[0]]].unsqueeze(0))
            acc = (torch.argmax(output, 1) == torch.argmax(one_hot_vector[index_label[actual_label[0]]].unsqueeze(0), 1)).float().mean()
            print_los+=float(loss)
            print_acc+=float(acc)
            bar.update(1)
            bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
            bar.refresh()

cf_matrix = confusion_matrix(y_actual, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in class_types],
                     columns = [i for i in class_types])
plt.figure(figsize = (25,25))
sn.heatmap(df_cm, annot=True)
plt.savefig('accuracy_matrix.png')

print(print_los/len(test_dataset))
print(print_acc/len(test_dataset))

"""
#training
for epoch in range(epochs):
    epoch_loss = []
    epoch_acc = []
    print_los=0.0
    print_acc=0.0
    count=0
    with tqdm.trange(len(train_dataset), unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for batch in train_loader:
            count+=1
            inputs, actual_label = batch
            output = model.forward(inputs)
            #print(inputs.size())
            optimizer.zero_grad()
            #print(one_hot_vector[index_label[actual_label[0]]].unsqueeze(0).size())
            loss= loss_fn(output, one_hot_vector[index_label[actual_label[0]]].unsqueeze(0))
            #print(output)
            #print(one_hot_vector[index_label[actual_label[0]]].unsqueeze(0))
            loss.backward()
            optimizer.step()
            acc = (torch.argmax(output, 1) == torch.argmax(one_hot_vector[index_label[actual_label[0]]].unsqueeze(0), 1)).float().mean()
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))
            print_los+=loss
            print_acc+=acc
            bar.update(1)
            bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
            bar.refresh()
    print(float(print_acc)/3903)
    print(float(print_los)/3903)
    torch.save(model.state_dict(), './weights_new/'+str(epoch)+'.pt')
"""