import os
import torch
from torch import nn
import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import skeletalData
import numpy as np
from model import *

data=[]
label={}
index_label={}
f='/home/sxp1428/sxp1428/numpy_arr'
count=0
distribution = {}
total=0

for file1 in os.listdir(f):
    index_label[file1] = count
    count+=1
    i=0
    for file2 in os.listdir(f+'/'+file1):
        i+=1
        total+=1
        data.append(f+'/'+file1+'/'+file2)
        label[f+'/'+file1+'/'+file2]=file1
    distribution[file1]=i

#print(distribution)
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

indices = torch.multinomial(weighted_tensor, len(complete_dataset), replacement=True)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_dataset = torch.utils.data.Subset(complete_dataset, train_indices)
test_dataset = torch.utils.data.Subset(complete_dataset, test_indices)

# Create data loaders for the training and testing sets
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

"""
for batch in train_loader:
    tensors, labels = batch
    # Do something with the tensors and labels
    print("Tensors:", tensors.size())
    print("Labels:", labels[0])
"""
#print(index_label)
model = Model(3, 400, {'layout':'openpose', 'strategy':'spatial'}, True)

pretrained_dict = torch.load('/home/sxp1428/sxp1428/st_gcn.kinetics.pt')

model.load_state_dict(pretrained_dict)
final_layer = nn.Sequential(
                nn.Conv2d(256,42, 1),
                #nn.Flatten(),
                #nn.Linear(128,42),
                nn.Sigmoid()
                )
model.fcn = final_layer

"""
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
"""

optimizer = torch.optim.Adam(model.parameters(), lr = 5e-3)
loss_fn = nn.CrossEntropyLoss(weight=weight_per_class_tensor)
epochs = 100

best_acc = - np.inf   # init to negative infinity
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []


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
            #print(inputs)
            optimizer.zero_grad()
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
    print(float(print_acc)/count)
    print(float(print_los)/count)
    torch.save(model.state_dict(), './weights/'+str(epoch)+'.pt')