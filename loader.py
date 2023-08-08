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
#import seaborn as sn
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import StepLR

class_types=[]
data=[]
label={}
index_label={}
strat_label = []
f='./numpy_arr'
distribution = {}
count=0
total=0
strat_label = []
train_disrtibution={}
test_distribution ={}


"""
def oversample(data_sampled, tmp_data_sampled):
    while(len(tmp_data_sampled)<136):
        sampled=random.choice(tmp_data_sampled)
        tmp_data_sampled.append(sampled)
        data_sampled.append(sampled)
        strat_label.append(sampled)
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
"""


for file1 in os.listdir(f):
    i=0
    if(file1!='thank_god' and file1!='the_day_after_tomorrow' and file1!='the_previous_month'
    and file1!='previous_month' and file1!='the_night_before_last' and file1!='the_following_week' and file1!='the_following_month'
    and file1!='the_previous_week' and file1!='the_previous_day' and file1!='previous_night' and file1!='at_present'
    and file1!='the_following_year' and file1!='the_day_before_yesterday' ):
        class_types.append(file1)
        for file2 in os.listdir(f+'/'+file1):
            i+=1
            data.append(f+'/'+file1+'/'+file2)
            label[f+'/'+file1+'/'+file2] = file1
            strat_label.append(count)
        index_label[file1]=count
        distribution[file1]=i
        train_disrtibution[file1]=0
        test_distribution[file1]=0
        count+=1
        total+=i




#print(len(distribution))
#print(data)
weight_per_class=[]
for label_class in distribution.keys():
    weight_per_class.append(total/distribution[label_class])
weighted_tensor = torch.tensor(weight_per_class)
print(index_label)

def oversample(data_sampled):
    tmp_data_sampled = data_sampled
    while(len(data_sampled)<=1):
        sampled=random.choice(tmp_data_sampled)
        data_sampled.append(sampled)
    return data_sampled


def perform_stratified_split(dataset, split_list,test_size=0.2, n_splits=1, random_state=None):
    # Create a StratifiedShuffleSplit object
    stratified_splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    # Convert the dataset to numpy arrays
    datas = np.array(dataset.data)
    train_oversample=[[] for _ in range(33)]
    train = []
    train_actual=[]
    test=[]
    # Perform the stratified split
    for train_index, test_index in stratified_splitter.split(datas, split_list):
        for t in train_index:
            train_oversample[index_label[dataset.label[dataset.data[t]]]].append(dataset.data[t])
            train_actual.append(dataset.data[t])
        for te in test_index:
            test.append(dataset.data[te])

    for te in test:
        test_distribution[dataset.label[te]] = test_distribution[dataset.label[te]]+1
    for t in train_actual:
        train_disrtibution[dataset.label[t]] = train_disrtibution[dataset.label[t]]+1

    #perform oversample on train dataset
    for i in range(29):
        tmp = oversample(train_oversample[i])
        train.extend(tmp)

    print(train_disrtibution)
    print(test_distribution)
    print(len(train_actual))
    # Create new custom datasets with tuples of data and labels
    train_dataset = skeletalData(train, dataset.label)
    test_dataset = skeletalData(test, dataset.label)

    # Convert the NumPy arrays back to tensors

    return train_dataset, test_dataset


complete_dataset=skeletalData(data,label)
one_hot_vector=torch.eye(len(index_label))

train_dataset, test_dataset = perform_stratified_split(complete_dataset, split_list=np.array(strat_label),test_size=0.3)

batch_size = 1

def batch_graph(batch):
    # Find the maximum depth in the batch
    tensors=[item[0] for item in batch]
    labels=[item[1] for item in batch]
    max_depth = max([item.size(1) for item in tensors])
    actual = torch.tensor([])

    
    for i in range(len(labels)):
        actual=torch.cat((actual,one_hot_vector[index_label[labels[i]]].unsqueeze(0)))
    """
        dif = max_depth - tensors[i].size(1)
        batched_graph = tensors[i]
        for j in range(dif):
            frame_no = random.randint(0,tensors[i].size(1)-1)
            frame_repeat = (tensors[i][:,frame_no,:,:]).unsqueeze(1)
            first_part=batched_graph[:,:frame_no,:,:]
            second_part = batched_graph[:,frame_no:,:,:]
            first_part=torch.cat((first_part,frame_repeat), dim=1)
            batched_graph=torch.cat((first_part,second_part),dim=1)
        tensors[i]=batched_graph
    """

    # Pad tensors in the batch to the maximum depth
    # Stack the padded tensors
    #tensors = [torch.nn.functional.pad(item, (0,0,0,0,0,60)) for item in tensors]
    padded_batch = [torch.nn.functional.pad(item, (0,0,0,0,0,max_depth - item.size(1))) for item in tensors]
    padded_batch = torch.stack(padded_batch, dim=0)

    return padded_batch, actual

# Create data loaders for the training and testing sets
train_loader = DataLoader(train_dataset ,batch_size=batch_size, shuffle=True, collate_fn=batch_graph)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=batch_graph)
torch.save(train_loader, './train_dataloader.pth')
torch.save(test_loader, './test_dataloader.pth')
#test_loader = torch.load("test_dataloader.pth")
#train_loader = torch.load("train_dataloader.pth")
"""
for batch in train_loader:
    tensors, labels = batch
    # Do something with the tensors and labels
    print("Tensors:", tensors.size())
    print("Labels:", labels[0])
"""
#print(index_label)
model = Model(3, 29, {'layout':'openpose_arm', 'strategy':'spatial'}, True)
#pretrained_dict = torch.load('/home/sxp1428/sxp1428/weights_/8.pt')
#model.load_state_dict(pretrained_dict)


"""
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
"""

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
loss_fn = nn.CrossEntropyLoss()
epochs = 100

best_acc = - np.inf   # init to negative infinity
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

"""
if 1==1:

    print("Testing--->>")
    #testing
    y_pred=[]
    y_actual=[]
    epoch_loss=[]
    epoch_acc=[]
    print_los=0.0
    print_acc=0.0
    count=0

    with tqdm.trange(int(len(test_dataset)/1), unit="batch", mininterval=0) as bar:
            bar.set_description(f"Epoch 1")
            for batch in test_loader:
                count+=1
                inputs, actual_label = batch
                output = model.forward(inputs)
                #print(inputs.size())
                #print(one_hot_vector[index_label[actual_label[0]]].unsqueeze(0).size())
                loss= loss_fn(output, actual_label)
                acc=torch.where(torch.argmax(output,dim=1)==torch.argmax(actual_label,dim=1),1,0).float().sum()
                print_acc+=float(acc)
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
                print_los+=float(loss)
                bar.update(1)
                bar.refresh()
                for i in range(actual_label.size()[0]):
                    y_pred.append(class_types[torch.argmax(output[i])])
                    y_actual.append(class_types[torch.argmax(actual_label[i])])
                #print(output)
                #print(one_hot_vector[index_label[actual_label[0]]].unsqueeze(0))
    cf_matrix = confusion_matrix(y_actual, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in class_types],
                        columns = [i for i in class_types])
    plt.figure(figsize = (25,25))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('accuracy_matrix.png')

    print(print_los/len(test_dataset))
    print(print_acc/len(test_dataset))

"""
#with torch.no_grad():
#    for param in model.parameters():
#        param.div_(torch.norm(param, p=2))
#training

for epoch in range(0,30):
    epoch_loss = []
    epoch_acc = []
    print_los=0.0
    print_acc=0.0
    count=0
    with tqdm.trange(int(len(train_dataset)/batch_size), unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for batch in train_loader:
            count+=1
            inputs, actual_label = batch
            output = model.forward(inputs)
            #print(inputs.size())
            optimizer.zero_grad()
            #print(one_hot_vector[index_label[actual_label[0]]].unsqueeze(0).size())
            #l2_reg_loss = 0.001 * sum(torch.sum(parameter ** 2) for parameter in model.parameters())
            #loss= loss_fn(output, actual_label)
            #print(output)
            loss = 1 - torch.mean(torch.sum(output * actual_label, dim=1))
            #print(output)
            #print(one_hot_vector[index_label[actual_label[0]]].unsqueeze(0))
            loss.backward()
            optimizer.step()
            scheduler.step()
            acc=torch.where(torch.argmax(output,dim=1)==torch.argmax(actual_label,dim=1),1,0).float().sum()
            print_acc+=float(acc)
            bar.set_postfix(
                loss=float(loss),
                acc=float(acc)
            )
            print_los+=float(loss)
            bar.update(1)
            bar.refresh()
    print(float(print_acc)/len(train_dataset))
    print(float(print_los)/len(train_dataset))
    torch.save(model.state_dict(), './weights_arm/'+str(epoch)+'.pt')

    print("Testing--->>")
    #testing
    y_pred=[]
    y_actual=[]
    epoch_loss=[]
    epoch_acc=[]
    print_los=0.0
    print_acc=0.0
    count=0


    with tqdm.trange(int(len(test_dataset)/1), unit="batch", mininterval=0) as bar:
            bar.set_description(f"Epoch 1")
            for batch in test_loader:
                count+=1
                inputs, actual_label = batch
                output = model.forward(inputs)
                #print(inputs.size())
                #print(one_hot_vector[index_label[actual_label[0]]].unsqueeze(0).size())
                #loss= loss_fn(output, actual_label)
                loss = 1 - torch.mean(torch.sum(output * actual_label, dim=1))
                acc=torch.where(torch.argmax(output,dim=1)==torch.argmax(actual_label,dim=1),1,0).float().sum()
                print_acc+=float(acc)
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
                print_los+=float(loss)
                bar.update(1)
                bar.refresh()
                for i in range(actual_label.size()[0]):
                    y_pred.append(class_types[torch.argmax(output[i])])
                    y_actual.append(class_types[torch.argmax(actual_label[i])])
                #print(output)
                #print(one_hot_vector[index_label[actual_label[0]]].unsqueeze(0))
    #cf_matrix = confusion_matrix(y_actual, y_pred)
    #df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in class_types],
                        #columns = [i for i in class_types])
    #plt.figure(figsize = (25,25))
    #sn.heatmap(df_cm, annot=True)
    #plt.savefig('accuracy_matrix.png')

    print(print_los/len(test_dataset))
    print(print_acc/len(test_dataset))

