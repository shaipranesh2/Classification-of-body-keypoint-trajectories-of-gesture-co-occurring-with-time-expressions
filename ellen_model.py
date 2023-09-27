import os
import torch
from torch import nn
import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import skeletalData
import numpy as np
import random
from model import *
import pandas as pd
from torch.optim.lr_scheduler import StepLR
import logging
from matplotlib import pyplot as plt


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # To log to the console
                        logging.FileHandler('training.log')  # To log to a file
                    ])

batch_size=16

test_pth = torch.load("test_dataloader_ellen.pth")
test_loader = DataLoader(dataset=test_pth, batch_size=batch_size, shuffle=True)
train_pth = torch.load("train_dataloader_ellen.pth")
train_loader = DataLoader(dataset=train_pth, batch_size=batch_size, shuffle=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = Model(3, 1, {'layout':'openpose_arm', 'strategy':'spatial'}, True).to(device)
# the below lines can be uncommented to use them to load the weights from appropriate file path
# pretrained_dict = torch.load('./weights_arm/0.pt')
# model.load_state_dict(pretrained_dict)



optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
# scheduler = StepLR(optimizer, step_size=6, gamma=0.5)
loss_fn = nn.BCEWithLogitsLoss()
epochs = 100

best_acc = - np.inf   # init to negative infinity
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

train_acc_list = []
train_loss_list = []
test_acc_list = []
test_loss_list = []


for epoch in range(0,200):
    epoch_loss = []
    epoch_acc = []
    print_los=0.0
    print_acc=0.0
    count=0
    index=0
    with tqdm.trange(int(len(train_loader)) , unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for _, batch in enumerate(train_loader):
            count+=1
            optimizer.zero_grad()
            inputs, actual_label = batch
            inputs, actual_label = inputs.to(device), actual_label.to(device)
            output = model.forward(inputs)
            output_pred = torch.where(output>0,1,0)
            loss= loss_fn(output.squeeze(1), actual_label.float())
            loss.backward()
            acc=torch.where(output_pred.squeeze(1)==actual_label.float(),1,0).float().sum()
            print_acc+=float(acc)
            print_los+=float(loss)
            optimizer.step()
            bar.set_postfix(
                loss=float(loss),
                acc=float(acc)
            )
            bar.update(1)
            bar.refresh()
    print(float(print_acc)/(len(train_loader)))
    print(float(print_los)/(len(train_loader)))
    train_acc_list.append(float(print_acc)/(len(train_loader)*batch_size))
    train_loss_list.append(float(print_los)/(len(train_loader)*batch_size))
    logging.info(f"Epoch {epoch}, Training Accuracy: {float(print_acc)/len(train_loader):.16f}, Training Loss: {float(print_los)/len(train_loader):.16f}")
    torch.save(model.state_dict(), './weights_arm/'+str(epoch)+'.pt')
    optimizer.zero_grad()
    print("Testing--->>")
    #testing
    y_pred=[]
    y_actual=[]
    epoch_loss=[]
    epoch_acc=[]
    print_los=0.0
    print_acc=0.0
    count=0
    
    with tqdm.trange(int(len(test_loader)), unit="batch", mininterval=0) as bar:
            bar.set_description(f"Epoch 1")
            for _,batch in enumerate(test_loader):
                count+=1
                inputs, actual_label = batch
                inputs, actual_label = inputs.to(device), actual_label.to(device)
                output = model.forward(inputs)
                output_pred = torch.where(output>0,1,0)
                loss= loss_fn(output.squeeze(1), actual_label.float())

                acc=torch.where(output_pred.squeeze(1)==actual_label.float(),1,0).float().sum()
                print_acc+=float(acc)
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
                print_los+=float(loss)
                bar.update(1)
                bar.refresh()
    print(print_los/len(test_loader))
    print(print_acc/(len(test_loader)))
    test_acc_list.append(float(print_acc)/(len(test_loader)*batch_size))
    test_loss_list.append(float(print_los)/(len(test_loader)*batch_size))
    logging.info(f"Test Accuracy: {float(print_acc)/len(test_loader):.16f}, Test Loss: {float(print_los)/len(test_loader):.16f}")
    optimizer.zero_grad()

x_axis = [i for i in range(len(test_acc_list))]
plt.plot(x_axis, test_acc_list, 'r-')
plt.title('Test accuracy')
plt.savefig('Test-acc.jpg')
plt.close()

x_axis = [i for i in range(len(test_acc_list))]
plt.plot(x_axis, train_acc_list, 'r-')
plt.title('Train accuracy')
plt.savefig('Train-acc.jpg')
plt.close()

x_axis = [i for i in range(len(test_acc_list))]
plt.plot(x_axis, test_loss_list, 'r-')
plt.title('Test loss')
plt.savefig('Test-loss.jpg')
plt.close()

x_axis = [i for i in range(len(test_acc_list))]
plt.plot(x_axis, train_loss_list, 'r-')
plt.title('Train loss')
plt.savefig('Train-loss.jpg')
plt.close()
