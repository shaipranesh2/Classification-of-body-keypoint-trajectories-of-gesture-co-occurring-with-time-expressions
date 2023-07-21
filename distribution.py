import os
from dataset import skeletalData
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
f='/home/sxp1428/sxp1428/numpy_arr'
distribution = []
count = []
length=[]
num_per_class = {}
highest = -1
lowest = 99999

fig = plt.figure(figsize = (150, 150))

for file1 in os.listdir(f):
    i=0
    tmp=[]
    for file2 in os.listdir(f+'/'+file1):
        i+=1
        numpy_tmp = np.load(f+'/'+file1+'/'+file2)
        tmp.append(np.shape(numpy_tmp)[1])
        length.append(np.shape(numpy_tmp)[1])
        if(np.shape(numpy_tmp)[1]>highest):
            highest=np.shape(numpy_tmp)[1]
        if(np.shape(numpy_tmp)[1]<lowest):
            lowest=np.shape(numpy_tmp)[1]
    distribution.append(file1)
    count.append(i)
    num_per_class[file1]=i
    tmp=np.array(tmp)
    print("mean length for class: "+file1)
    print(np.mean(tmp))
    print("std deviation length for class: "+file1)
    print(np.std(tmp))
    print("")

length = np.array(length)
print("mean for overall length:")
print(np.mean(length))
print("std deviation for overall length:")
print(np.std(length))

print("")

print("highest number of frames")
print(highest)
print("lowest number of frames")
print(lowest)

print("")
print(num_per_class)

count_np = np.array(count)
print(np.mean(count_np))
print(np.std(count_np))

plt.bar(distribution, count, color ='red',
        width = 0.4,)
 
plt.xlabel("Class label")
plt.ylabel("No. of examples")
plt.title("Data Distribution")
plt.savefig("./distribution.png")

