import csv
import os
import numpy as np

def parse_csv(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        i=1
        for row in reader:
            print(row[0])  # Modify this line to process each row as desired
            if i==2:
                break
            i+=1

# Example usage

f='/home/sxp1428/sxp1428/tidyData'

for file1 in os.listdir(f):
    if(file1=='thank_god'):
        continue
    for file2 in os.listdir(f+'/'+file1+'/'):
        arr=[[], [], []]
        print(f+'/'+file1+'/'+file2)
        with open(f+'/'+file1+'/'+file2, 'r') as csvfile:
            reader = csv.reader(csvfile)
            i=-1
            x=[]
            y=[]
            score=[]
            for row in reader:
                if i==-1:
                    i+=1
                    continue
                if i%25==0 and i!=0:
                    frame=[]
                    arr[0].append(x)
                    arr[1].append(y)
                    arr[2].append(score)
                    x=[]
                    y=[]
                    score=[]
                if row[1]!='NA':
                    tmp_x=float(row[1])
                else :
                    tmp_x=0.0
                if row[0]!='NA':
                    tmp_score=float(row[0])
                else :
                    tmp_score=0.0
                if row[2]!='NA':
                    tmp_y=float(row[2])
                else :
                    tmp_y=0.0

                score.append([tmp_score])
                x.append([tmp_x])
                y.append([tmp_y])
                i+=1
        npy=np.array(arr)
        npy[0]=(npy[0]-np.min(npy[0]))/(np.max(npy[0])-np.min(npy[0]))
        npy[1]=(npy[1]-np.min(npy[1]))/(np.max(npy[1])-np.min(npy[1]))
        # npy[2]=(npy[2]-np.min(npy[2]))/(np.max(npy[2])-np.min(npy[2]))
        np.save('numpy_arr'+'/'+file1+'/'+file2[:-3]+'npy', npy)
