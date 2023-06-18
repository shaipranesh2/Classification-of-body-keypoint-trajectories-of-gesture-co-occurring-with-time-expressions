#Converts the json file of each frame and joins with all the frames' keyposes into a single .npy file

import subprocess
import os
import shutil
import json
import sys
import numpy as np

person_keypose={}

#this function matches the person across frames by measuting the euclidean distance between the keyposes.
def match(arr,st_gcn):
	lowest=999999
	low_id=-1
	for id in person_keypose.keys():
		distance=np.linalg.norm(arr-person_keypose[id])
		if distance<=lowest:
			low_id=id
			lowest=distance	
	if (lowest<=1000 and lowest>=-1000):
		person_keypose[low_id]=arr
		n,_,t,_=arr.shape
		tmp=np.resize(arr,(n,t))
		st_gcn[:,-1, :,low_id]=tmp
	else:
		n,c,t,w=st_gcn.shape
		new_shape = (n, c, t, w+1)
		tmp = np.zeros(new_shape)
		tmp[:, :, :, :-1] = st_gcn
		st_gcn=tmp
		n,_,t,_=arr.shape
		tmp=np.resize(arr,(n,t))
		st_gcn[:,-1,:,-1]=tmp
		person_keypose[w]=arr
	print(np.shape(st_gcn))
	return st_gcn
			
		

if __name__ == '__main__':
	f = sys.argv[1]
	for file0 in os.listdir(f+'/'):
		st_graph=None
		print(file0)
		for file1 in os.listdir(f+'/'+file0+'/'):
			if file1[-4:] == "json":
				json_file=sorted(os.listdir(f+'/'+file0+'/'+file1+'/'))
				person_keypose={}
				for file2 in json_file:
					frame=[]
					x=[]
					y=[]
					score=[]
					file_path = f+'/'+file0+'/'+file1+'/'+file2
					with open(file_path, 'r') as file:
						data = json.load(file)
						peoples = data['people']
						person=0
						for people in peoples:
							pose = people['pose_keypoints_2d']
							i=0
							x_i=0
							y_i=1
							score_i=2
							while i <= 17:
								x.append([pose[x_i]])
								x_i+=3
								y.append([pose[y_i]])
								y_i+=3
								score.append([pose[score_i]])
								score_i+=3
								i+=1
								a=[]
							a.append(x)
							a.append(y)
							a.append(score)
							a=np.array(a)
							a=np.resize(a,(3,1,18,1))
							if not bool(person_keypose):
								person_keypose[0]=a
								st_graph=a
								n, _, t, w = st_graph.shape
								zero_layer = np.zeros((n, 1, t, w))
								st_graph = np.concatenate((st_graph, zero_layer),axis=1)
							else:
								st_graph=match(a, st_graph)
								n, _, t, w = st_graph.shape
								zero_layer = np.zeros((n, 1, t, w))
								st_graph = np.concatenate((st_graph, zero_layer),axis=1)
							person+=1
		st_graph=st_graph[:,:-1,:,:]
		st_graph=np.expand_dims(st_graph, axis=0)
		print(np.shape(st_graph))
		np.save(f+'/'+file0+'/'+file0+'.npy',st_graph)
