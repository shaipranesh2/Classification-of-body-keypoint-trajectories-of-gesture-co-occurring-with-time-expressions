#Converts the json file of each frame and joins with all the frames' keyposes into a single .npy file
import subprocess
import os
import shutil
import json
import sys
import numpy as np

if __name__ == '__main__':
	f = sys.argv[1] # File directory passed in as argument. see run.sh
	for file0 in os.listdir(f+'/'):
		st_graph=None
		for file1 in os.listdir(f+'/'+file0+'/'):
			if file1[-4:] == "json":
				json_file=sorted(os.listdir(f+'/'+file0+'/'+file1+'/'))
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
							if(person>0): # appending multiple person's coordinates.
								while i <= 17:
									x[i].append(pose[x_i])
									x_i+=3
									y[i].append(pose[y_i])
									y_i+=3
									score[i].append(pose[score_i])
									score_i+=3
									i+=1
							else:
								while i <= 17:
									x.append([pose[x_i]])
									x_i+=3
									y.append([pose[y_i]])
									y_i+=3
									score.append([pose[score_i]])
									score_i+=3
									i+=1
							person+=1
					frame.append(x)
					frame.append(y)
					frame.append(score)
					print(file0,file1,file2)
					tmp = np.array(frame)
					if(tmp.shape[-1]==0):
						continue
					if(st_graph is not None):
						tmp_graph = np.array(frame)
						tmp_graph = np.expand_dims(tmp_graph, axis=1)
						print(np.shape(tmp_graph))
						print(np.shape(st_graph))
						num_zeros = st_graph.shape[-1] - tmp_graph.shape[-1]
						# when joining the current frames numpy array with the initial one, we check for the bigger numpy array and calculate the difference to pad in the smaller array to match the bigger one.
						if(num_zeros>0):
							tmp_graph = np.pad(tmp_graph, [(0, 0)] * (tmp_graph.ndim - 1) + [(0, num_zeros)])
						elif(num_zeros<0):
							num_zeros*=-1
							st_graph = np.pad(st_graph, [(0, 0)] * (st_graph.ndim - 1) + [(0, num_zeros)])
						st_graph = np.concatenate((st_graph, tmp_graph),axis=1)
					else:
						st_graph = np.array(frame)
						st_graph = np.expand_dims(st_graph, axis=1)
		st_graph=np.expand_dims(st_graph, axis=0)
		np.save(f+'/'+file0+'/'+file0+'.npy',st_graph)
		# The npy file has numpy array in the shape of (batch, channel, frame, joint, max_person). here by default I chose batch to be 1.
		
