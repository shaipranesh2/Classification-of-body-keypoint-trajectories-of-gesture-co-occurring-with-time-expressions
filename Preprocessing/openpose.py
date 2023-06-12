#Runs openpose and gives the body kepose in json format.
import subprocess
import os
import shutil

def run_bash_command(command):
	subprocess.run(command, shell=True)

if __name__ == '__main__':
	for file0 in os.listdir('./frames'):
		for file1 in os.listdir('./frames/'+file0+'/'):
			for file2 in os.listdir('./frames/'+file0+'/'+file1+"/"):
				for file3 in os.listdir('./frames/'+file0+'/'+file1+"/"+file2):
					command="./openpose/build/examples/openpose/openpose.bin --image_dir "+"./frames/"+file0+"/"+file1+"/"+file2+"/"+file3+"/"+" --write_json ./keypose/"+file0+"/"+file1+"/"+file2+"/"+file3+"/"
					run_bash_command(command)
