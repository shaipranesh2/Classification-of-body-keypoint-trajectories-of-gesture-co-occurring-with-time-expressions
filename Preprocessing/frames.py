# Extracts the frames of each video using ffmpeg and places it in the subdirectory.
import subprocess
import os
import shutil

def run_bash_command(command):
	subprocess.run(command, shell=True)

if __name__ == '__main__':
	for file0 in os.listdir('./GSoC_DATASET'):
		for file1 in os.listdir('./GSoC_DATASET/'+file0+'/'):
			for video in os.listdir('./GSoC_DATASET/'+file0+'/'+file1+"/"):
				vid=video[0:-4]
				run_bash_command("ffmpeg -i ./GSoC_DATASET/"+file0+"/"+file1+"/"+video+" ./frames/"+file0+"/"+file1+"/"+vid+"/%05d.png")
