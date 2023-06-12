#Creates subdirectories of the same structure as in the data's subdirectories
import os
import shutil

def duplicate_directory_structure(src_directory, dest_directory):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    # Traverse the source directory recursively
    for root, directories, _ in os.walk(src_directory):
        # Create corresponding directories in the destination
        for directory in directories:
            src_path = os.path.join(root, directory)
            dest_path = os.path.join(dest_directory, os.path.relpath(src_path, src_directory))
            os.makedirs(dest_path, exist_ok=True)
            print(f"Created directory: {dest_path}")

        # Create subdirectories with the same names as the files in the source directory
        for file in os.listdir(root):
            src_path = os.path.join(root, file)
            if os.path.isfile(src_path):
                file_name = os.path.splitext(file)[0]
                dest_path = os.path.join(dest_directory, os.path.relpath(root, src_directory), file_name)
                os.makedirs(dest_path, exist_ok=True)
                print(f"Created subdirectory: {dest_path}")

if __name__ == '__main__':
    source_directory = './GSoC_DATASET'
    destination_directory = './frames'

    duplicate_directory_structure(source_directory, destination_directory)

