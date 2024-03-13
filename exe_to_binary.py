import os
import math
import argparse
from PIL import Image
from queue import Queue
from threading import Thread
import py7zr


def getBinaryData(filename):
    """
    Extract byte values from binary executable file
    """

    with open(filename, 'rb') as fileobject:

        # read file byte by byte
        data = fileobject.read()

        # while data != b'':
        #     data = fileobject.read(1)

    return data


def save_binary_data(filename, binary_data):
    try:
        dirname = os.path.dirname(filename)
        name, _ = os.path.splitext(filename)
        name = os.path.basename(name)
        folder_path = "/Users/yintianyunxi/Desktop/Malware_1D_CNN/raw/malware" \
            + os.sep + name + '.bin'
        os.makedirs(os.path.dirname(folder_path), exist_ok=True)

        with open(folder_path, 'wb') as file:
            file.write(binary_data)
        print('Binary data saved.')
    except Exception as err:
        print("Error", err)


def run(file_queue):

    while not file_queue.empty():
        filename = file_queue.get()
        binary_data = getBinaryData(filename)
        # target_length = get_size(len(binary_data), width=None)
        # resized_data = resize_binary_data(binary_data, target_length)
        save_binary_data(filename, binary_data)
        file_queue.task_done()


def main(input_dir, thread_number=7):
    # Get all executable files in input directory and add them into queue
    file_queue = Queue()
    for root, directories, files in os.walk(input_dir):
        for filename in files:

            file_path = os.path.join(root, filename)
            file_queue.put(file_path)

    # Start thread
    for index in range(thread_number):
        thread = Thread(target=run, args=(file_queue, width))
        thread.daemon = True
        thread.start()
    file_queue.join()

# zip_file_path = 'path/of/zipfile.7z'

# with py7zr.SevenZipFile(zip_file_path, mode='r', password='infected') as archive:
#     # Extract the contents of the 7z file to the specified directory
#     archive.extractall(path='/Users/yintianyunxi/Desktop/Malware_1D_CNN')
# print("Extraction complete.")

main('/Users/path/of/folder')
