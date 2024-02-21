# This file takes .exe folders and convert exe files into byte sequence

import os
import math
import argparse
from PIL import Image
from queue import Queue
from threading import Thread


def getBinaryData(filename):
    """
    Extract byte values from binary executable file and store them into list
    :param filename: executable file name
    :return: byte value list
    """

    with open(filename, 'rb') as fileobject:

        # read file byte by byte
        data = fileobject.read(1)

        while data != b'':
            data = fileobject.read(1)

    return data


def get_size(data_length, width=None):

    if width is None:

        size = data_length

        if (size < 10000):
            width = 1024
        elif (10000 <= size <= 10000 * 3):
            width = 4096
        elif (10000 * 3 <= size <= 10000 * 6):
            width = 128 * 128
        elif (10000 * 6 <= size <= 10000 * 10):
            width = 256 * 256
        elif (10000 * 10 <= size <= 10000 * 20):
            width = 384 * 384
        elif (10000 * 20 <= size <= 10000 * 50):
            width = 512 * 512
        elif (10000 * 50 <= size <= 10000 * 100):
            width = 768 * 768
        else:
            width = 1024 * 1024

    return width


def resize_binary_data(binary_data, target_length):
    binary_data = bytes(binary_data)
    if len(binary_data) >= target_length:
        return binary_data[:target_length]
    else:
        padding = bytes(target_length - len(binary_data))
        return binary_data + padding


def save_binary_data(filename, binary_data):
    try:
        dirname = os.path.dirname(filename)
        name, _ = os.path.splitext(filename)
        name = os.path.basename(name)
        folder_path = "/data/tiny_binary_files/benign" \
            + os.sep + name + '.bin'
        os.makedirs(os.path.dirname(folder_path), exist_ok=True)

        with open(folder_path, 'wb') as file:
            file.write(binary_data)
        print('Binary data saved.')
    except Exception as err:
        print("Error", err)


def run(file_queue, width):

    while not file_queue.empty():
        filename = file_queue.get()
        binary_data = getBinaryData(filename)
        target_length = get_size(len(binary_data), width=None)
        resized_data = resize_binary_data(binary_data, target_length)
        save_binary_data(filename, resized_data)
        file_queue.task_done()


def main(input_dir, width=None, thread_number=7):
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


main('/Users/yintianyunxi/Desktop/Benign-NET/cnetnet')
