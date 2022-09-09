import os
import envirment_utils
import json


def batch_check():
    processed_images_dir = os.listdir(envirment_utils.processed_directory)
    data_len = len(processed_images_dir)
    batch_size = envirment_utils.batch_size

    print('Checking environment...')
    if data_len < batch_size:
        print(f'BATCH_SIZE: {batch_size}, DATA_IMAGE_COUNT: {data_len}')
        print(f"Batch size must be smaller or equal to data image count within:\n  ")
        print(envirment_utils.processed_directory, '\n')
        batch_prompt = input("Please input a lower number: ")
        envirment_utils.batch_size = int(batch_prompt)
        print('Batch size updated to', envirment_utils.batch_size)


def trailing_slash_check(dir_path):
    try:
        last_char = dir_path[-1]

        if last_char == '/':
            return dir_path
        print('Correcting file path...')
        return dir_path + '/'

    except Exception as err:
        print(err)

