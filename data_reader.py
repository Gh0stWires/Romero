import envirment_utils
import os
import json


def trial_reader(trial_num):
    path = f'{envirment_utils.checkpoint_dir}trial-{trial_num}/trial_{trial_num}_meta.json'
    if os.path.exists(path):
        with open(path, 'rb') as data:

            meta = json.load(data)
            print("-*- "*15)
            print(f"META DATA FOR TRIAL {trial_num}")
            print("-*- "*15)
            for i in meta:

                if i == 'training_env':
                    print("-*- " * 15)
                    print(f"ENV VARIABLES FOR TRIAL {trial_num}")
                    print("-*- " * 15)
                    for x in meta[i]: print(f'{x}: {meta[i][x]}')
                else:
                    print(f'{i}: ', meta[i])
    else:
        print('Designated path: ', path)
        print('ERROR: PATH NOT FOUND')


if __name__ == '__main__':
    trial_num_promt = input('Desigate trial number: ')
    trial_reader(trial_num_promt)