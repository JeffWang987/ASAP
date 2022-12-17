from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import copy
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
import json
import os
import mmcv

input_hz = 12

if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=False)
    input_token_dict = {}
    if input_hz == 12:
        input_datas = mmcv.load('./out/image_12Hz_pkl/12Hz_image_infos_val.pkl')['infos']
    else:
        input_datas = mmcv.load('./data/nuscenes/v1.0-trainval/sample.json')
    for idx, input_data in tqdm(enumerate(input_datas)):

        if idx == len(input_datas) - 1:
            input_token_dict[input_data['token']] = ''
            continue

        this_token = input_data['token']
        next_token = input_datas[idx + 1]['token']
        try:
            if nusc.get('sample', this_token)['next'] != '':
                input_token_dict[this_token] = next_token
            else:
                input_token_dict[this_token] = ''
        except KeyError:
            input_token_dict[this_token] = next_token
    os.makedirs('./out/input_token_sequence/', exist_ok=True)
    mmcv.dump(input_token_dict, './out/input_token_sequence/{}Hz_input_token_dict_val.pkl'.format(input_hz))


# python nusc_input_token_generator.py 