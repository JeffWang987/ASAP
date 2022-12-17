from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import copy
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
import json
import os
from sAP3D.options import SP3DOptions
import mmcv

# mmdet3d input json requirement: camera_types should not be reordered
camera_types = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]
nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier')


def fill_one_sample(sensor_token, sample_token, coco_id_idx):
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    (height, width, _) = mmcv.imread(os.path.join(opts.data_path, sd_rec['filename'])).shape
    
    return dict(
            file_name=sd_rec['filename'],
            # id=sd_rec['token'],
            # tricky here: as pycoco initialization needs createIndex(), which reindex images
            id=str(coco_id_idx),  
            token=sample_token,
            cam2ego_rotation=cs_record['rotation'],
            cam2ego_translation=cs_record['translation'],
            ego2global_rotation=pose_record['rotation'],
            ego2global_translation=pose_record['translation'],
            cam_intrinsic=cs_record['camera_intrinsic'],
            width=width,
            height=height)


def fill_val_infos(nusc, val_scene_tokens):
    non_key_nums = []
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    coco_id_idx = 0

    for sample in mmcv.track_iter_progress(nusc.sample):
        if sample['scene_token'] in val_scene_tokens:

            # add this key frame
            sample_token = sample['token']
            cam_token_dict = {} 
            for cam_type in camera_types:
                # obtain dict for a single key-frame
                cam_token_dict[cam_type] = sample['data'][cam_type]
                key_frame_dict = fill_one_sample(cam_token_dict[cam_type], sample_token, coco_id_idx)
                coco_id_idx += 1
                coco_2d_dict['images'].append(key_frame_dict)
                
            # obtain dicts for 5 non-key-frames
            non_key_idx = 0
            lead_sensor_token = sample['data']['CAM_FRONT'] 
            lead_sensor_token = nusc.get('sample_data', lead_sensor_token)['next'] # use CAM_FRONT instead of lidar
            while (lead_sensor_token != '') and (nusc.get('sample_data', lead_sensor_token)['is_key_frame'] == False):

                this_copy_sample = 0
                for cam_type in camera_types:
                    next_sensor_token = nusc.get('sample_data', cam_token_dict[cam_type])['next']
                    if next_sensor_token == '':
                        this_copy_sample += 1
                    else:
                        cam_token_dict[cam_type] = next_sensor_token
                if this_copy_sample >= 3:
                    break
                
                non_key_idx += 1
                this_sample_token = sample_token + str(non_key_idx)
                for cam_type in camera_types:
                    non_key_frame_dict = fill_one_sample(cam_token_dict[cam_type], this_sample_token, coco_id_idx)
                    coco_id_idx += 1
                    coco_2d_dict['images'].append(non_key_frame_dict)

                lead_sensor_token = nusc.get('sample_data', lead_sensor_token)['next']
                
            non_key_nums.append(non_key_idx)
    return coco_2d_dict, non_key_nums


def get_nusc_val_scene_token(nusc):
    splits = create_splits_scenes()
    val_scene_ids = splits['val']
    val_scene_tokens = []
    for scene in nusc.scene:
        if scene['name'] in val_scene_ids:
            val_scene_tokens.append(scene['token'])
    return val_scene_tokens


def create_nuscenes_infos(nusc):

    val_scene_tokens = get_nusc_val_scene_token(nusc)
    coco_2d_dict, non_key_nums = fill_val_infos(nusc, val_scene_tokens)
    print('val sample: {}'.format(len(coco_2d_dict['images'])))
    return coco_2d_dict, non_key_nums


if __name__ == '__main__':

    options = SP3DOptions()
    opts = options.parse()
    print('loading nuscenes dataset...')
    nusc = NuScenes(version=opts.data_version, dataroot=opts.data_path, verbose=False)
    coco_2d_dict, non_key_nums = create_nuscenes_infos(nusc)
    # save
    save_path = os.path.join('./out/img_12Hz', 'nuscenes_infos_val_12Hz_mono3d.coco.json')
    mmcv.dump(coco_2d_dict, save_path)
    non_key_nums = np.array(non_key_nums)
    print("non_key_nums mean:{}, std:{}, min:{}, max:{}".format(np.mean(non_key_nums), np.std(non_key_nums), np.min(non_key_nums), np.max(non_key_nums)))
