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

dist_thres = 0.5
NameMapping = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}

def get_prev_next_instance_tokens(sample):
    if sample['next'] == '':
        next_inst_map = dict()
    else:
        next_sample_record = nusc.get('sample', sample['next'])
        next_ann_recs = [nusc.get('sample_annotation', token) for token in next_sample_record['anns']]
        next_inst_map = {entry['instance_token']: entry for entry in next_ann_recs}
    if sample['prev'] == '':
        prev_inst_map = dict()
    else:
        prev_sample_record = nusc.get('sample', sample['prev'])
        prev_ann_recs = [nusc.get('sample_annotation', token) for token in prev_sample_record['anns']]
        prev_inst_map = {entry['instance_token']: entry for entry in prev_ann_recs}
    
    return prev_inst_map, next_inst_map

def get_nusc_val_token(nusc):
    splits = create_splits_scenes()
    val_scene_ids = splits['val']
    sample_tokens_all = [s['token'] for s in nusc.sample]
    sample_val_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)
        if scene_record['name'] in val_scene_ids:
            sample_val_tokens.append(sample_token)
    print("total samples in val: {}".format(len(sample_val_tokens)))  # 6019
    return sample_val_tokens, val_scene_ids


def generate_instance_tokens(sample_val_tokens, nusc_20Hz_inf_rst):
    new_20Hz_inf_rst = copy.deepcopy(nusc_20Hz_inf_rst)
    prev_num = []
    next_num = []
    bad_num = 0
 
    for sample_idx, this_key_sample_token in tqdm(enumerate(sample_val_tokens)):
        this_key_sample = nusc.get('sample', this_key_sample_token)
        prev_inst_map, next_inst_map = get_prev_next_instance_tokens(this_key_sample)

        # if instance token does not exist, save instance token according to LiDAR results
        key_frame_anns = [nusc.get('sample_annotation', token) for token in this_key_sample['anns']]
        frame0_anns = copy.deepcopy(key_frame_anns)
        inf_list = new_20Hz_inf_rst[this_key_sample_token]
        prev_sample_token = this_key_sample['prev']
        next_sample_token = this_key_sample['next']

        for frame0_ann in frame0_anns:
            if frame0_ann['instance_token'] not in prev_inst_map:

                gt_translation = np.array(frame0_ann['translation'])
                try:
                    gt_category_name = NameMapping[frame0_ann['category_name']]
                except KeyError:
                    continue
                min_dist = 1e6
                keep_inf_rst = None
                for this_inf in inf_list:
                    this_translation = np.array(this_inf['translation'])
                    this_category_name = this_inf['detection_name']
                    if this_category_name == gt_category_name:
                        this_dist = np.linalg.norm(gt_translation - this_translation)
                        if this_dist < min_dist:
                            min_dist = this_dist
                            keep_inf_rst = this_inf
                if min_dist < dist_thres:
                    keep_inf_rst['instance_token'] = frame0_ann['instance_token']
                else:
                    bad_num += 1
                    continue

                if prev_sample_token != '':
                    for frame_idx in range(9, 0 , -1):
                        save_num = 0
                        min_dist = 1e6
                        keep_prev_inf_rst = None
                        try:
                            prev_inf_list = new_20Hz_inf_rst[prev_sample_token+str(frame_idx)]
                        except KeyError:
                            continue
                        
                        for prev_inf in prev_inf_list:
                            if prev_inf['detection_name'] == keep_inf_rst['detection_name']:
                                prev_translation = np.array(prev_inf['translation'])
                                prev_translation[:2] += (10-frame_idx)/20*np.array(prev_inf['velocity'])
                                prev_dist = np.linalg.norm(np.array(keep_inf_rst['translation']) - prev_translation)
                                if prev_dist < min_dist:
                                    min_dist = prev_dist
                                    keep_prev_inf_rst = prev_inf
                        if min_dist < dist_thres: 
                            keep_prev_inf_rst['instance_token'] = frame0_ann['instance_token']
                            save_num += 1
                    prev_num.append(save_num)
    
            if frame0_ann['instance_token'] not in next_inst_map:
                gt_translation = np.array(frame0_ann['translation'])
                try:
                    gt_category_name = NameMapping[frame0_ann['category_name']]
                except KeyError:
                    continue

                min_dist = 1e6
                keep_inf_rst = None
                for this_inf in inf_list:
                    this_translation = np.array(this_inf['translation'])
                    this_category_name = this_inf['detection_name']
                    if this_category_name == gt_category_name:
                        this_dist = np.linalg.norm(gt_translation - this_translation)
                        if this_dist < min_dist:
                            min_dist = this_dist
                            keep_inf_rst = this_inf
                if min_dist < dist_thres:
                    keep_inf_rst['instance_token'] = frame0_ann['instance_token']
                else:
                    bad_num += 1
                    continue

                if next_sample_token != '':
                    for frame_idx in range(1, 10):
                        save_num = 0
                        min_dist = 1e6
                        keep_next_inf_rst = None
                        try:
                            next_inf_list = new_20Hz_inf_rst[next_sample_token+str(frame_idx)]
                        except KeyError:
                            continue
                        for next_inf in next_inf_list:
                            if next_inf['detection_name'] == keep_inf_rst['detection_name']:
                                next_translation = np.array(next_inf['translation'])
                                next_translation[:2] -= frame_idx/20*np.array(next_inf['velocity'])
                                next_dist = np.linalg.norm(np.array(keep_inf_rst['translation']) - next_translation)
                                if next_dist < min_dist:
                                    min_dist = next_dist
                                    keep_next_inf_rst = next_inf
                        if min_dist < dist_thres: 
                            keep_next_inf_rst['instance_token'] = frame0_ann['instance_token']
                            save_num += 1
                    next_num.append(save_num)
    return new_20Hz_inf_rst, prev_num, next_num, bad_num

if __name__ == '__main__':
    options = SP3DOptions()
    opts = options.parse()
    print('loading nuscenes dataset...')
    nusc = NuScenes(version=opts.data_version, dataroot=opts.data_path, verbose=False)

    # load 20Hz LiDAR inference result
    print('loading 20Hz LiDAR inference results...')
    nusc_20Hz_inf_rst = mmcv.load(opts.lidar_inf_rst_path)['results']

    # generate instance token
    print('generating instance tokens...')
    sample_val_tokens, val_scene_ids = get_nusc_val_token(nusc)
    new_20Hz_inf_rst, prev_num, next_num, bad_num = generate_instance_tokens(sample_val_tokens, nusc_20Hz_inf_rst)
    print('prev_num:', np.array(prev_num).mean())
    print('next_num:', np.array(next_num).mean())
    print('bad_num:', bad_num)

    # save new json files
    print('Saving...')
    out_dir = opts.lidar_inf_rst_path.split('.json')[0]
    with open(out_dir+ '_with_instance_token.json', 'w') as f:
        json.dump(new_20Hz_inf_rst, f)

    
