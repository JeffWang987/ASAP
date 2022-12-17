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


key_frame_frequency = 2

def get_new_ann(nusc, key_frame_token, timestamp, frame_idx, hz_factor, next_inst_map=None):
    curr_sample_record = nusc.get('sample', key_frame_token)
    next_sample_record = nusc.get('sample', curr_sample_record['next'])
    curr_ann_recs = [nusc.get('sample_annotation', token) for token in curr_sample_record['anns']]
    if next_inst_map is None:
        next_ann_recs = [nusc.get('sample_annotation', token) for token in next_sample_record['anns']]
        next_inst_map = {entry['instance_token']: entry for entry in next_ann_recs}
    t0 = curr_sample_record['timestamp']
    t1 = next_sample_record['timestamp']
    t = timestamp
    # There are rare situations where the timestamps in the DB are off so ensure that t0 < t < t1.
    t = max(t0, min(t1, t))
    new_anns = []
    for curr_ann_rec in curr_ann_recs:

        if curr_ann_rec['instance_token'] in next_inst_map:
            # If the annotated instance existed in the previous frame, interpolate center & orientation.
            next_ann_rec = next_inst_map[curr_ann_rec['instance_token']]
            # Interpolate center.
            new_translation = [np.interp(t, [t0, t1], [c0, c1]) for c0, c1 in zip(curr_ann_rec['translation'], next_ann_rec['translation'])]
            # Interpolate orientation.
            new_rotation = Quaternion.slerp(q0=Quaternion(curr_ann_rec['rotation']),
                                        q1=Quaternion(next_ann_rec['rotation']),
                                        amount=(t - t0) / (t1 - t0))
        else:
            if opts.ann_strategy == 'interp':
                continue
            elif opts.ann_strategy == 'advanced':
                selected_idx = []
                for lidar_idx in range(1, 10):
                    try:
                        if check_instance_in_lidar_inf(
                            nusc_20Hz_rst[curr_ann_rec['sample_token']+str(lidar_idx)],
                            curr_ann_rec['instance_token']):
                            selected_idx.append(lidar_idx)
                    except KeyError:
                        continue

                if len(selected_idx) != 0:
                    matched_idx = np.argmin(np.abs(np.array(selected_idx)/10 - frame_idx/hz_factor))
                    matched_rst = nusc_20Hz_rst[curr_ann_rec['sample_token']+str(selected_idx[matched_idx])]
                    for lidar_rst in matched_rst:
                        try:
                            if lidar_rst['instance_token'] == curr_ann_rec['instance_token']:
                                new_translation = lidar_rst['translation']
                                new_rotation = Quaternion(lidar_rst['rotation'])
                        except KeyError:
                            continue
                else:
                    continue


        if frame_idx == hz_factor-1:
            if curr_ann_rec['instance_token'] in next_inst_map:
                new_next = curr_ann_rec['next']
            else:
                continue
        else:
            new_next = curr_ann_rec['token'] + str(frame_idx+1)

        if frame_idx == 1:
            new_prev = curr_ann_rec['token']
        else:
            new_prev = curr_ann_rec['token'] + str(frame_idx-1)

        new_sample_token = curr_ann_rec['sample_token'] if frame_idx == 0 else curr_ann_rec['sample_token'] + str(frame_idx)

        new_anns.append({
            'token': curr_ann_rec['token'] + str(frame_idx),
            'sample_token': new_sample_token,
            'instance_token': curr_ann_rec['instance_token'],
            'visibility_token': curr_ann_rec['visibility_token'],
            'attribute_tokens': curr_ann_rec['attribute_tokens'],
            'translation': new_translation,
            'rotation': list(new_rotation),
            'size': curr_ann_rec['size'],
            'prev': new_prev,
            'next': new_next,
            'num_lidar_pts': curr_ann_rec['num_lidar_pts'],
            'num_radar_pts': curr_ann_rec['num_radar_pts'],
        })
    return new_anns


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

def check_instance_in_lidar_inf(lidar_inf, instance_token):
    for this_inf in lidar_inf:
        try:
            if this_inf['instance_token'] == instance_token:
                return True
        except KeyError:
            continue
    return False


def generate_sample_with_ann(opts, nusc, sample_val_tokens):
    hz_factor = int(opts.ann_frequency / key_frame_frequency)  # keyframe rate = 2Hz
    new_sample_list = []
    new_sample_annotation_list = []
 
    for sample_idx, this_key_sample_token in tqdm(enumerate(sample_val_tokens)):
        this_key_sample = nusc.get('sample', this_key_sample_token)

        # for ann_strategy == 'interp', it needs next or prev instance tokens to determinate new sample token
        prev_inst_map, next_inst_map = get_prev_next_instance_tokens(this_key_sample)

        # if it is the last sample, stop here
        if this_key_sample['next'] == '':
            # sample
            frame_0 = copy.deepcopy(this_key_sample)
            frame_0['prev'] = frame_0['prev'] + str(hz_factor-1)
            new_sample_list.append(frame_0)
            # sample ann
            key_frame_anns = [nusc.get('sample_annotation', token) for token in this_key_sample['anns']]
            frame0_anns = copy.deepcopy(key_frame_anns)
            for frame0_ann in frame0_anns:
                if opts.ann_strategy == 'interp':
                    if frame0_ann['instance_token'] in prev_inst_map.keys():
                        frame0_ann['prev'] = frame0_ann['prev'] + str(hz_factor-1)

                elif opts.ann_strategy == 'advanced':
                    if frame0_ann['instance_token'] in prev_inst_map.keys():
                        frame0_ann['prev'] = frame0_ann['prev'] + str(hz_factor-1)

                    else:
                        exist_flag = False
                        for select_idx in range(9, 0, -1):
                            try:
                                this_select_lidar_inf =  nusc_20Hz_rst[frame0_ann['prev'] + str(select_idx)]
                            except KeyError:
                                continue
                            if check_instance_in_lidar_inf(this_select_lidar_inf, frame0_ann['instance_token']):
                                exist_flag =True
                                break
                        if exist_flag and frame0_ann['prev'] != '':
                            frame0_ann['prev'] = frame0_ann['prev'] + str(hz_factor-1)

            new_sample_annotation_list.extend(frame0_anns)
            continue
        
        # else, continue with the new sample
        for frame_idx in range(int(hz_factor)):
            # save sample list
            if frame_idx == 0:
                frame_0 = copy.deepcopy(this_key_sample)
                frame_0['next'] = frame_0['token'] + '1'
                if frame_0['prev'] != '':
                    frame_0['prev'] = frame_0['prev'] + str(hz_factor-1)
                new_sample_list.append(frame_0)
            else:
                # 1. token next prev
                new_token = this_key_sample['token'] + str(frame_idx)
                if frame_idx == 1:
                    new_prev = this_key_sample['token']
                else:
                    new_prev = this_key_sample['token'] + str(frame_idx-1)
                if frame_idx == int(hz_factor)-1:
                    new_next = this_key_sample['next']
                else:
                    new_next = this_key_sample['token'] + str(frame_idx+1)
                # 2. scene_token
                new_scene_token = this_key_sample['scene_token']
                # 3. timestamp
                new_timestamp = this_key_sample['timestamp'] + float(frame_idx)/hz_factor * (nusc.get('sample', this_key_sample['next'])['timestamp'] - this_key_sample['timestamp'])
                # 4. data
                new_data = 0  # not used
                # 5. anns
                new_anns = [ann + str(frame_idx) for ann in this_key_sample['anns']]
                # save new sample frame
                new_sample_list.append({
                    'token': new_token,
                    'prev': new_prev,
                    'next': new_next,
                    'scene_token': new_scene_token,
                    'timestamp': new_timestamp,
                })
            
            # save sample_annotation list
            if frame_idx == 0:
                key_frame_anns = [nusc.get('sample_annotation', token) for token in this_key_sample['anns']]
                frame0_anns = copy.deepcopy(key_frame_anns)

                for frame0_ann in frame0_anns:
                    if opts.ann_strategy == 'interp':
                        if frame0_ann['instance_token'] in next_inst_map.keys():
                            frame0_ann['next'] = frame0_ann['token'] + '1'
                        if frame0_ann['instance_token'] in prev_inst_map.keys():
                            frame0_ann['prev'] = frame0_ann['prev'] + str(hz_factor-1)
                    elif opts.ann_strategy == 'advanced':
                        if frame0_ann['instance_token'] in prev_inst_map.keys():
                            frame0_ann['prev'] = frame0_ann['prev'] + str(hz_factor-1)
                            
                        else:
                            exist_flag = False
                            for select_idx in range(9, 0, -1):
                                try:
                                    this_select_lidar_inf =  nusc_20Hz_rst[this_key_sample['prev'] + str(select_idx)]
                                except KeyError:
                                    continue
                                if check_instance_in_lidar_inf(this_select_lidar_inf, frame0_ann['instance_token']):
                                    exist_flag =True
                                    break
                            if exist_flag and frame0_ann['prev'] != '':
                                frame0_ann['prev'] = frame0_ann['prev'] + str(hz_factor-1)


                        if frame0_ann['instance_token'] in next_inst_map.keys():
                            frame0_ann['next'] = frame0_ann['token'] + str(1)
                        else: 
                            exist_flag = False
                            for select_idx in range(1, 10):
                                try:
                                    this_select_lidar_inf = nusc_20Hz_rst[frame0_ann['smaple_token'] + str(select_idx)]
                                except KeyError:
                                    continue
                                if check_instance_in_lidar_inf(this_select_lidar_inf, frame0_ann['instance_token']):
                                    exist_flag =True
                                    break
                            if exist_flag and frame0_ann['next'] != '':
                                frame0_ann['next'] = frame0_ann['token'] + str(1)

                new_sample_annotation_list.extend(frame0_anns)
            else:
                new_ann = get_new_ann(nusc, key_frame_token=this_key_sample['token'], timestamp=new_timestamp, frame_idx=frame_idx, hz_factor=hz_factor, next_inst_map=next_inst_map)
                new_sample_annotation_list.extend(new_ann)

    return new_sample_list, new_sample_annotation_list


def generate_sample_data(opts, nusc, val_scene_ids):
    hz_factor = int(opts.ann_frequency / key_frame_frequency)  # keyframe rate = 2Hz
    new_sample_data_list = []
    for this_sample_data in tqdm(nusc.sample_data):
        if this_sample_data['is_key_frame'] is False:
            continue
        this_scene_token = nusc.get('sample', this_sample_data['sample_token'])['scene_token']
        if nusc.get('scene', this_scene_token)['name'] not in val_scene_ids:
            continue
        if nusc.get('sample', this_sample_data['sample_token'])['next'] == '':
            frame0 = copy.deepcopy(this_sample_data)
            frame0['prev'] = frame0['prev'] + str(hz_factor-1)
            new_sample_data_list.append(frame0)
            continue
        for frame_idx in range(int(hz_factor)):
            if frame_idx == 0:
                frame0 = copy.deepcopy(this_sample_data)
                frame0['next'] = frame0['token'] + '1'
                new_sample_data_list.append(frame0)
            else:
                # note: 'data' is unchanged, but in DetectionEval.__init__, there is a function 'add_center_dist', it needs lidar pose in every frame,
                # the lidar pose is used to calculate the bbox distances to the lidar to filter out some distant boxes. However, we assume the lidar poses
                # are the same during the two key frames, this approximation is not accurate, but it is good enough for the filter process.
                extra_sample_data = copy.deepcopy(this_sample_data)
                extra_sample_data['token'] = extra_sample_data['token'] + str(frame_idx)
                extra_sample_data['sample_token'] = extra_sample_data['sample_token'] + str(frame_idx)
                extra_sample_data['prev'] = extra_sample_data['token'] + str(frame_idx-1)
                extra_sample_data['next'] = this_sample_data['next'] if frame_idx==hz_factor-1 else extra_sample_data['token'] + str(frame_idx+1)
                next_sample_token = nusc.get('sample', this_sample_data['sample_token'])['next']
                extra_sample_data['timestamp'] = this_sample_data['timestamp'] + float(frame_idx)/hz_factor * (nusc.get('sample', next_sample_token)['timestamp'] - this_sample_data['timestamp'])
                
                # ego_pose_token
                # calibrated_sensor_token
                # filename
                append_sample_data = copy.deepcopy(this_sample_data)
                for _ in range(frame_idx):
                    if nusc.get('sample_data', append_sample_data['next'])['is_key_frame']:
                        break
                    append_sample_data = nusc.get('sample_data', append_sample_data['next'])

                extra_sample_data['ego_pose_token'] = nusc.get('sample_data', append_sample_data['token'])['ego_pose_token']
                extra_sample_data['calibrated_sensor_token'] = nusc.get('sample_data', append_sample_data['token'])['calibrated_sensor_token']
                extra_sample_data['filename'] = nusc.get('sample_data', append_sample_data['token'])['filename']
                extra_sample_data['is_key_frame'] = True


                new_sample_data_list.append(extra_sample_data)
    return new_sample_data_list


def save_json(opts, new_sample_list, new_sample_annotation_list, new_sample_data_list=None):
    out_dir = os.path.join(opts.output_tmp_path, str(key_frame_frequency) + '_' + str(opts.ann_frequency))
    out_dir = out_dir + '_{}'.format(opts.ann_strategy)
    os.makedirs(out_dir, exist_ok=True)
    print('Saving new samples, list length: {}'.format(len(new_sample_list)))
    with open(os.path.join(out_dir, 'sample.json'), 'w') as f:
        json.dump(new_sample_list, f)

    print('Saving new sample annotation, list length: {}'.format(len(new_sample_annotation_list)))
    with open(os.path.join(out_dir, 'sample_annotation.json'), 'w') as f:
        json.dump(new_sample_annotation_list, f)

    if new_sample_data_list!= None:
        print('Saving new sample data, list length: {}'.format(len(new_sample_data_list)))
        with open(os.path.join(out_dir, 'sample_data.json'), 'w') as f:
            json.dump(new_sample_data_list, f)

    misc_names = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor', 'ego_pose', 'log', 'scene', 'map']
    for misc_name in misc_names:
        p_misc_name = os.path.join(out_dir, misc_name + '.json')
        if not os.path.exists(p_misc_name):
            source_path = os.path.join(opts.data_path, opts.data_version, misc_name + '.json')
            os.system('cp {} {}'.format(source_path ,p_misc_name))
    return out_dir


if __name__ == '__main__':
    options = SP3DOptions()
    opts = options.parse()
    print('loading nuscenes dataset...')
    nusc = NuScenes(version=opts.data_version, dataroot=opts.data_path, verbose=False)

    # load 20Hz LiDAR inference results
    print('loading 20Hz LiDAR inference results...')
    nusc_20Hz_rst = mmcv.load(opts.lidar_inf_rst_path)

    # generate sample list and sample_annotation list
    print('processing sample lists and annotation lists...')
    sample_val_tokens, val_scene_ids = get_nusc_val_token(nusc)
    new_sample_list, new_sample_annotation_list = generate_sample_with_ann(opts, nusc, sample_val_tokens)

    # gnerate sample_data list
    print('processing sample data lists...')
    new_sample_data_list = generate_sample_data(opts, nusc, val_scene_ids)
    # new_sample_data_list = None
   
    # save new json files
    out_dir = save_json(opts, new_sample_list, new_sample_annotation_list, new_sample_data_list)

    # cp to ./data/nuscenes/
    final_path = os.path.join(opts.data_path, '{}_{}Hz_trainval'.format(opts.ann_strategy, opts.ann_frequency))
    os.system('cp -r {} {}'.format(out_dir, final_path))
    