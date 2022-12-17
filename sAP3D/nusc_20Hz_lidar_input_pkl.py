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


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):

    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def fill_one_sample(lidar_token, sample_token, timestamp):
    # lidar token is the token of the top lidar
    sd_rec = nusc.get('sample_data', lidar_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

    mmcv.check_file_exist(lidar_path)

    info = {
        'lidar_path': lidar_path,
        'token': sample_token,
        'sweeps': [],
        'lidar2ego_translation': cs_record['translation'],
        'lidar2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': timestamp,
    }

    l2e_r = info['lidar2ego_rotation']
    l2e_t = info['lidar2ego_translation']
    e2g_r = info['ego2global_rotation']
    e2g_t = info['ego2global_translation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    # obtain sweeps for a single key-frame
    sweeps = []
    while len(sweeps) < 10:
        if not sd_rec['prev'] == '':
            sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                    l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
            sweeps.append(sweep)
            sd_rec = nusc.get('sample_data', sd_rec['prev'])
        else:
            break
    info['sweeps'] = sweeps
    return info


def fill_val_infos(nusc, val_scene_tokens):
    val_nusc_infos = []
    non_key_nums = []

    for sample in mmcv.track_iter_progress(nusc.sample):

        if sample['scene_token'] in val_scene_tokens:
            # obtain info for a single key-frame
            lidar_token = sample['data']['LIDAR_TOP']
            sample_token = sample['token']
            timestamp = sample['timestamp']
            info = fill_one_sample(lidar_token, sample_token, timestamp)
            val_nusc_infos.append(info)
            # obtain infos for the next 9 non-key-frames
            non_key_idx = 0
            lidar_token = nusc.get('sample_data', lidar_token)['next']
            while (lidar_token != '') and (nusc.get('sample_data', lidar_token)['is_key_frame'] == False):
                non_key_idx += 1
                # save non-key frame
                this_sample_token = sample_token + str(non_key_idx)
                this_timestamp = nusc.get('sample_data', lidar_token)['timestamp']
                info = fill_one_sample(lidar_token, this_sample_token, this_timestamp)
                val_nusc_infos.append(info)
                # next sample
                lidar_token = nusc.get('sample_data', lidar_token)['next']

            non_key_nums.append(non_key_idx)

    return val_nusc_infos, non_key_nums


def get_nusc_val_scene_token(nusc):
    splits = create_splits_scenes()
    val_scene_ids = splits['val']
    val_scene_tokens = []
    for scene in nusc.scene:
        if scene['name'] in val_scene_ids:
            val_scene_tokens.append(scene['token'])
    return val_scene_tokens


def create_nuscenes_infos(nusc, version='v1.0-trainval'):
    val_scene_tokens = get_nusc_val_scene_token(nusc)
    val_nusc_infos, non_key_nums = fill_val_infos(nusc, val_scene_tokens)
    metadata = dict(version=version)
    print('val sample: {}'.format(len(val_nusc_infos)))
    data = dict(infos=val_nusc_infos, metadata=metadata)
    return data, non_key_nums


if __name__ == '__main__':
    options = SP3DOptions()
    opts = options.parse()
    print('loading nuscenes dataset...')
    nusc = NuScenes(version=opts.data_version, dataroot=opts.data_path, verbose=False)
    nusc_infos, non_key_nums = create_nuscenes_infos(nusc, version=opts.data_version)
    # save
    info_val_path = os.path.join('./out/lidar_20Hz', '20Hz_lidar_infos_val.pkl')
    mmcv.dump(nusc_infos, info_val_path)
    non_key_nums = np.array(non_key_nums)
    # non_key_nums mean:8.719388602757933, std:1.4948017225512469, min:0, max:11
    print("non_key_nums mean:{}, std:{}, min:{}, max:{}".format(np.mean(non_key_nums), np.std(non_key_nums), np.min(non_key_nums), np.max(non_key_nums)))
