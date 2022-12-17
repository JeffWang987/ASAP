from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import copy
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
import json
import os
from nuscenes.utils.data_classes import Box
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import imageio
from glob import glob
from sAP3D.options import SP3DOptions


def make_video_from_images(image_dir, save_video_path, fps=12):
    if isinstance(image_dir, str):
        image_names = os.listdir(image_dir)
        image_names.sort()
        image_dir = [os.path.join(image_dir, image_name) for image_name in image_names if image_name[-3:] in ['jpg', 'png']]
    assert isinstance(image_dir, list)
    save_mp4_filename = save_video_path + '/gt_vedio.mp4'
    with imageio.get_writer(save_mp4_filename, mode='I', fps=fps) as writer:
        for i in range(len(image_dir)):
            image = imageio.imread(image_dir[i])
            writer.append_data(image)

def view_points(points, view, normalize):
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        d = points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
        points = points / np.maximun(d, torch.ones_like(d) * 1e-9)

    return points


def box_in_image(box, intrinsic, imsize):
    thres = 5
    corners_3d = box.corners()
    corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]

    visible = np.logical_and(corners_img[0, :] > thres, corners_img[0, :] < imsize[0]-thres)
    visible = np.logical_and(visible, corners_img[1, :] < imsize[1]-thres)
    visible = np.logical_and(visible, corners_img[1, :] > thres)
    visible = np.logical_and(visible, corners_3d[2, :] > thres)
    in_front = corners_3d[2, :].mean() > 0.5  # True if the center is at least 0.5 meter in front of the camera.

    return any(visible) and in_front

def render_single_view(nusc, axes, sample, cam_name, ann_tokens):
    axes.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    sample_data_token = sample['data'][cam_name]
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    data_path = os.path.join('./data/nuscenes', sd_record['filename'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])
    imsize = (sd_record['width'], sd_record['height'])
    axes.set_xlim([0, sd_record['width']])
    axes.set_ylim([sd_record['height'], 0])
    im = Image.open(data_path)
    axes.imshow(im)
    if opts.render_anns:
        for ann_token in ann_tokens:
            record = nusc.get('sample_annotation', ann_token)
            this_box = Box(record['translation'], record['size'], Quaternion(record['rotation']),
                    name=record['category_name'], token=record['token'])
            this_box.translate(-np.array(pose_record['translation']))
            this_box.rotate(Quaternion(pose_record['rotation']).inverse)
            this_box.translate(-np.array(cs_record['translation']))
            this_box.rotate(Quaternion(cs_record['rotation']).inverse)
            if not box_in_image(this_box, cam_intrinsic, imsize,):
                continue
            this_box.render(axes, view=cam_intrinsic, normalize=True, colors=['orange']*3, linewidth=4)



def render_surround(nusc, sample_token, save_scene_path, save_id):
    CAM_NAMES = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
    this_sample = nusc.get('sample', sample_token)

    ann_tokens = this_sample['anns']
    fig, axes = plt.subplots(2, 3, figsize=(48,18))

    for cam_idx, cam_name in enumerate(CAM_NAMES):
        render_single_view(nusc, axes[cam_idx//3][cam_idx%3], this_sample, cam_name, ann_tokens)
    plt.savefig(os.path.join(save_scene_path, '{:0>4d}.jpg'.format(save_id)), bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.close("all")


if __name__ == '__main__':
    options = SP3DOptions()
    opts = options.parse()

    nusc = NuScenes(version=opts.data_version, dataroot=opts.data_path, verbose=False)

    render_fps = 12 if '12' in opts.data_version else 2
    if opts.render_anns:
        save_dir = os.path.join(opts.render_rst_path, '{}Hz_gt'.format(render_fps))
    else:
        save_dir = os.path.join(opts.render_rst_path, '{}Hz_raw'.format(render_fps))
    os.makedirs(save_dir, exist_ok=True)

    splits = create_splits_scenes()
    val_scene_ids = splits['val']

    scene_dict = {}
    for s in nusc.scene:
        scene_dict[s['name']] = s

    # for render_scene_idx in range(3):
    for render_scene_idx in [8, 9]:
        save_img_dir = os.path.join(save_dir, 'img-scene-{}'.format(render_scene_idx))
        save_video_dir = os.path.join(save_dir, 'video-scene-{}'.format(render_scene_idx))
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_video_dir, exist_ok=True)
        first_val_scene = scene_dict[val_scene_ids[render_scene_idx]]
        this_val_scene_sample_token = first_val_scene['first_sample_token']  # fd8420396768425eabec9bdddf7e64b6
        save_id = 0
        while nusc.get('sample', this_val_scene_sample_token)['next'] != '':
            render_surround(nusc, this_val_scene_sample_token, save_img_dir, save_id)
            this_val_scene_sample_token = nusc.get('sample', this_val_scene_sample_token)['next']
            save_id += 1

        make_video_from_images(save_img_dir, save_video_dir, fps=render_fps)
        os.system('rm -r {}'.format(save_img_dir))