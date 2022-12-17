from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import copy
import numpy as np
import json
import os
from glob import glob
from sAP3D.eval.evaluate import DetectionEval as NuScenesEval
from sAP3D.options import SP3DOptions
import mmcv
from nuscenes.eval.detection.data_classes import DetectionConfig
from sAP3D.utils.kalman_filter import KalmanFilter

kf = KalmanFilter(R_fac=40)
clsmap = {
    'car': 0,
    'pedestrian': 1,
    'trailer': 2,
    'truck': 3,
    'bus': 4,
    'motorcycle': 5,
    'construction_vehicle': 6,
    'bicycle': 7,
    'barrier': 8,
    'traffic_cone': 9,
}

def config_factory(cfg_path):
    assert os.path.exists(cfg_path), \
        'Requested unknown configuration {}'.format(cfg_path)

    # Load config file and deserialize it.
    with open(cfg_path, 'r') as f:
        data = json.load(f)
    cfg = DetectionConfig.deserialize(data)

    return cfg

class Empirical():
    def __init__(self, samples, time_factor=1):
        self.samples = np.array(samples)
        assert time_factor > 0, time_factor
        if time_factor != 1:
            self.samples *= time_factor
        print('mean model time: {}'.format(self.mean()))
        print('std model time: {}'.format(self.std()))
        print('draw model time: {}'.format(self.draw()))
        print('max model time: {}'.format(self.max()))
        print('min model time: {}'.format(self.min()))

    def draw(self):
        this_time = np.random.choice(self.samples)
        return this_time
    def mean(self):
        return self.samples.mean()
    def std(self):
        return self.samples.std(ddof=1)
    def min(self):
        return self.samples.min()
    def max(self):
        return self.samples.max()


def modify_token(data, new_token):
    new_datas = copy.deepcopy(data)
    for new_data in new_datas:
        new_data['sample_token'] = new_token
    return new_datas


def kf_interpolation(data, new_token, time_delta):
    new_datas = copy.deepcopy(data)
    # Zt: [x, y, z, vx, vy]
    # cls: [M, 1] label
    Zt = []
    cls = []
    for i in range(len(new_datas)):
        tmp = copy.deepcopy(new_datas[i]['translation'])
        tmp.extend(copy.deepcopy(new_datas[i]['velocity']))
        Zt.append(tmp)
        cls.append(clsmap[new_datas[i]['detection_name']])
    kf_X = kf(np.array(Zt).reshape(-1, 5), np.array(cls), time_delta)  # M, 5

    for i, new_data in enumerate(new_datas):
        new_data['sample_token'] = new_token
        if new_data['detection_name'] in ['cone', 'barrier']:
            continue
        
        new_data['translation'] = kf_X[i, :3]
        new_data['velocity'] = kf_X[i, -2:]
        new_data['translation'][:2] = new_data['translation'][:2] + new_data['velocity'] * time_delta/1000
        new_data['translation'] = list(new_data['translation'])
        new_data['velocity'] = list(new_data['velocity'])
    return new_datas

def streaming_simulation(rst_json_dir):
    with open(opts.results_nusc_path, 'r') as f:
        rets = json.load(f)
    sample_time_itv = 1000 / opts.input_frequency
    ann_time_itv = 1000 / opts.ann_frequency
    time_dist = Empirical(model_time, time_factor=opts.time_factor)

    scene_dict = {}
    for s in nusc2.scene:
        scene_dict[s['name']] = s

    new_rst = {}
    splits = create_splits_scenes()
    val_scene_names = splits['val']
    for val_scene_name in val_scene_names:
        if opts.sp_strategy == 'kf':
            kf.reset()
        val_scene = scene_dict[val_scene_name]
        
        # {time:prediction}
        time_elapsed = 0
        time_input = 0
        this_val_preds = []
        this_sample_token = val_scene['first_sample_token']
        # print('predicting {}...'.format(val_scene_name))
        while True:
            this_model_time = time_dist.draw()
            time_elapsed = time_elapsed + this_model_time
            this_rst = rets['results'][this_sample_token]
            this_pred_info = {'time_input': time_input, 'results': this_rst}
            # time_elapsed: output time of the current sample
            # this_rst: most recent prediction
            this_val_preds.append({time_elapsed: this_pred_info})
            time_input += sample_time_itv  
            this_sample_token = input_token_sequence[this_sample_token]
            if this_sample_token == '':
                break

            if time_elapsed < time_input:
                time_elapsed = time_input

            elif time_elapsed >= time_input:
                while time_elapsed >= time_input + sample_time_itv:
                    time_input += sample_time_itv
                    if input_token_sequence[this_sample_token] != '':
                        this_sample_token = input_token_sequence[this_sample_token]
                    else:
                        this_sample_token = this_sample_token

                if opts.dynamic_schedule:
                    if time_elapsed % sample_time_itv > (time_elapsed+time_dist.mean()) % sample_time_itv:
                        time_elapsed = time_elapsed + sample_time_itv - (time_elapsed % sample_time_itv)

        # time mathcing for streaming evaluation (strategy: copying)
        # final format {sample_token: results}
        # print('matching {}...'.format(val_scene_name))
        time_ann = 0
        match_idx = 0
        this_sample_token = val_scene['first_sample_token']
        this_rst = []
        this_input_time = 0
        while True:
            inf_time = next(iter(this_val_preds[match_idx].keys()))

            if time_ann < inf_time:
                if opts.sp_strategy == 'copy':
                    new_rst[this_sample_token] = modify_token(this_rst, this_sample_token)

                elif opts.sp_strategy == 'kf':
                    time_delta = time_ann - this_input_time
                    assert time_delta >= 0
                    new_rst[this_sample_token] = kf_interpolation(this_rst, this_sample_token, time_delta)

                this_sample_token = nusc2.get('sample', this_sample_token)['next']
                if this_sample_token == '':
                    break
                time_ann += ann_time_itv

            else:
                this_rst = next(iter(this_val_preds[match_idx].values()))['results']
                this_input_time = next(iter(this_val_preds[match_idx].values()))['time_input']
                match_idx += 1
                if match_idx >= len(this_val_preds):
                    while this_sample_token != '':
                        new_rst[this_sample_token] = modify_token(this_rst, this_sample_token)
                        this_sample_token = nusc2.get('sample', this_sample_token)['next']
                    break

    rets['results'] = new_rst

    with open(rst_json_dir, 'w') as f:
        json.dump(rets, f)
    return rst_json_dir

if __name__ == '__main__':
    options = SP3DOptions()
    opts = options.parse()
    if opts.dynamic_schedule:
        print('dynamic schedule!')

    print('loading nuscenes dataset...')
    input_token_sequence = mmcv.load(opts.input_token_sequence_path)
    version2 = str(opts.ann_frequency) + 'Hz_trainval'
    version2 = opts.ann_strategy + '_' + version2
    nusc2 = NuScenes(version=version2, dataroot=opts.data_path, verbose=False)

    # run time config
    rst_dir = os.path.join(opts.model_rst_path, opts.model_name)
    with open(os.path.join(rst_dir, 'model_time.json'), 'r') as f:
        model_time = json.load(f)

    # streaming perception simulation
    if opts.dynamic_schedule:
        rst_json_dir = os.path.join(opts.model_rst_path, opts.model_name, '{}_{}_dy_rst.json'.format(opts.input_frequency, opts.ann_frequency))
    else:
        rst_json_dir = os.path.join(opts.model_rst_path, opts.model_name, '{}_{}_rst.json'.format(opts.input_frequency, opts.ann_frequency))
    if os.path.exists(rst_json_dir):
        os.system('rm {}'.format(rst_json_dir))
    if True:
        print('streaming perception simulation...')
        streaming_simulation(rst_json_dir)
        if opts.sp_strategy == 'kf':
            print('kf track ratio: {}'.format(np.mean(kf.track_ratio)))
    
    # evaluation
    print('evaluating...')
    eval_detection_configs = config_factory(opts.detection_cfg)
    nusc_eval = NuScenesEval(
        nusc2,
        config=eval_detection_configs,
        result_path=rst_json_dir,
        eval_set='val',
        output_dir=rst_dir,
        verbose=False)
    nusc_eval.main(render_curves=False)

