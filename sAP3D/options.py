import argparse


class SP3DOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="test a mode options")
        self.parser.add_argument('--model_name', 
                                help='model name',
                                type=str,
                                default='FCOS3D')
        self.parser.add_argument('--data_path', 
                                help='only suppory nuscenes',
                                type=str)
        self.parser.add_argument('--input_token_sequence_path', 
                                type=str,
                                default='./assets/12Hz_input_token_dict_val.pkl')
        self.parser.add_argument('--detection_cfg', 
                                type=str,
                                default='./sAP3D/detection_cfg/detection_cvpr_2019.json')
        self.parser.add_argument('--lidar_inf_rst_path', 
                                type=str,
                                default='./out/lidar_20Hz/results_nusc.json')
        self.parser.add_argument('--data_version', 
                                help='nuscenes version',
                                type=str,
                                default='v1.0-trainval')
        self.parser.add_argument('--input_frequency', 
                                help='input frequency',
                                type=int,
                                default=2)
        self.parser.add_argument('--ann_frequency', 
                                help='annotation frequency',
                                type=int,
                                default=2)
        self.parser.add_argument('--time_factor', 
                                type=float,
                                default=1)
        self.parser.add_argument('--dynamic_schedule',
                                action='store_true')
        self.parser.add_argument('--render_anns',
                                action='store_true')
        self.parser.add_argument('--single_view_pred',
                                action='store_true')
        self.parser.add_argument('--lidar_input_json',
                                action='store_true')
        self.parser.add_argument('--output_tmp_path', 
                                type=str,
                                default='./out')
        self.parser.add_argument('--model_rst_path', 
                                type=str,
                                default='./model_rst')
        self.parser.add_argument('--rst_path', 
                                type=str)  # mmdetection3d/sAP/PGD/2Hz_pred/rst.pkl
        self.parser.add_argument('--results_nusc_path', 
                                type=str)  # mmdetection3d/sAP/PGD/2Hz_pred/img_bbox/results_nusc.json
        self.parser.add_argument('--render_rst_path', 
                                type=str,
                                default='./out/render_rst')
        self.parser.add_argument('--imagenet_data', 
                                type=str,
                                default='/mnt/cfs/algorithm/public_data/imagenet-100')
        self.parser.add_argument('--sp_strategy', 
                                type=str,
                                default='copy',
                                choices=['copy', 'discard', 'vel', 'kf'])
        self.parser.add_argument('--ann_strategy', 
                                type=str,
                                default='interp',
                                choices=['interp', 'advanced'])
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options