
# Prepare nuScenes-H
Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download). 
**Folder structure:**
```
ASAP
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
```
Generate 12Hz annotations:
For convinience, the 12Hz annotations can be simiply calculated by the object interpolation:
```
bash scripts/ann_generator.sh 12 --ann_strategy 'interp' 
```
Or you can use the advanced method (object interpolation + temporal database) to generate 12Hz annotation. Firstly you should train CenterPoint on the trainval set of nuScnese (follow the [official instructions in MMDetection3D](https://github.com/open-mmlab/mmdetection3d), and we provide config files in ./asset/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_trainval.py). Then we use the pretrained CenterPoint to generate detection results for 20Hz LiDAR input:
1. Generate 20Hz LiDAR input pkl file for CenterPoint
```
bash scripts/nusc_20Hz_lidar_input_pkl.sh
```
2. Generate 20Hz detection results. We provide a template inference script for CenterPoint inference (**use MMDetection3D**):
```
python tools/test.py \
    $PATH_TO_ASAP/assets/centerpoint_20Hz_lidar_input.py \
    $pretrained_model_path (use the above)\
    --eval-options 'jsonfile_prefix=$PATH_TO_MMDetection3D/work_dirs/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_trainval/' \
    --out $PATH_TO_MMDetection3D/work_dirs/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_trainval/rst.pkl \
    --format-only
```
3. Consequently, we obtain 20Hz inference results at $PATH_TO_MMDetection3D/work_dirs/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_trainval/pts_bbox/results_nusc.json. Then we build the temporal database:
```
bash scripts/nusc_20Hzlidar_instance-token_generator.sh --lidar_inf_rst_path $PATH_TO_MMDetection3D/work_dirs/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_trainval/pts_bbox/results_nusc.json
```

4. Finaly, generate the 12Hz annotation:
```
bash scripts/ann_generator.sh 12 \
   --ann_strategy 'advanced' \
   --lidar_inf_rst_path ./out/lidar_20Hz/results_nusc_with_instance_token.json
```
After the data preparation , the folder structure is:
```
ASAP
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── interp_12Hz_trainval/
|   |   ├── advanced_12Hz_trainval/ (optional)
```

5. Visualize the 12Hz annotation.
```
bash scripts/render_ann.sh interp_12Hz_trainval --render_anns
```
