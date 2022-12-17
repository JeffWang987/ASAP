# Streaming evaluation
**1. Generate 12Hz input file for camera-based 3D detector (e.g., FCOS3D in MMDetection3D). Then we obtain 12Hz input file at ./out/img_12Hz/nuscenes_infos_val_12Hz_mono3d.coco.json.**
```
bash scripts/nusc_12Hz_image_input_json.sh 
```
**2. Generate 20Hz detection results (we provide a template inference script for FCOS3D inference in MMDetection3D). Then we obtain 12Hz results in $PATH_TO_MMDetection3D/work_dirs/12Hz/img_bbox/results_nusc.json.**:
```
python tools/test.py \
    ./configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py \
    $pretrained_model_path (provided by MMDet3D) \
    --eval-options 'jsonfile_prefix=$PATH_TO_MMDetection3D/work_dirs/12Hz/' \
    --cfg-options 'data.test.ann_file=$PATH_TO_ASAP/out/img_12Hz/nuscenes_infos_val_12Hz_mono3d.coco.json' \
    --out $PATH_TO_MMDetection3D/work_dirs/12Hz/rst.pkl \
    --format-only
```

**3. The model runtime should be mannually recorded during the above inference  (on a specific GPU / when GPU is simultaneously processing other tasks), a template script is as bellow. Please save the time file at  $PATH_TO_ASAP/model_rst/model_name/mode_time.json.**
```
time_dist = []
torch.cuda.synchronize()
with torch.no_grad():
    start_time = time.perf_counter()
    result = model(**data)
    torch.cuda.synchronize()
    duration = time.perf_counter() - start_time
    time_dist.append(1000 * duration)  # ms
with open('$PATH_TO_ASAP/model_rst/FCOS3D/mode_time.json', 'w') as f:
    json.dump(time_dist, f)
```

**4. Streaming evaluatin (e.g., FCOS3D)**
```
bash scripts/streaming_eval.sh 12 12 \
    $PATH_TO_MMDetection3D/work_dirs/12Hz/img_bbox/results_nusc.json FCOS3D \
    --input_token_sequence_path ./assets/12Hz_input_token_dict_val.pkl \
    --ann_strategy 'interp'

# To use the velocity-based updating baseline
bash scripts/streaming_eval.sh 12 12 \
    $PATH_TO_MMDetection3D/work_dirs/12Hz/img_bbox/results_nusc.json FCOS3D \
    --input_token_sequence_path ./assets/12Hz_input_token_dict_val.pkl \
    --ann_strategy 'interp' \
    --sp_strategy "kf"
```
