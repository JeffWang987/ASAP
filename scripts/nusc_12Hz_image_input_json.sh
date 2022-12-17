PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

data_path="./data/nuscenes"
data_version="v1.0-trainval"
PY_ARGS=${@:1}

OUT_DIR="./out/"
LOG_DIR=$OUT_DIR/'img_12Hz'
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python -m sAP3D.nusc_12Hz_image_input_json \
    --data_path $data_path \
    --data_version $data_version \
    $PY_ARGS | tee -a $LOG_DIR/log.txt
