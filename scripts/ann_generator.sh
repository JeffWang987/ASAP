PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

data_path="./data/nuscenes"
data_version="v1.0-trainval"
ann_frequency=$1
PY_ARGS=${@:2}

OUT_DIR="./out/"
LOG_DIR=$OUT_DIR/$input_frequency/'2'_$ann_frequency
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python -m sAP3D.nusc_annotation_generator \
    --data_path $data_path \
    --data_version $data_version \
    --ann_frequency $ann_frequency \
    $PY_ARGS | tee -a $LOG_DIR/log_generate_ann.txt

