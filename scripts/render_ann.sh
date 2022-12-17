PYTHONPATH="$(dirname $0)/..":$PYTHONPATH 
data_path="./data/nuscenes"
data_version=$1
PY_ARGS=${@:2}

OUT_DIR="./out/render_rst"
LOG_DIR=$OUT_DIR/$model_name
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python -m render.nusc_annotation_render \
    --data_path $data_path \
    --data_version $data_version \
    $PY_ARGS | tee -a $LOG_DIR/log_render_gt.txt

