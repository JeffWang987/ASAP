PYTHONPATH="$(dirname $0)/..":$PYTHONPATH 
data_path="./data/nuscenes"
data_version="v1.0-trainval"
input_frequency=$1
ann_frequency=$2
results_nusc_path=$3
model_name=$4
PY_ARGS=${@:5}

OUT_DIR="./model_rst"
LOG_DIR=$OUT_DIR/$model_name
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python -m sAP3D.nusc_sAP3D \
    --data_path $data_path \
    --data_version $data_version \
    --input_frequency $input_frequency \
    --ann_frequency $ann_frequency \
    --results_nusc_path $results_nusc_path \
    --model_name $model_name \
    $PY_ARGS | tee -a $LOG_DIR/log_sAP3d.txt
