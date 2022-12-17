# Step-by-step installation instructions


**1. Create a conda virtual environment and activate it.**
```shell
conda create -n ASAP python=3.7 -y
conda activate ASAP
```

**2. Install nuscenes-devkit following the [official instructions](https://github.com/nutonomy/nuscenes-devkit).**
```shell
pip install nuscenes-devkit
```

**3. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge nvidia
```

**4. Install MMCV following the [official instructions](https://github.com/open-mmlab/mmcv).**
```shell
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

**5. Install imageio for visualization (optional).**
```shell
conda install imageio
```