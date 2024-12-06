# BEVDet_Dual

## Introduction
We build a dual-branch bird=eye-view perception model and mainly refer to the following two papers:
1.  https://arxiv.org/abs/2112.11790
2.  https://arxiv.org/abs/2203.17054


## Get Started

#### Installation and Data Preparation

step 1. Please prepare environment:
```
torch=1.10.0+cu113
torchvision=0.11.1+cu113
mmcv-full=1.5.3
mmdet=2.25.1
mmsegmentation=1.0.0rc4
```

step 2. Prepare bevdet repo by.
```shell script
git clone https://github.com/Wj-costumer/BEVDet_Dual.git
cd BEVDet
pip install -v -e .
```

step 3. Prepare nuScenes dataset as introduced in [nuscenes_det.md](docs/en/datasets/nuscenes_det.md) and create the pkl for BEVDet by running:
```shell
python tools/create_data_bevdet.py
```
step 4. For Occupancy Prediction task, download (only) the 'gts' from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) and arrange the folder as:
```shell script
└── nuscenes
    ├── v1.0-trainval (existing)
    ├── sweeps  (existing)
    ├── samples (existing)
    └── gts (new)
```

#### Train model
```shell
# single gpu
python tools/train.py configs/bevdet_dual_occ/bevdet-occ-r50-4d-stereo.py
# multiple gpu
./tools/dist_train.sh configs/bevdet_dual_occ/bevdet-occ-r50-4d-stereo.py num_gpu
```

#### Test model
```shell
# single gpu
python tools/test.py configs/bevdet_dual_occ/bevdet-occ-r50-4d-stereo.py $checkpoint --eval mAP
# multiple gpu
./tools/dist_test.sh configs/bevdet_dual_occ/bevdet-occ-r50-4d-stereo.py $checkpoint num_gpu --eval mAP
```

## Next Steps
- Optimize the model structure
- Finish the tensorrt accelerating version
- Add fps test code
- Optimize the visualization code