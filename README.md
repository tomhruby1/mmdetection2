# MMDetection 2 - Based Traffic Signs Detector
Cascade R-CNN detector used for detection of Czech Traffic Signs. The environment is deliviered as docker container.

## Docker Prerequisites

Before running the Cascade R-CNN detector, make sure you have Docker installed on your system.

To install Docker, follow the official installation guide:

- [Docker Engine for Linux](https://docs.docker.com/engine/install/)

To use nvidia acceleration the NVIDIA Container Toolkit is necessary, folow the instruction;
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Getting started
1. clone this repository
2. download weights to `[repository_dir]/checkpoints`
3. build docker container using
```
    cd mmdetection-2.28
    docker build -t mmdet2 docker
```
5. drop into the docker container environement, while mounting the input and output data as in you system as /data to docker container:
```
    docker run -it -v ./:/root/mmdet2_mount -v [path to data]:/data --gpus all mmdet2
```
4. run inference on a camera reel in format of `cam0, cam1, ...,cam5` subdirectories containing camera frames using 
    `python inference_reel.py /data/[path_to_mounted_reel] /data/[path_to_desired_output]` 
