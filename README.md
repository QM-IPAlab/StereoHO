<div align="center">
<h2><center> Stereo Hand-Object Reconstruction for Human-to-Robot Handover </h2>

<a href='https://arxiv.org/abs/2412.07487'><img src='https://img.shields.io/badge/ArXiv-2412.14803-red'></a> 
<a href='https://qm-ipalab.github.io/StereoHO/'><img src='https://img.shields.io/badge/Project-Page-Blue'></a> 

</div>

This is the official repository for "Stereo Hand-Object Reconstruction for Human-to-Robot Handover". This repository contains code for the stereo hand-object reconstruction pipeline and UR5 robot control.

## Installation
This code was tested on Ubuntu 18.04 and ROS Melodic.
### Create conda environments for each module
```
conda env create --file conda_envs/robot_py2.yml --name robot_py2
conda env create --file conda_envs/6dofgraspnet.yml --name 6dofgraspnet
conda env create --file conda_envs/frankmocap.yml --name frankmocap
conda env create --file conda_envs/stereoho.yml --name stereoho
```
### Setup catkin workspace
```
cd catkin_ws
catkin init
catkin config --extend /opt/ros/melodic --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin build
```

### Download model checkpoints
Follow instructions in each module's repository to download model checkpoints
* [pytorch_6dof-graspnet](https://github.com/jsll/pytorch_6dof-graspnet)
* [frankmocap](https://github.com/facebookresearch/frankmocap)
* [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)

Download checkpoints for StereoHO. Unzip and place in `catkin_ws/src/stereo_ho/saved_ckpt`
* [StereoHO model checkpoints](https://drive.google.com/file/d/1uiKxVp2QV8JKkwBFdihFXQEjR9IjEhpc/view?usp=sharing)

## Human-to-robot handover
This code utilises 2 Intel realsense D435i cameras, UR5 robot arm and Robotic 2F-85 gripper.
For each section open a new terminal and run the code from the StereoHO directory.

### Prerequisites
* Images from cameras streaming under the topic `/camera1/image_list` and `/camera2/image_list`
* Corresponding camera calibration files placed in `calib/` directory
    * We used a Charuco board for calibration. Each `cam*` directory contains the extrinsics of the camera relative to the board origin.
    * We additionally perform hand-eye calibration to obtain the transformation matrix from the board to the robot base which is placed in the `robot2` directory.

### Start visualisation node
This node takes the images and overlay the predicted hand and object segmentation masks and pointclouds for visualisation.
```
cd catkin_ws
source devel/setup.bash
conda activate robot_py2 
rosrun main recording.py
```

### Start grasp detection node
This node takes the predicted pointclouds and output valid and safe grasp candidates.
```
cd catkin_ws
source devel/setup.bash
cd src/pytorch_6dof-graspnet
conda activate 6dofgraspnet
python -m demo.ros_srv
```

### Start hand tracking node
This node tracks the 6D pose of the hand wrist.
```
cd catkin_ws
source devel/setup.bash
cd src/frankmocap
conda activate frankmocap
python -m demo.ros_node
```

### Start StereoHO recosntruction node
This node runs StereoHO and reconstruct the hand and object pointclounds.
```
cd catkin_ws
source devel/setup.bash
cd src/stereo_ho
conda activate stereoho
python -m demo.recon_node
```

### Start robot control node
This node takes the grasp candidates and control the UR5 robot arm to receive the object.
```
cd catkin_ws
source devel/setup.bash
cd src/stereo_ho
conda activate robot_py2
python -m demo.robot_node --subject_id 99 --config_id 99
```

## Acknowledgements
This code is built on [AutoSDF](https://github.com/yccyenchicheng/AutoSDF), [pytorch_6dof-graspnet](https://github.com/jsll/pytorch_6dof-graspnet), [frankmocap](https://github.com/facebookresearch/frankmocap) and [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM).
