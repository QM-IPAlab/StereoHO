# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys, shutil
import os.path as osp
import numpy as np
import cv2
import json
import torch
from torchvision.transforms import Normalize

from demo.demo_options import DemoOptions
import mocap_utils.general_utils as gnu
import mocap_utils.demo_utils as demo_utils

from handmocap.hand_mocap_api import HandMocap
from handmocap.hand_bbox_detector import HandBboxDetector

import renderer.image_utils as imu
from renderer.viewer2D import ImShow
import time

import rospy
from rospy.numpy_msg import numpy_msg
from frankmocap.msg import Int16, Float32, MocapOutput
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError
# cvBridge = CvBridge()
# From IHOI
from ihoi.nnutils.hand_utils import ManopthWrapper
from ihoi.nnutils.handmocap import get_handmocap_predictor, process_mocap_predictions, get_handmocap_detector
from ihoi.nnutils.geom_utils import se3_to_matrix
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms import Transform3d, Rotate, Translate, Scale
from ihoi.nnutils import geom_utils
import pickle
from renderer.visualizer import Visualizer

class ImgViz:
    def __init__(self):
        self.cam1_image = None
        self.cam2_image = None
        self.img_crop = None
        self.hand_pose = None
        self.pred_output_list = None
        self.hand_bbox_list = None
        self.vertices_l = None
        self.faces_l = None
        self.vertices_r = None
        self.faces_r = None
        self.cam_f = torch.tensor([[10, 10]], dtype=torch.float32)
        self.cam_p = torch.tensor([[0, 0]], dtype=torch.float32)
        self.img_size = 256
        self.camera = PerspectiveCameras(self.cam_f, self.cam_p, device='cuda', image_size=(self.img_size, self.img_size))
        # Points for coordinate system
        coords = np.array([[0, 0, 0],
                            [0.1, 0, 0],
                            [0, 0.1, 0],
                            [0, 0, 0.1]])
        self.coords = torch.from_numpy(coords).float().unsqueeze(0)

        rospy.init_node('frankmocap', anonymous=True, disable_signals=True)
        self.rate = rospy.Rate(30)
        rospy.Subscriber("/camera1/image_list", numpy_msg(Int16), self.callbackCam1)
        rospy.Subscriber("/camera2/image_list", numpy_msg(Int16), self.callbackCam2)
        rospy.Subscriber("/hand_mocap/cropped_image", numpy_msg(Int16), self.callbackImgCrop)
        rospy.Subscriber('/hand_mocap/hand_pose', numpy_msg(Float32), self.callbackHandPose)
        rospy.Subscriber("/hand_mocap/pred_output_r", MocapOutput, self.callbackMCR)
        rospy.Subscriber("/hand_mocap/pred_output_l", MocapOutput, self.callbackMCL)
        self.visualizer = Visualizer("opengl")

        time.sleep(2.0)

    def callbackCam1(self, data):
        bgr = np.reshape(data.data, (720,1280,3))
        bgr = bgr.astype(np.uint8)

        self.cam1_image = cv2.resize(bgr, (640,360))
    
    def callbackCam2(self, data):
        bgr = np.reshape(data.data, (720,1280,3))
        bgr = bgr.astype(np.uint8)

        self.cam2_image = cv2.resize(bgr, (640,360))
    
    def callbackMCR(self, data):
        self.vertices_r = np.array(data.vertices).reshape(-1, 3)
        self.faces_r = np.array(data.faces).reshape(-1, 3).astype(np.int32)
    
    def callbackMCL(self, data):
        self.vertices_l = np.array(data.vertices).reshape(-1, 3)
        self.faces_l = np.array(data.faces).reshape(-1, 3).astype(np.int32)
    
    def callbackImgCrop(self, data):
        img_crop = np.reshape(data.data, (self.img_size,self.img_size,3))
        self.img_crop = img_crop.astype(np.uint8)
    
    def callbackHandPose(self, data):
        hand_pose = np.reshape(data.data, (1,4,4))
        self.hand_pose = torch.from_numpy(hand_pose).float()
    
    def drawPose(self, img):
        # Construct transform
        trans = Transform3d(matrix=self.hand_pose.transpose(1,2))
        norm_coords = trans.transform_points(self.coords)

        camera = PerspectiveCameras(self.cam_f, self.cam_p, device='cpu', image_size=(256, 256))
        coords_2d = camera.transform_points_screen(norm_coords)
        coords_2d = coords_2d[0][:,:2].numpy()

        # Draw coords on cropped image
        for i in range(4):
            colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
            if i != 0:
                start = (int(256-coords_2d[0][0]), int(256-coords_2d[0][1]))
                end = (int(256-coords_2d[i][0]), int(256-coords_2d[i][1]))
                cv2.line(img, start, end,  colours[i-1], 2)
        
        return img


    def run(self):

        while not rospy.is_shutdown():

            img1 = self.cam1_image
            # img2 = self.cam2_image

            if img1 is not None:
                pred_mesh_list = []
                if self.vertices_l is not None:
                    pred_mesh_l = dict(vertices=self.vertices_l, faces=self.faces_l)
                    pred_mesh_list.append(pred_mesh_l)
                if self.vertices_r is not None:
                    pred_mesh_r = dict(vertices=self.vertices_r, faces=self.faces_r)
                    pred_mesh_list.append(pred_mesh_r)
                    # visualize
                    img1 = self.visualizer.visualize(
                        img1, 
                        pred_mesh_list = pred_mesh_list)
                img = np.concatenate([img1], axis=1)
                # ImShow(img, name='full')
            if self.img_crop is not None:
                img_crop = self.drawPose(self.img_crop)
                ImShow(img_crop, name='cropped')

            self.rate.sleep()
        rospy.spin()

  
def main():
    args = DemoOptions().parse()
    args.use_smplx = True

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    # run
    viz_= ImgViz()
    viz_.run()


if __name__ == '__main__':
    main()
