#!/usr/bin/env python2
import cv2
import rospy
import numpy as np
import copy
import shutil
import os
from tqdm import tqdm

from rospy.numpy_msg import numpy_msg
from main.msg import Int16, Float32
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--no_save', action='store_true', help='Do not save the images')
args = parser.parse_args()

def to_cloud_msg(points, colors=None, frame=None, stamp=None):
    """Convert list of unstructured points to a PointCloud2 message.

    Args:
        points: Point coordinates as array of shape (N,3).
        colors: Colors as array of shape (N,3).
        frame
        stamp
    """
    msg = PointCloud2()
    msg.header.frame_id = frame
    msg.header.stamp = stamp or rospy.Time.now()

    msg.height = 1
    msg.width = points.shape[0]
    msg.is_bigendian = False
    msg.is_dense = False

    msg.fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]
    msg.point_step = 12
    data = points

    if colors is not None:
        msg.fields += [
            PointField("r", 12, PointField.FLOAT32, 1),
            PointField("g", 16, PointField.FLOAT32, 1),
            PointField("b", 20, PointField.FLOAT32, 1),
        ]
        msg.point_step += 12
        data = np.hstack([points, colors])

    msg.row_step = msg.point_step * points.shape[0]
    msg.data = data.astype(np.float32).tostring()

    return msg

class HandObjectReconRosNode:
    def __init__(self):
        self.idx = 0
        self.img_crop_1 = None
        self.img_crop_orig_1 = None
        self.img_crop_2 = None
        self.img_crop_orig_2 = None
        self.seg_output = None
        self.hand_pose_1 = None
        self.hand_pose_2 = None
        self.cam1_image = None
        self.cam2_image = None
        self.debug_image = np.zeros((512, 512, 3))
        self.pc_h = None
        self.pc_o = None
        self.pc_h_received = False
        self.pc_o_received = False
        self.hand_pose_received = False
        self.all_grasps_received = False
        self.robot_mode = 'init'
        self.subject = '1'
        self.config = '1'
        
        self.img_size = 256
        rospy.init_node('show', anonymous=True, disable_signals=True)
        self.rate = rospy.Rate(30)


        rospy.Subscriber("/camera1/image_list", numpy_msg(Int16), self.callback1)
        rospy.Subscriber("/camera2/image_list", numpy_msg(Int16), self.callback2)
        rospy.Subscriber('/hand_mocap/debug_image', numpy_msg(Int16), self.calldebug)
        rospy.Subscriber('/hand_mocap/pc_h', numpy_msg(Float32), self.callbackPCH)
        rospy.Subscriber('/hand_mocap/pc_o', numpy_msg(Float32), self.callbackPCO)
        rospy.Subscriber('/hand_mocap/hand_pose_base', numpy_msg(Float32), self.callbackHandPose)
        rospy.Subscriber('/recording/robotMode', String, self.callbackRobotMode)
        rospy.Subscriber('/recording/subject_id', String, self.callbackSubject)
        rospy.Subscriber('/recording/config_id', String, self.callbackConfig)
        rospy.Subscriber('/recording/handover_time', String, self.callbackHandoverTime)

        self.pub_pc_h_vis = rospy.Publisher('/hand_mocap/pc_h_vis', PointCloud2, queue_size=1, latch=True)
        self.pub_pc_o_vis = rospy.Publisher('/hand_mocap/pc_o_vis', PointCloud2, queue_size=1, latch=True)
        self.pub_hand_pose = rospy.Publisher("/hand_mocap/hand_pose_base_repeat", numpy_msg(Float32), queue_size=1, latch=True)


    def callback1(self, data):
        bgr = np.reshape(data.data, (720,1280,3))
        bgr = bgr.astype(np.uint8)
        h, w, c = bgr.shape

        self.cam1_image = cv2.resize(bgr, (640,360))
    
    def callback2(self, data):
        bgr = np.reshape(data.data, (720,1280,3))
        bgr = bgr.astype(np.uint8)
        h, w, c = bgr.shape

        self.cam2_image = cv2.resize(bgr, (640,360))
    
    def calldebug(self, data):
        bgr = np.reshape(data.data, (512, 512,3))
        bgr = bgr.astype(np.uint8)
        h, w, c = bgr.shape

        self.debug_image = cv2.resize(bgr, (512,512))
    
    def callbackPCH(self, data):
        self.pc_h = np.reshape(data.data, (-1,3)).astype(np.float32)
        self.pc_h_received = True

    def callbackPCO(self, data):
        self.pc_o = np.reshape(data.data, (-1,3)).astype(np.float32)
        self.pc_o_received = True

    def callbackHandPose(self, data):
        self.hand_pose_data = data
        self.hand_pose = np.reshape(data.data, (4,4))
        self.hand_pose_received = True
    
    def callbackRobotMode(self, data):
        self.robot_mode = str(data.data)

    def callbackSubject(self, data):
        self.subject = str(data.data)
    
    def callbackConfig(self, data):
        self.config = str(data.data)
    
    def callbackHandoverTime(self, data):
        self.handover_time = str(data.data)


    def run(self):
        rate = rospy.Rate(15)
        t=1
        raw_cam1 = []
        raw_cam2 = []
        while not rospy.is_shutdown():

            if self.cam1_image is not None and self.cam2_image is not None:
                t+=1

                cam1_img_full = copy.deepcopy(self.cam1_image).astype(np.uint8)
                cam2_img_full = copy.deepcopy(self.cam2_image).astype(np.uint8)
                debug_img = copy.deepcopy(self.debug_image).astype(np.uint8)
                
                cam1_img_full = cv2.resize(cam1_img_full, (640, 360))
                cam2_img_full = cv2.resize(cam2_img_full, (640, 360))
                show_img = debug_img

                if self.hand_pose_received:
                    if self.pc_h_received and self.pc_o_received:
                        hand_pose = copy.deepcopy(self.hand_pose)
                        pc_h = copy.deepcopy(self.pc_h)
                        pc_h = np.matmul(hand_pose, np.concatenate((pc_h, np.ones((pc_h.shape[0],1))), axis=1).T).T[:,:3]
                        pc_o = copy.deepcopy(self.pc_o)
                        pc_o = np.matmul(hand_pose, np.concatenate((pc_o, np.ones((pc_o.shape[0],1))), axis=1).T).T[:,:3]
                        pc_h_msg = to_cloud_msg(pc_h, colors=np.repeat(np.array([[0,255,0]]), pc_h.shape[0], axis=0), frame='base')
                        pc_o_msg = to_cloud_msg(pc_o, colors=np.repeat(np.array([[0,0,255]]), pc_o.shape[0], axis=0), frame='base')
                        self.pub_pc_h_vis.publish(pc_h_msg)
                        self.pub_pc_o_vis.publish(pc_o_msg)
                    self.pub_hand_pose.publish(self.hand_pose_data)

                cv2.imshow('HOrecon', show_img)
                cv2.waitKey(1)
                rate.sleep()

                if not args.no_save:
                    if self.robot_mode == 'stop_recording':
                        save_dir = os.path.join('/home/robot_tutorial/Downloads/HANDOVER_RAL', 'subject_{}'.format(self.subject), 'recordings', 'raw', 'config_{}'.format(self.config))
                        cam1_dir = os.path.join(save_dir, 'cam1', 'rgb')
                        cam2_dir = os.path.join(save_dir, 'cam2', 'rgb')
                        print("Handover time: {:.03f}s".format(float(self.handover_time)))
                        print("Saving to: ", save_dir)
                        if os.path.exists(cam1_dir):
                            shutil.rmtree(cam1_dir)
                            shutil.rmtree(cam2_dir)
                        if not os.path.exists(cam1_dir):
                            os.makedirs(cam1_dir, exist_ok=True)
                            os.makedirs(cam2_dir, exist_ok=True)

                        for fr in tqdm(range(len(raw_cam1))):
                            cv2.imwrite(os.path.join(cam1_dir, '{:05d}.jpg'.format(fr)), raw_cam1[fr])
                            cv2.imwrite(os.path.join(cam2_dir, '{:05d}.jpg'.format(fr)), raw_cam2[fr])
                        
                        raw_cam1 = []
                        raw_cam2 = []
                        self.robot_mode = 'idle'
                    
                    elif self.robot_mode == 'start_recording':
                        raw_cam1 = []
                        raw_cam2 = []
                    
                    else:
                        raw_cam1.append(cam1_img_full)
                        raw_cam2.append(cam2_img_full)
                
        rospy.spin()

if __name__ == '__main__':
    node = HandObjectReconRosNode()
    node.run()