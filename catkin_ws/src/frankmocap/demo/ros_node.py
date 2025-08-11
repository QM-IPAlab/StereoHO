# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys
import numpy as np
import cv2
import torch
import time
import rospy
from scipy.spatial.transform import Rotation as R

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, '../'))

from demo.demo_options import DemoOptions
from handmocap.hand_mocap_api import HandMocap

from rospy.numpy_msg import numpy_msg
from frankmocap.msg import Int16, Float32, MocapOutput

# From IHOI
from ihoi.nnutils.hand_utils import ManopthWrapper
from ihoi.nnutils.handmocap import process_mocap_predictions
from ihoi.nnutils.geom_utils import se3_to_matrix
from ihoi.nnutils import geom_utils
import pickle

# HO detection
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, 'detectors/hand_object_detector/lib/'))
from detectors.hand_object_detector import _init_paths
from detectors.hand_object_detector.demo import _get_image_blob
from detectors.hand_object_detector.lib.model.utils.config import cfg, cfg_from_file
from detectors.hand_object_detector.lib.model.rpn.bbox_transform import clip_boxes
from detectors.hand_object_detector.lib.model.rpn.bbox_transform import bbox_transform_inv
from detectors.hand_object_detector.lib.model.roi_layers import nms
from detectors.hand_object_detector.lib.model.utils.net_utils import vis_detections_filtered_objects_PIL
from detectors.hand_object_detector.lib.model.faster_rcnn.resnet import resnet


def getPoseFromRotationTranslation(rmat, tvec):
    C = np.concatenate((rmat, tvec), axis=1)
    return np.concatenate((C, np.array([0,0,0,1]).reshape(1,4)), axis=0)

def getRotationTranslationfromPose(mat):
    rmat = mat[0:3,0:3]
    tvec = mat[0:3,3]
    return rmat, tvec

def quaternion2mat(quat):
    r = R.from_quat(quat)
    try:
        return r.as_matrix()
    except:
        return r.as_dcm()
    
def least_squares_fit(p1_t, p2_t):
    p1 = p1_t.transpose()
    p2 = p2_t.transpose()

    #Calculate centroids
    p1_c = np.mean(p1, axis = 1).reshape((-1,1)) #If you don't put reshape then the outcome is 1D with no rows/colums and is interpeted as rowvector in next minus operation, while it should be a column vector
    p2_c = np.mean(p2, axis = 1).reshape((-1,1))

    #Subtract centroids
    q1 = p1-p1_c
    q2 = p2-p2_c

    #Calculate covariance matrix
    H=np.matmul(q1,q2.transpose())

    #Calculate singular value decomposition (SVD)
    U, X, V_t = np.linalg.svd(H) #the SVD of linalg gives you Vt

    #Calculate rotation matrix
    R = np.matmul(V_t.transpose(),U.transpose())

    assert np.allclose(np.linalg.det(R), 1.0), "Rotation matrix of N-point registration not 1, see paper Arun et al."

    #Calculate translation matrix
    T = p2_c - np.matmul(R,p1_c)
    
    return R, T

class HandMocapRosNode:
    def __init__(self):
        print("Starting Hand Mocap Node")
        self.cam1_image = None
        self.cam2_image = None
        self.cam1_id = 1
        self.cam2_id = 2

        rospy.init_node('frankmocap', anonymous=True, disable_signals=True)
        self.rate = rospy.Rate(30)

        rospy.Subscriber("/camera{}/image_list".format(self.cam1_id), numpy_msg(Int16), self.callback1)
        rospy.Subscriber("/camera{}/image_list".format(self.cam2_id), numpy_msg(Int16), self.callback2)
        self.pub_hand_pose_base = rospy.Publisher('/hand_mocap/hand_pose_base', numpy_msg(Float32), queue_size=10)
        self.pub_mocap_output = rospy.Publisher('/hand_mocap/mocap_output', MocapOutput, queue_size=10)

        # HO detection
        cfg_from_file('detectors/hand_object_detector/cfgs/res101.yml')
        self.pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])
        self.fasterRCNN = resnet(self.pascal_classes, 101, pretrained=False, class_agnostic=False)
        self.fasterRCNN.create_architecture()
        checkpoint = torch.load('detectors/hand_object_detector/models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth')
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
            print(cfg.POOLING_MODE)
        cfg.CUDA = True
        self.fasterRCNN.cuda()
        self.fasterRCNN.eval()

        print('Loaded HO detection model successfully!')

        # Read camera calibration
        calib_path = '../../../calib'
        target2base_f = os.path.join(calib_path, 'robot2', 'C_target2base.pkl')
        cam2gripper_f = os.path.join(calib_path, 'robot2', 'C_cam2gripper.pkl')
        cam3intrinsics = os.path.join(calib_path, 'robot2', 'cam_intrinsics.pkl')
        target2cam1_f = os.path.join(calib_path, 'cam1', 'C_baseline.pkl')
        target2cam2_f = os.path.join(calib_path, 'cam2', 'C_baseline.pkl')

        with open(target2cam1_f, 'rb') as f:
            target2cam1_data = pickle.load(f)
            projMatrix1 = target2cam1_data['extrinsic']['rgb']['projMatrix']
            target2cam1 = np.eye(4)
            target2cam1[:3, :3] = target2cam1_data['extrinsic']['rgb']['rvec']
            target2cam1[:3, 3] = target2cam1_data['extrinsic']['rgb']['tvec'].reshape(-1)
            cam1_2target = np.linalg.inv(target2cam1)

        with open(target2cam2_f, 'rb') as f:
            target2cam2_data = pickle.load(f)
            projMatrix2 = target2cam2_data['extrinsic']['rgb']['projMatrix']
            target2cam2 = np.eye(4)
            target2cam2[:3, :3] = target2cam2_data['extrinsic']['rgb']['rvec']
            target2cam2[:3, 3] = target2cam2_data['extrinsic']['rgb']['tvec'].reshape(-1)
            cam2_2target = np.linalg.inv(target2cam2)

        with open(target2base_f, 'rb') as f:
            self.target2base = pickle.load(f)
        
        with open(cam2gripper_f, 'rb') as f:
            self.cam2gripper = pickle.load(f)
            self.gripper2cam = np.linalg.inv(self.cam2gripper)

        with open(cam3intrinsics, 'rb') as f:
            self.cam3intrinsics = pickle.load(f)

        self.projMatrix1 = projMatrix1
        self.projMatrix2 = projMatrix2
        self.cam1_2target = cam1_2target
        self.cam2_2target = cam2_2target
        self.target2cam1 = target2cam1
        self.target2cam2 = target2cam2
        
        print("Initialized!")

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
    
    def callbackcaminfo1(self, data):
        self.K1 = np.array(data.K).reshape(3,3)

    def callbackcaminfo2(self, data):
        self.K2 = np.array(data.K).reshape(3,3)
        
    def handmocap(self, img, hand_bbox_list, hand_mocap, hand_wrapper, bbox_len=90):

        # Hand Pose Regression
        pred_output_list = hand_mocap.regress(img, hand_bbox_list, add_margin=True)
        if pred_output_list[0] is None:
            print(f"No hand pose detected")
            return False, None, None, None
        if pred_output_list[0]['right_hand'] is None:
            print(f"No hand pose detected")
            return False, None, None, None
        object_mask = np.ones_like(img[..., 0]) * 255
        ihoi_data = process_mocap_predictions(pred_output_list, img, hand_wrapper.cpu(), mask=object_mask, bbox_len=bbox_len)
        cropped_image = ihoi_data['image'].squeeze(0)
        cropped_image = (cropped_image/2.0 + 0.5) * 255.0
        cropped_image = cropped_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        return True, cropped_image, ihoi_data
    
    def pub_mat(self, cropped_img, ihoi_data, root_3d, cam2target):
        cTh = ihoi_data['cTh']
        cTh_mat = se3_to_matrix(cTh)
        cropped_img = cv2.resize(cropped_img, (256, 256))

        cTh_hand_cam = ihoi_data['cTh_mat_orig'].squeeze(0).numpy()
        cTh_hand_target = np.matmul(cam2target, cTh_hand_cam)
        cTh_hand_target[:3, 3] = root_3d

        cTh_hand_base = np.matmul(self.target2base, cTh_hand_target)

        # Get hand segmentation
        verts_2d = ihoi_data['verts_2d']
        # get convex hull
        hull = cv2.convexHull(verts_2d.astype(np.int32))
        hand_mask = np.zeros_like(cropped_img)
        cv2.fillPoly(hand_mask, [hull], (255, 255, 255))
        # threshold
        hand_mask_bin = cv2.threshold(cv2.cvtColor(hand_mask, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]

        return cTh_hand_base, np.array(cropped_img).reshape(-1).astype(np.int16), cTh_mat.numpy().reshape(-1), hand_mask_bin.reshape(-1).astype(np.int16)

    def run_hand_mocap(self, args, bbox_detector, hand_mocap, visualizer):

        hand_wrapper = ManopthWrapper(mano_path="extra_data/mano/models").to('cuda')
        hand_wrapper_vis = ManopthWrapper(mano_path="extra_data/mano/models", center_idx=0).to('cuda')
        while not rospy.is_shutdown():
            with torch.no_grad():
                print("--------------------------------------")
                if self.cam1_image is None or self.cam2_image is None:
                    print("No image")
                    continue

                img1 = self.cam1_image.copy()
                img2 = self.cam2_image.copy()
                offset1 = 0
                offset2 = 100
                img_scale = 1

                im_data = torch.FloatTensor(1).cuda()
                im_info = torch.FloatTensor(1).cuda()
                num_boxes = torch.LongTensor(1).cuda()
                gt_boxes = torch.FloatTensor(1).cuda()
                box_info = torch.FloatTensor(1) 
                img_show = cv2.hconcat([img1[:,offset1:offset1+480,:], img2[:,offset2:offset2+480,:]])
                img = cv2.resize(img_show, (int(img_show.shape[1]/img_scale), int(img_show.shape[0]/img_scale)))

                blobs, im_scales = _get_image_blob(img)
                assert len(im_scales) == 1, "Only single-image batch implemented"
                im_blob = blobs
                im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

                im_data_pt = torch.from_numpy(im_blob)
                im_data_pt = im_data_pt.permute(0, 3, 1, 2).cuda()
                im_info_pt = torch.from_numpy(im_info_np).cuda()

                im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                gt_boxes.resize_(1, 1, 5).zero_()
                num_boxes.resize_(1).zero_()
                box_info.resize_(1, 1, 5).zero_() 


                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label, loss_list = self.fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info, cfg) 

                scores = cls_prob.data
                boxes = rois.data[:, :, 1:5]

                # extact predicted params
                contact_vector = loss_list[0][0] # hand contact state info
                offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
                lr_vector = loss_list[2][0].detach() # hand side info (left/right)

                # get hand contact 
                _, contact_indices = torch.max(contact_vector, 2)
                contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

                # get hand side 
                lr = torch.sigmoid(lr_vector) > 0.5
                lr = lr.squeeze(0).float()
                pred_boxes = torch.tile(boxes, (1, scores.shape[1]))

                pred_boxes /= im_scales[0]

                scores = scores.squeeze()
                pred_boxes = pred_boxes.squeeze()
                obj_dets, hand_dets = None, None
                thresh_hand, thresh_obj = 0.4, 0.4
                for j in range(1, len(self.pascal_classes)):
                    if self.pascal_classes[j] == 'hand':
                        inds = torch.nonzero(scores[:,j]>thresh_hand).view(-1)
                    elif self.pascal_classes[j] == 'targetobject':
                        inds = torch.nonzero(scores[:,j]>thresh_obj).view(-1)

                    if inds.numel() > 0:
                        cls_scores = scores[:,j][inds]
                        _, order = torch.sort(cls_scores, 0, True)
                        if args.class_agnostic:
                            cls_boxes = pred_boxes[inds, :]
                        else:
                            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                    
                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
                        cls_dets = cls_dets[order]
                        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                        cls_dets = cls_dets[keep.view(-1).long()]
                        if self.pascal_classes[j] == 'targetobject':
                            obj_dets = cls_dets.cpu().numpy()
                        if self.pascal_classes[j] == 'hand':
                            hand_dets = cls_dets.cpu().numpy()
                    

                img, img1_det, img2_det = vis_detections_filtered_objects_PIL(img, obj_dets, hand_dets, thresh_hand, thresh_obj)

                if type(img1_det['hand']) == type(None) or type(img2_det['hand']) == type(None):
                    print(f"No hand deteced")
                    mocap_output_msg = MocapOutput()
                    mocap_output_msg.cropped_img_1 = np.zeros(256*256*3).astype(np.int16)
                    mocap_output_msg.cropped_img_2 = np.zeros(256*256*3).astype(np.int16)
                    mocap_output_msg.hand_pose_cam_1 = np.zeros(16).astype(np.float32)
                    mocap_output_msg.hand_pose_cam_2 = np.zeros(16).astype(np.float32)
                    mocap_output_msg.hand_pose_base = np.zeros(16).astype(np.float32)
                    mocap_output_msg.hand_mask_1 = np.zeros(256*256).astype(np.int16)
                    mocap_output_msg.hand_mask_2 = np.zeros(256*256).astype(np.int16)
                    mocap_output_msg.bbox_cam_1 = np.zeros(4).astype(np.float32)
                    mocap_output_msg.bbox_cam_2 = np.zeros(4).astype(np.float32)
                    mocap_output_msg.obj_bbox_cam_1 = np.zeros(4).astype(np.float32)
                    mocap_output_msg.obj_bbox_cam_2 = np.zeros(4).astype(np.float32)
                    self.pub_mocap_output.publish(mocap_output_msg)
                    continue

                img1_det['hand'] = img1_det['hand'] * img_scale
                img2_det['hand'] = img2_det['hand'] * img_scale
                hand_bbox_list_cam1 = []
                hand_bbox_list_cam2 = []
                bbox = {}
                bbox['left_hand'] = None
                bbox['right_hand'] = np.array([offset1+img1_det['hand'][0], img1_det['hand'][1],
                                            img1_det['hand'][2]-img1_det['hand'][0],
                                            img1_det['hand'][3]-img1_det['hand'][1]])
                hand_bbox_list_cam1.append(bbox)
                bbox = {}
                bbox['left_hand'] = None
                bbox['right_hand'] = np.array([offset2+img2_det['hand'][0], img2_det['hand'][1],
                                                img2_det['hand'][2]-img2_det['hand'][0],
                                                img2_det['hand'][3]-img2_det['hand'][1]])
                hand_bbox_list_cam2.append(bbox)

                # Hand mocap
                ret1, cropped_img1, ihoi_data_1 = self.handmocap(img1, hand_bbox_list_cam1, hand_mocap, hand_wrapper, bbox_len=70)
                ret2, cropped_img2, ihoi_data_2 = self.handmocap(img2, hand_bbox_list_cam2, hand_mocap, hand_wrapper, bbox_len=70)

                if not ret1 or not ret2:
                    mocap_output_msg = MocapOutput()
                    mocap_output_msg.cropped_img_1 = np.zeros(256*256*3).astype(np.int16)
                    mocap_output_msg.cropped_img_2 = np.zeros(256*256*3).astype(np.int16)
                    mocap_output_msg.hand_pose_cam_1 = np.zeros(16).astype(np.float32)
                    mocap_output_msg.hand_pose_cam_2 = np.zeros(16).astype(np.float32)
                    mocap_output_msg.hand_pose_base = np.zeros(16).astype(np.float32)
                    mocap_output_msg.hand_mask_1 = np.zeros(256*256).astype(np.int16)
                    mocap_output_msg.hand_mask_2 = np.zeros(256*256).astype(np.int16)
                    mocap_output_msg.bbox_cam_1 = np.zeros(4).astype(np.float32)
                    mocap_output_msg.bbox_cam_2 = np.zeros(4).astype(np.float32)
                    mocap_output_msg.obj_bbox_cam_1 = np.zeros(4).astype(np.float32)
                    mocap_output_msg.obj_bbox_cam_2 = np.zeros(4).astype(np.float32)
                    self.pub_mocap_output.publish(mocap_output_msg)
                    continue
                else:
                    # show cropped image
                    print("Hand detected")
                
                # Calculate object bbox
                if type(img1_det['object']) == type(None) or type(img2_det['object']) == type(None):
                    print(f"No object deteced")
                    cam1_obj_bbox = np.zeros(4)
                    cam2_obj_bbox = np.zeros(4)
                else:
                    img1_det['object'] = img1_det['object'] * img_scale
                    img2_det['object'] = img2_det['object'] * img_scale
                    bbox_1_size = ihoi_data_1['hoi_bbox'][2]-ihoi_data_1['hoi_bbox'][0]
                    bbox_2_size = ihoi_data_2['hoi_bbox'][2]-ihoi_data_2['hoi_bbox'][0]
                    cam_offset1 = np.array([offset1, 0, offset1, 0])
                    cam_offset2 = np.array([offset2, 0, offset2, 0])
                    cam1_obj_bbox = 255*((cam_offset1 + img1_det['object'] - np.tile(ihoi_data_1['hoi_bbox'][:2],2))/bbox_1_size)
                    cam2_obj_bbox = 255*((cam_offset2 + img2_det['object'] - np.tile(ihoi_data_2['hoi_bbox'][:2],2))/bbox_2_size)

                # enlarge bbox
                margin = np.array([-5, -5, 5, 5])
                cam1_obj_bbox = cam1_obj_bbox + margin
                cam2_obj_bbox = cam2_obj_bbox + margin

                # bound between 0 and 255
                cam1_obj_bbox = np.clip(cam1_obj_bbox, 0, 255)
                cam2_obj_bbox = np.clip(cam2_obj_bbox, 0, 255)

                eye = torch.from_numpy(np.eye(4)).float().unsqueeze(0)
                glb_se3 = geom_utils.matrix_to_se3(eye)
                _, mano_joints = hand_wrapper_vis(glb_se3.cuda(), (ihoi_data_2['mano_pose'][...,3:] + hand_wrapper.hand_mean.cpu()).cuda())

                root_3d = cv2.triangulatePoints(self.projMatrix1, self.projMatrix2, ihoi_data_1['root_2d'][:2]*2, ihoi_data_2['root_2d'][:2]*2)
                root_3d = root_3d.reshape(4)
                root_3d = root_3d[:3] / root_3d[3]

                cTh_hand_base1, cropped_img1_pub, hand_pose_cam_1_pub, hand_mask_1_pub = self.pub_mat(cropped_img1, ihoi_data_1, root_3d, self.cam1_2target)
                cTh_hand_base2, cropped_img2_pub, hand_pose_cam_2_pub, hand_mask_2_pub = self.pub_mat(cropped_img2, ihoi_data_2, root_3d, self.cam2_2target)

                mocap_output_msg = MocapOutput()
                mocap_output_msg.cropped_img_1 = list(cropped_img1_pub)
                mocap_output_msg.cropped_img_2 = list(cropped_img2_pub)
                mocap_output_msg.hand_pose_cam_1 = list(hand_pose_cam_1_pub)
                mocap_output_msg.hand_pose_cam_2 = list(hand_pose_cam_2_pub)
                mocap_output_msg.hand_mask_1 = list(hand_mask_1_pub)
                mocap_output_msg.hand_mask_2 = list(hand_mask_2_pub)
                mocap_output_msg.bbox_cam_1 = list(ihoi_data_1['hoi_bbox'])
                mocap_output_msg.bbox_cam_2 = list(ihoi_data_2['hoi_bbox'])
                mocap_output_msg.obj_bbox_cam_1 = list(cam1_obj_bbox)
                mocap_output_msg.obj_bbox_cam_2 = list(cam2_obj_bbox)
                
                # Triangulate 2D joints
                joints_3d = cv2.triangulatePoints(self.projMatrix1, self.projMatrix2, ihoi_data_1['joints_2d'][:,:2].T*2, ihoi_data_2['joints_2d'][:,:2].T*2)
                joints_3d = joints_3d[:3] / joints_3d[3]
                joints_3d = joints_3d.T

                cTh_hand_target = np.matmul(np.linalg.inv(self.target2base), cTh_hand_base1)
                mano_target = np.concatenate((mano_joints.squeeze(0).cpu().numpy(), np.ones((mano_joints.shape[1], 1))), axis=1)
                mano_target = np.matmul(cTh_hand_target, mano_target.T).T
                mano_target = mano_target[:, :3]
                root_diff = joints_3d[0] - mano_target[0]
                # Compute transformation between mano_pcd and obman_pcd with known correspondences
                try:
                    R, T = least_squares_fit(mano_target.astype(np.float32), joints_3d-root_diff.reshape(1,3).astype(np.float32))
                except:
                    print("Error in least squares fit")
                    continue
                T = T+root_diff.reshape(3,1)
                C = np.concatenate((R, T), axis=1)
                Trans = np.concatenate((C, np.array([0,0,0,1]).reshape(1,4)), axis=0)

                cTh_hand_base = np.matmul(self.target2base, np.matmul(Trans, cTh_hand_target))

                # Publish
                self.pub_hand_pose_base.publish(cTh_hand_base.reshape(-1).astype(np.float32))
                mocap_output_msg.hand_pose_base = list(cTh_hand_base.reshape(-1).astype(np.float32))
                self.pub_mocap_output.publish(mocap_output_msg)

        rospy.spin()
        cv2.destroyAllWindows()

  
def main():
    args = DemoOptions().parse()
    args.use_smplx = True
    args.class_agnostic = False

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    bbox_detector = None
    hand_mocap = HandMocap(args.checkpoint_hand, args.smpl_dir, device = device)
    visualizer = None

    hand_mocap_ros_node = HandMocapRosNode()
    hand_mocap_ros_node.run_hand_mocap(args, bbox_detector, hand_mocap, visualizer)
   

if __name__ == '__main__':
    main()
