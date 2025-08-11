#!/usr/bin/env python2
import cv2
import time
import rospy
import numpy as np
import torch
import torch.nn.functional as F
import copy
import shutil
import os
import sys
import pickle

from rospy.numpy_msg import numpy_msg
from frankmocap.msg import Int16, Float32, MocapOutput
from graspnet.msg import Grasps
from graspnet.srv import EstimateGrasps
import std_msgs.msg

from models.resnet2vq_model_lightning import ResNet2VQModelLightning
from options.train_options import TrainOptions
from termcolor import cprint

from pytorch3d.transforms import Transform3d
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.io import IO
import torchvision.transforms as transforms

# FastSAM
current_dir = os.path.dirname(os.path.realpath(__file__))
fastsam_dir = os.path.join(current_dir, '../../FastSAM')
sys.path.append(fastsam_dir)
from fastsam import FastSAM, FastSAMPrompt

    
def compute2Dcentroid(_rgb, _objmask):
    img_gray = cv2.cvtColor(_rgb, cv2.COLOR_BGR2GRAY)

    objmask = copy.deepcopy(_objmask)

    img_mask = img_gray
    img_mask[objmask == 0] = 0

    # Intensity centroid
    m00 = m01 = m10 = 0

    x = np.array([a for a in range(0,img_mask.shape[1])])
    y = np.array([a for a in range(0,img_mask.shape[0])])

    m00 = np.sum(img_mask)
    m01 = np.sum(np.matmul(y, img_mask))
    m10 = np.sum(np.matmul(x, img_mask.transpose()))

    try:
        centroid = np.round(np.array((m10/m00, m01/m00)))
        return centroid
    except:
        print('Centroid not found')
        return None

def compute3Dcentroid2views(c1, c2, point1, point2):

    if (point1.dtype != 'float64'):
        point1 = point1.astype(np.float64)

    if (point2.dtype != 'float64'):
        point2 = point2.astype(np.float64)

    point3d = cv2.triangulatePoints(c1['extrinsic']['rgb']['projMatrix'], c2['extrinsic']['rgb']['projMatrix'], point1.reshape(2,1), point2.reshape(2,1)).transpose()
    for point in point3d:
        point /= point[-1]

    ### Re-projection error verification
    imagePoints1, _ = cv2.projectPoints(point3d.reshape(-1)[:3], c1['extrinsic']['rgb']['rvec'], c1['extrinsic']['rgb']['tvec'], c1['intrinsic']['rgb'], np.array([0.,0.,0.,0.,0.]))
    imagePoints2, _ = cv2.projectPoints(point3d.reshape(-1)[:3], c2['extrinsic']['rgb']['rvec'], c2['extrinsic']['rgb']['tvec'], c2['intrinsic']['rgb'], np.array([0.,0.,0.,0.,0.]))

    reperr = np.sqrt((cv2.norm(point1, imagePoints1.squeeze().astype(np.float64)) + cv2.norm(point2, imagePoints2.squeeze().astype(np.float64))) / 2)

    if reperr > 5 or np.isnan(point3d).any():
        return None, 999.9, None, None
    else:
        return point3d.reshape(-1)[:3], reperr, imagePoints1.squeeze().astype(np.float64), imagePoints2.squeeze().astype(np.float64)

def computeIOU(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

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
        self.obj_bbox_cam_1 = np.zeros(4)
        self.obj_bbox_cam_2 = np.zeros(4)

        
        self.img_size = 256
        cam_f = np.array([10., 10.])
        cam_p = np.array([0., 0.])
        self.cam_f = torch.from_numpy(cam_f).float().unsqueeze(0)
        self.cam_p = torch.from_numpy(cam_p).float().unsqueeze(0)
        self.camera = PerspectiveCameras(self.cam_f, self.cam_p, image_size=(self.img_size, self.img_size)).to('cpu')
        # Points for coordinate system
        coords = np.array([[0, 0, 0],
                            [0.1, 0, 0],
                            [0, 0.1, 0],
                            [0, 0, 0.1]])
        self.coords = torch.from_numpy(coords).float().unsqueeze(0)

        rospy.init_node('HOrecon', anonymous=True, disable_signals=True)
        self.rate = rospy.Rate(30)

        rospy.Subscriber('/hand_mocap/mocap_output', MocapOutput, self.callbackMocapOutput)
        rospy.Subscriber("/camera2/image_list", numpy_msg(Int16), self.callback2rgb)

        self.pub_tsdf_h = rospy.Publisher('/hand_mocap/tsdf_h', numpy_msg(Float32), queue_size=10)
        self.pub_tsdf_o = rospy.Publisher('/hand_mocap/tsdf_o', numpy_msg(Float32), queue_size=10)
        self.pub_pc_h = rospy.Publisher('/hand_mocap/pc_h', numpy_msg(Float32), queue_size=10)
        self.pub_pc_o = rospy.Publisher('/hand_mocap/pc_o', numpy_msg(Float32), queue_size=10)
        self.pub_hand_pose_view = rospy.Publisher('/hand_mocap/hand_pose_view', std_msgs.msg.Int16, queue_size=10)
        self.pub_debug_image = rospy.Publisher('/hand_mocap/debug_image', numpy_msg(Int16), queue_size=10)
        self.pub_all_grasps = rospy.Publisher('/handover/all_grasps', Grasps, queue_size=10)

        # Load models
        opt = TrainOptions().parse()
        opt.vq_dset = 'obman'
        opt.ho_mode = 'joint'
        opt.mlp_decoder = 1
        opt.decoder_type = 'MLP'
        opt.trunc_thres=0.02
        
        opt.vq_model='pvqvae'
        opt.vq_cfg='configs/pvqvae_obman.yaml'
        opt.tf_cfg='configs/rand_tf_obman_code_joint.yaml'
        
        opt.vq_ckpt_h='saved_ckpt/vqvae_h.ckpt'
        opt.vq_ckpt_o='saved_ckpt/vqvae_h.ckpt'
        opt.tf_ckpt='saved_ckpt/rand_tf'
        opt.resnet2vq_ckpt = 'saved_ckpt/resnet2vq'
        
        self.model_img2vq = ResNet2VQModelLightning.load_from_checkpoint(opt.resnet2vq_ckpt, opt=opt).cuda()
        self.model_img2vq.eval()

        self.to_tensor = transforms.ToTensor()
        self.to_PIL = transforms.ToPILImage()
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.transforms_norm = transforms.Normalize(mean, std)
        self.transforms_norm_seg = transforms.Normalize([0.5], [0.5])
        self.use_obj_seg = True
        cprint(f'[*] Image to vq model initialized.', 'cyan')
        
        # FastSAM
        self.fastsam = FastSAM(os.path.join(fastsam_dir, 'weights/FastSAM-x.pt'))
        
        cprint(f'[*] Object segmentation model initialized.', 'cyan')

        # 6Dof-graspnet
        rospy.wait_for_service('grasp_estimation')
        self.estimate_grasps_srv = rospy.ServiceProxy('grasp_estimation', EstimateGrasps)

        calib_path = '../../../calib'
        target2base_f = os.path.join(calib_path, 'robot2', 'C_target2base.pkl')
        target2cam1_f = os.path.join(calib_path, 'cam1', 'C_baseline.pkl')
        target2cam2_f = os.path.join(calib_path, 'cam2', 'C_baseline.pkl')
        with open(target2cam1_f, 'rb') as f:
            self.target2cam1_data = pickle.load(f)
            self.target2cam1 = np.eye(4)
            self.target2cam1[:3, :3] = self.target2cam1_data['extrinsic']['rgb']['rvec']
            self.target2cam1[:3, 3] = self.target2cam1_data['extrinsic']['rgb']['tvec'].reshape(-1)
            self.cam1_intrinsic = self.target2cam1_data['intrinsic']['rgb']

        with open(target2cam2_f, 'rb') as f:
            self.target2cam2_data = pickle.load(f)
            self.target2cam2 = np.eye(4)
            self.target2cam2[:3, :3] = self.target2cam2_data['extrinsic']['rgb']['rvec']
            self.target2cam2[:3, 3] = self.target2cam2_data['extrinsic']['rgb']['tvec'].reshape(-1)
            self.cam2_intrinsic = self.target2cam2_data['intrinsic']['rgb']

        with open(target2base_f, 'rb') as f:
            self.target2base = pickle.load(f)



    def callback2rgb(self, data):
        bgr = np.reshape(data.data, (720,1280,3))
        bgr = bgr.astype(np.uint8)
        h, w, c = bgr.shape
        self.cam2_image = bgr

    def callbackMocapOutput(self, data):
        cropped_image_1 = np.array(data.cropped_img_1).reshape(256,256,3)
        cropped_image_2 = np.array(data.cropped_img_2).reshape(256,256,3)
        self.img_crop_orig_1 = cropped_image_1.astype(np.uint8)
        self.img_crop_1 = torch.from_numpy(cropped_image_1).float().unsqueeze(0).permute(0,3,1,2)
        self.img_crop_orig_2 = cropped_image_2.astype(np.uint8)
        self.img_crop_2 = torch.from_numpy(cropped_image_2).float().unsqueeze(0).permute(0,3,1,2)

        self.hand_pose_1 = torch.from_numpy(np.array(data.hand_pose_cam_1).reshape(1,4,4)).float()
        self.hand_pose_2 = torch.from_numpy(np.array(data.hand_pose_cam_2).reshape(1,4,4)).float()
        self.hand_pose_base = torch.from_numpy(np.array(data.hand_pose_base).reshape(1,4,4)).float()

        self.hand_mask_1 = np.array(data.hand_mask_1).reshape(256,256)
        self.hand_mask_2 = np.array(data.hand_mask_2).reshape(256,256)

        self.bbox_cam_1 = np.array(data.bbox_cam_1)
        self.bbox_cam_2 = np.array(data.bbox_cam_2)

        self.obj_bbox_cam_1 = np.array(data.obj_bbox_cam_1)
        self.obj_bbox_cam_2 = np.array(data.obj_bbox_cam_2)

    def drawPose(self, img, hand_pose):
        img = img.copy()
        # Construct transform
        trans = Transform3d(matrix=hand_pose.transpose(1,2))
        norm_coords = trans.transform_points(self.coords)

        coords_2d = self.camera.transform_points_screen(norm_coords)
        coords_2d = coords_2d[0][:,:2].numpy()

        # Draw coords on cropped image
        for i in range(4):
            colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
            if i != 0:
                start = (int(256-coords_2d[0][0]), int(256-coords_2d[0][1]))
                end = (int(256-coords_2d[i][0]), int(256-coords_2d[i][1]))
                cv2.line(img, start, end,  colours[i-1], 2)
        
        return img

    def drawDebug(self, img, proj_2d, proj_axis):
        img1 = img.copy()
        img2 = img.copy()
        # Draw proj 2d points
        proj_2d = proj_2d[0].cpu().numpy()
        for i in range(proj_2d.shape[0]):
            x, y = proj_2d[i]
            cv2.circle(img1, (int(x), int(y)), 2, (0, 0, 255), -1)
            
        # Draw proj axis
        proj_axis = proj_axis[0].cpu().numpy()
        pairs = [[0,1], [0,2], [0,3]]
        colors = [(0,0,255), (0,255,0), (255,0,0)]
        for i, pair in enumerate(pairs):
            x1, y1 = proj_axis[pair[0]]
            x2, y2 = proj_axis[pair[1]]
            cv2.line(img2, (int(x1), int(y1)), (int(x2), int(y2)), colors[i], 2)
        
        return img2

    def process_object_seg(self, seg_img):
        seg_img_vis = np.zeros((seg_img.shape[0], seg_img.shape[1], 3))
        binary_mask = np.zeros((seg_img.shape[0], seg_img.shape[1], 3)).astype(np.float32)
        seg_img_vis[seg_img == 1] = [0, 255, 0]
        binary_mask[seg_img == 1] = [1, 1, 1]

        return seg_img_vis, binary_mask, np.sum(seg_img == 1)
    
    def process_hand_seg_EgoHOS(self, seg_img, cam, seg_img_vis=None, binary_mask=None):
        hand_seg_path = '/home/robot_tutorial/vgn_ws/src/autosdf_j2/demo/pred_twohands/tmp{}.png'.format(cam)
        hand_seg = cv2.imread(hand_seg_path, cv2.IMREAD_GRAYSCALE)
        seg_img_vis[hand_seg == 1] = [0, 0, 255]
        seg_img_vis[hand_seg == 2] = [255, 0, 0]
        binary_mask[hand_seg == 1] = [1, 1, 1]
        binary_mask[hand_seg == 2] = [1, 1, 1]

        return seg_img_vis, binary_mask, np.sum(seg_img == 1)
    
    def process_hand_seg_MaskRCNN(self, hand_seg, seg_img_vis=None, binary_mask=None):
        seg_img_vis[hand_seg == 255] = [0, 0, 255]
        ho_binary_mask = copy.deepcopy(binary_mask)
        ho_binary_mask[hand_seg == 255] = [1, 1, 1]
        hand_binary_mask = np.zeros((ho_binary_mask.shape[0], ho_binary_mask.shape[1], 3)).astype(np.float32)
        hand_binary_mask[hand_seg == 255] = [1, 1, 1]

        return seg_img_vis, ho_binary_mask, hand_binary_mask, np.sum(hand_seg == 255)


    def run(self):
        save_mesh = False
        save_vis = False
        draw_vis = True
        
        IOU_sum_best = 0.0
        pc_h_best = np.zeros((1,3))
        pc_o_best = np.zeros((1,3))
        debug_img = np.zeros((512, 1024, 3))
        pc_h = np.zeros((1,3))
        pc_o = np.zeros((1,3))
        rate = rospy.Rate(0.5)
        not_found_cnt = 0
        while not rospy.is_shutdown():
            fn = "{:04d}".format(self.idx)
            rate.sleep()

            if self.img_crop_1 is not None and self.img_crop_2 is not None and self.hand_pose_base is not None:
                print("=========================================")
                img_crop_1 = copy.deepcopy(self.img_crop_1)
                img_crop_orig_1 = copy.deepcopy(self.img_crop_orig_1)

                img_crop_2 = copy.deepcopy(self.img_crop_2)
                img_crop_orig_2 = copy.deepcopy(self.img_crop_orig_2)

                hand_pose_base = copy.deepcopy(self.hand_pose_base)
                bbox_cam_1 = copy.deepcopy(self.bbox_cam_1)
                bbox_cam_2 = copy.deepcopy(self.bbox_cam_2)
                hand_mask_1 = copy.deepcopy(self.hand_mask_1)
                hand_mask_2 = copy.deepcopy(self.hand_mask_2)
                obj_bbox_cam_1 = copy.deepcopy(self.obj_bbox_cam_1)
                obj_bbox_cam_2 = copy.deepcopy(self.obj_bbox_cam_2)

                hand_pose_target = np.matmul(np.linalg.inv(self.target2base), hand_pose_base.cpu().numpy().squeeze())
                cTh_hand_cam1 = np.matmul(self.target2cam1, hand_pose_target)
                cTh_hand_cam2 = np.matmul(self.target2cam2, hand_pose_target)
                                
                # Forward pass
                with torch.no_grad():
                    if True:

                        if np.sum(pc_h_best) != 0 and np.sum(pc_o_best) != 0:
                            resp = self.estimate_grasps_srv(list(pc_h_best.reshape(-1).astype(np.float32)), list(pc_o_best.reshape(-1).astype(np.float32)), list(np.eye(4).reshape(-1).astype(np.float32)))
                            grasp_msg = Grasps()
                            grasp_msg.all_grasps = np.array(resp.all_grasps).astype(np.float32)
                            grasp_msg.all_scores = np.array(resp.all_scores).astype(np.float32)
                            self.pub_all_grasps.publish(grasp_msg)


                        # draw bbox
                        if np.sum(obj_bbox_cam_2) == 0 or np.sum(obj_bbox_cam_1) == 0:
                            print("Object not found")
                            not_found_cnt += 1
                            if not_found_cnt > 2:
                                IOU_sum_best = 0.0
                                not_found_cnt = 0
                            continue

                        # FastSAM
                        img_both = cv2.hconcat([img_crop_orig_1,img_crop_orig_2])
                        start_time = time.time()
                        everything_results = self.fastsam(img_both, device='cuda', retina_masks=True, imgsz=256, conf=0.4, iou=0.9)
                        prompt_process = FastSAMPrompt(img_both, everything_results, device='cuda')
                        box_prompt = [list(obj_bbox_cam_1.astype(np.int32)), list(obj_bbox_cam_2.astype(np.int32)+np.array([256,0,256,0]))]

                        try:
                            ann = prompt_process.box_prompt(bboxes=box_prompt)
                        except:
                            print("Object not found")
                            not_found_cnt += 1
                            if not_found_cnt > 2:
                                IOU_sum_best = 0.0
                                not_found_cnt = 0
                            continue

                        if len(ann) < 2:
                            print("Object not found")
                            not_found_cnt += 1
                            if not_found_cnt > 2:
                                IOU_sum_best = 0.0
                                not_found_cnt = 0
                            continue
                        # determine image side
                        ann0 = ann[0]
                        ann1 = ann[1]
                        if np.sum(ann0[:,:256]) > 0:
                            binary_mask_1 = ann0[:,:256]
                            binary_mask_2 = ann1[:,256:]
                        else:
                            binary_mask_1 = ann1[:,:256]
                            binary_mask_2 = ann0[:,256:]

                        # remove anything outside of bounding box
                        mask_1 = np.zeros_like(binary_mask_1)
                        mask_1[int(obj_bbox_cam_1[1]):int(obj_bbox_cam_1[3]), int(obj_bbox_cam_1[0]):int(obj_bbox_cam_1[2])] = 1
                        binary_mask_1 = binary_mask_1 * mask_1
                        mask_2 = np.zeros_like(binary_mask_2)
                        mask_2[int(obj_bbox_cam_2[1]):int(obj_bbox_cam_2[3]), int(obj_bbox_cam_2[0]):int(obj_bbox_cam_2[2])] = 1
                        binary_mask_2 = binary_mask_2 * mask_2

                        binary_mask_1 = np.expand_dims(binary_mask_1, axis=2)
                        binary_mask_2 = np.expand_dims(binary_mask_2, axis=2)
                        
                        # object seg
                        seg_img_vis_1, binary_mask_1, n_obj_1 = self.process_object_seg(binary_mask_1[:,:,0])
                        seg_img_vis_2, binary_mask_2, n_obj_2 = self.process_object_seg(binary_mask_2[:,:,0])

                        # Check reprojection error of centroid
                        centroid_1 = compute2Dcentroid(img_crop_orig_1, binary_mask_1[:,:,0])
                        centroid_2 = compute2Dcentroid(img_crop_orig_2, binary_mask_2[:,:,0])
                        if not(np.isnan(centroid_1).any()) and not(np.isnan(centroid_2).any()):
                            point1 = np.array([centroid_1[0], centroid_1[1]])/256. * (bbox_cam_1[2]-bbox_cam_1[0]) + np.array([bbox_cam_1[0], bbox_cam_1[1]])
                            point2 = np.array([centroid_2[0], centroid_2[1]])/256. * (bbox_cam_2[2]-bbox_cam_2[0]) + np.array([bbox_cam_2[0], bbox_cam_2[1]])
                            point1 = point1*2.0
                            point2 = point2*2.0
                            point3d, reperr, cam1_proj2d, cam2_proj2d = compute3Dcentroid2views(self.target2cam1_data, self.target2cam2_data, point1, point2)
                            if point3d is not None:
                                print("Reprojection error: ", reperr)
                                cam1_proj2d = cam1_proj2d/2.0
                                cam2_proj2d = cam2_proj2d/2.0
                                # if point3d[2] < 0.4:
                                #     print("Object on the table, starting reconstruction...")
                                #     IOU_sum_best = 0.0
                                #     continue
                                object_point3d_base = np.matmul(self.target2base, np.array([point3d[0], point3d[1], point3d[2], 1.0]))
                                if object_point3d_base[1] < 1.0:
                                    print("Handover started")
                                    print("Object 3D point: ", object_point3d_base)
                                    IOU_sum_best = 0.0
                                    continue
                            else:
                                print("Triangulation failed")
                                not_found_cnt += 1
                                if not_found_cnt > 2:
                                    IOU_sum_best = 0.0
                                    not_found_cnt = 0
                                continue
                        else:
                            print("2D Centroid not found")
                            not_found_cnt += 1
                            if not_found_cnt > 2:
                                IOU_sum_best = 0.0
                                not_found_cnt = 0
                            continue

                        not_found_cnt = 0

                        # MaskRCNN
                        hand_seg_vis_1, ho_binary_mask_1, hand_binary_mask_1, n_hand_1 = self.process_hand_seg_MaskRCNN(hand_mask_1, seg_img_vis_1, binary_mask_1)
                        hand_seg_vis_2, ho_binary_mask_2, hand_binary_mask_2, n_hand_2 = self.process_hand_seg_MaskRCNN(hand_mask_2, seg_img_vis_2, binary_mask_2)
                        
                        # stereo recon
                        img_crop_1 = img_crop_1.cuda().squeeze(0)
                        img_crop_2 = img_crop_2.cuda().squeeze(0)

                        img_crop_1 = img_crop_1 * torch.from_numpy(ho_binary_mask_1).float().permute(2,0,1).cuda()
                        img_crop_2 = img_crop_2 * torch.from_numpy(ho_binary_mask_2).float().permute(2,0,1).cuda()

                        img_crop_1 = self.to_PIL(img_crop_1)
                        img_crop_1 = self.to_tensor(img_crop_1)
                        img_crop_1 = self.transforms_norm(img_crop_1)
                        img_crop_1 = img_crop_1.unsqueeze(0).cuda()
                        if self.use_obj_seg:
                            binary_mask_1 = np.expand_dims(binary_mask_1[:,:,0], axis=2)
                            obj_seg_1 = torch.from_numpy(binary_mask_1).float().permute(2,0,1).cuda()
                            obj_seg_1 = (obj_seg_1 * 255.0).to(torch.uint8)
                            obj_seg_1 = self.transforms_norm_seg(self.to_tensor(self.to_PIL(obj_seg_1)))
                            obj_seg_1 = obj_seg_1.unsqueeze(0).cuda()
                            img_crop_1 = torch.cat([img_crop_1, obj_seg_1], dim=1)

                        img_crop_2 = self.to_PIL(img_crop_2)
                        img_crop_2 = self.to_tensor(img_crop_2)
                        img_crop_2 = self.transforms_norm(img_crop_2)
                        img_crop_2 = img_crop_2.unsqueeze(0).cuda()
                        if self.use_obj_seg:
                            binary_mask_2 = np.expand_dims(binary_mask_2[:,:,0], axis=2)
                            obj_seg_2 = torch.from_numpy(binary_mask_2).float().permute(2,0,1).cuda()
                            obj_seg_2 = (obj_seg_2 * 255.0).to(torch.uint8)
                            obj_seg_2 = self.transforms_norm_seg(self.to_tensor(self.to_PIL(obj_seg_2)))
                            obj_seg_2 = obj_seg_2.unsqueeze(0).cuda()
                            img_crop_2 = torch.cat([img_crop_2, obj_seg_2], dim=1)

                        start_time = time.time()
                        img_cond_cam1, proj_2d_cam1, proj_axis_cam1 = self.model_img2vq(img_crop_1, cTh_hand_cam1, proj_mode='FullPerspective', K=self.cam1_intrinsic, bbox=bbox_cam_1)
                        img_cond_cam2, proj_2d_cam2, proj_axis_cam2 = self.model_img2vq(img_crop_2, cTh_hand_cam2, proj_mode='FullPerspective', K=self.cam2_intrinsic, bbox=bbox_cam_2)
                        print("Recon time: ", time.time()-start_time)
                        img_cond_h_cam1, img_cond_o_cam1 = img_cond_cam1
                        img_cond_h_cam2, img_cond_o_cam2 = img_cond_cam2
                        img_cond_h = F.softmax(F.log_softmax(img_cond_h_cam1, dim=1) + F.log_softmax(img_cond_h_cam2, dim=1), dim=1)
                        img_cond_o = F.softmax(F.log_softmax(img_cond_o_cam1, dim=1) + F.log_softmax(img_cond_o_cam2, dim=1), dim=1)

                        start_time = time.time()
                        p3d_mesh, rendered_img, self.tsdf_h, self.tsdf_o = self.model_img2vq.test_recon(img_cond_h, img_cond_o, 
                                                                            save_path='/home/robot_tutorial/vgn_ws/src/autosdf_j2/demo/{}.gif'.format(fn), 
                                                                            hand_pose=None, save_mesh=save_mesh, save_vis=save_vis)
                        print("Inference time: ", time.time()-start_time)
                        # check IOU of hand and object with seg mask
                        # get hand and object point cloud by thresholding TSDF
                        thresh = 0.005
                        tsdf_h = self.tsdf_h.cpu().numpy()
                        tsdf_o = self.tsdf_o.cpu().numpy()

                        # compare and get binary array of hand between -thresh and thresh
                        hand_occp = np.zeros_like(tsdf_h)
                        hand_occp[np.logical_and(tsdf_h > -0.05, tsdf_h < thresh)] = 1
                        obj_occp = np.zeros_like(tsdf_o)
                        obj_occp[np.logical_and(tsdf_o > -0.05, tsdf_o < thresh)] = 1

                        # get coordinates
                        hand_coords = np.argwhere(hand_occp == 1)
                        obj_coords = np.argwhere(obj_occp == 1)

                        voxel_size = 0.52/40.
                        BBOX_ORIG_X = ((0.2+(-0.32))/2.0)
                        BBOX_ORIG_Y = ((0.16+(-0.36))/2.0)
                        BBOX_ORIG_Z = ((0.25+(-0.27))/2.0)
                        bbox_orig = np.array([BBOX_ORIG_X, BBOX_ORIG_Y, BBOX_ORIG_Z])
                        hand_coords = hand_coords[:,2:]*voxel_size + 0.5*voxel_size - 0.52/2. + bbox_orig
                        obj_coords = obj_coords[:,2:]*voxel_size + 0.5*voxel_size - 0.52/2. + bbox_orig
                        pc_h = hand_coords
                        pc_o = obj_coords

                        # Project to 2D (global hand pose base)
                        rvec = cv2.Rodrigues(cTh_hand_cam1[:3, :3])[0]
                        tvec = cTh_hand_cam1[:3, 3]
                        dist_coeffs = np.zeros((4,1))
                        hand_coords_2d_cam1 = cv2.projectPoints(hand_coords, rvec, tvec, self.cam1_intrinsic, dist_coeffs)[0]
                        hand_coords_2d_cam1 = np.squeeze(hand_coords_2d_cam1, axis=1)/2.0
                        hand_coords_2d_cam1 = (hand_coords_2d_cam1 - np.array([bbox_cam_1[0], bbox_cam_1[1]])) / (bbox_cam_1[2]-bbox_cam_1[0]) * 256.0

                        obj_coords_2d_cam1 = cv2.projectPoints(obj_coords, rvec, tvec, self.cam1_intrinsic, dist_coeffs)[0]
                        obj_coords_2d_cam1 = np.squeeze(obj_coords_2d_cam1, axis=1)/2.0
                        obj_coords_2d_cam1 = (obj_coords_2d_cam1 - np.array([bbox_cam_1[0], bbox_cam_1[1]])) / (bbox_cam_1[2]-bbox_cam_1[0]) * 256.0

                        rvec = cv2.Rodrigues(cTh_hand_cam2[:3, :3])[0]
                        tvec = cTh_hand_cam2[:3, 3]
                        dist_coeffs = np.zeros((4,1))
                        hand_coords_2d_cam2 = cv2.projectPoints(hand_coords, rvec, tvec, self.cam2_intrinsic, dist_coeffs)[0]
                        hand_coords_2d_cam2 = np.squeeze(hand_coords_2d_cam2, axis=1)/2.0
                        hand_coords_2d_cam2 = (hand_coords_2d_cam2 - np.array([bbox_cam_2[0], bbox_cam_2[1]])) / (bbox_cam_2[2]-bbox_cam_2[0]) * 256.0

                        obj_coords_2d_cam2 = cv2.projectPoints(obj_coords, rvec, tvec, self.cam2_intrinsic, dist_coeffs)[0]
                        obj_coords_2d_cam2 = np.squeeze(obj_coords_2d_cam2, axis=1)/2.0
                        obj_coords_2d_cam2 = (obj_coords_2d_cam2 - np.array([bbox_cam_2[0], bbox_cam_2[1]])) / (bbox_cam_2[2]-bbox_cam_2[0]) * 256.0
                    
                        # Get IOU for hand and object
                        hull = cv2.convexHull(hand_coords_2d_cam1.astype(np.int32))
                        pred_hand_mask_cam1 = np.zeros((256, 256), dtype=np.uint8)
                        cv2.fillConvexPoly(pred_hand_mask_cam1, hull, 255)
                        hand_iou_cam1 = computeIOU(hand_binary_mask_1[:,:,0], pred_hand_mask_cam1)

                        hull = cv2.convexHull(hand_coords_2d_cam2.astype(np.int32))
                        pred_hand_mask_cam2 = np.zeros((256, 256), dtype=np.uint8)
                        cv2.fillConvexPoly(pred_hand_mask_cam2, hull, 255)
                        hand_iou_cam2 = computeIOU(hand_binary_mask_2[:,:,0], pred_hand_mask_cam2)

                        hull = cv2.convexHull(obj_coords_2d_cam1.astype(np.int32))
                        pred_obj_mask_cam1 = np.zeros((256, 256), dtype=np.uint8)
                        cv2.fillConvexPoly(pred_obj_mask_cam1, hull, 255)
                        obj_iou_cam1 = computeIOU(binary_mask_1[:,:,0], pred_obj_mask_cam1)

                        hull = cv2.convexHull(obj_coords_2d_cam2.astype(np.int32))
                        pred_obj_mask_cam2 = np.zeros((256, 256), dtype=np.uint8)
                        cv2.fillConvexPoly(pred_obj_mask_cam2, hull, 255)
                        obj_iou_cam2 = computeIOU(binary_mask_2[:,:,0], pred_obj_mask_cam2)

                        print("Hand IOU cam1: ", hand_iou_cam1)
                        print("Hand IOU cam2: ", hand_iou_cam2)
                        print("Object IOU cam1: ", obj_iou_cam1)
                        print("Object IOU cam2: ", obj_iou_cam2)

                        # Remove points from pc if it is not in the mask
                        # swap x y
                        hand_coords_2d_cam1_s = np.flip(hand_coords_2d_cam1.astype(np.int32), axis=1).T
                        hand_coords_2d_cam2_s = np.flip(hand_coords_2d_cam2.astype(np.int32), axis=1).T
                        obj_coords_2d_cam1_s = np.flip(obj_coords_2d_cam1.astype(np.int32), axis=1).T
                        obj_coords_2d_cam2_s = np.flip(obj_coords_2d_cam2.astype(np.int32), axis=1).T
                        hand_coords_2d_cam1_s = np.maximum(np.minimum(hand_coords_2d_cam1_s, 255), 0)
                        hand_coords_2d_cam2_s = np.maximum(np.minimum(hand_coords_2d_cam2_s, 255), 0)
                        obj_coords_2d_cam1_s = np.maximum(np.minimum(obj_coords_2d_cam1_s, 255), 0)
                        obj_coords_2d_cam2_s = np.maximum(np.minimum(obj_coords_2d_cam2_s, 255), 0)

                        select_h_cam1 = hand_binary_mask_1[:,:,0][tuple(hand_coords_2d_cam1_s)]
                        select_h_cam2 = hand_binary_mask_2[:,:,0][tuple(hand_coords_2d_cam2_s)]
                        select_o_cam1 = ho_binary_mask_1[:,:,0][tuple(obj_coords_2d_cam1_s)]
                        select_o_cam2 = ho_binary_mask_2[:,:,0][tuple(obj_coords_2d_cam2_s)]

                        pc_h = pc_h[np.logical_and(select_h_cam1 == 1, select_h_cam2 == 1)]
                        pc_o = pc_o[np.logical_and(select_o_cam1 == 1, select_o_cam2 == 1)]
                        hand_coords_2d_cam1 = hand_coords_2d_cam1[np.logical_and(select_h_cam1 == 1, select_h_cam2 == 1)]
                        hand_coords_2d_cam2 = hand_coords_2d_cam2[np.logical_and(select_h_cam1 == 1, select_h_cam2 == 1)]
                        obj_coords_2d_cam1 = obj_coords_2d_cam1[np.logical_and(select_o_cam1 == 1, select_o_cam2 == 1)]
                        obj_coords_2d_cam2 = obj_coords_2d_cam2[np.logical_and(select_o_cam1 == 1, select_o_cam2 == 1)]
                        
                        IOU_sum = hand_iou_cam1 + hand_iou_cam2 + obj_iou_cam1 + obj_iou_cam2
                        print("IOU sum: ", IOU_sum)
                        if IOU_sum < 2.1 and IOU_sum < IOU_sum_best:
                            print("IOU below threshold")
                            continue
                        IOU_sum_best = IOU_sum

                        pc_h_best = pc_h
                        pc_o_best = pc_o
                        
                        if draw_vis:
                            for i in range(hand_coords_2d_cam1.shape[0]):
                                x, y = hand_coords_2d_cam1[i]
                                cv2.circle(seg_img_vis_1, (int(x), int(y)), 2, (255, 125, 0), -1)
                            for i in range(obj_coords_2d_cam1.shape[0]):
                                x, y = obj_coords_2d_cam1[i]
                                cv2.circle(seg_img_vis_1, (int(x), int(y)), 2, (0, 125, 255), -1)
                            
                            for i in range(hand_coords_2d_cam2.shape[0]):
                                x, y = hand_coords_2d_cam2[i]
                                cv2.circle(seg_img_vis_2, (int(x), int(y)), 2, (255, 125, 0), -1)
                            for i in range(obj_coords_2d_cam2.shape[0]):
                                x, y = obj_coords_2d_cam2[i]
                                cv2.circle(seg_img_vis_2, (int(x), int(y)), 2, (0, 125, 255), -1)

                            debug_img1 = img_crop_orig_1
                            debug_img2 = img_crop_orig_2
                            debug_img1 = np.concatenate([debug_img1, seg_img_vis_1], axis=1).astype(np.uint8)
                            debug_img2 = np.concatenate([debug_img2, seg_img_vis_2], axis=1).astype(np.uint8)
                            debug_img = np.concatenate([debug_img1, debug_img2], axis=0).astype(np.uint8)

                        if save_vis:
                            shutil.copy('/home/robot_tutorial/vgn_ws/src/autosdf_j2/demo/{}.gif'.format(fn), '/home/robot_tutorial/vgn_ws/src/autosdf_j2/demo/test.gif')
                            cv2.imwrite('/home/robot_tutorial/vgn_ws/src/autosdf_j2/demo/{}.jpg'.format(fn), debug_img)

                        if save_mesh and (p3d_mesh is not None):
                            print("Saving mesh...")
                            IO().save_mesh(p3d_mesh, '/home/robot_tutorial/vgn_ws/src/autosdf_j2/demo/{}.obj'.format(fn))

                    print("Publishing point clouds...")
                    print("Hand point cloud shape: ", pc_h_best.shape)
                    print("Object point cloud shape: ", pc_o_best.shape)
                    self.pub_pc_h.publish(pc_h_best.reshape(-1).astype(np.float32))
                    self.pub_pc_o.publish(pc_o_best.reshape(-1).astype(np.float32))


                    self.pub_debug_image.publish(debug_img.reshape(-1).astype(np.int16))
                    self.idx += 1
                    
            else:
                print("Waiting for data...")
            

        rospy.spin()

if __name__ == '__main__':
    node = HandObjectReconRosNode()
    node.run()