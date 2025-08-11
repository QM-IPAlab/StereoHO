"""
Wrapper for Hand Pose Estimator using HandMocap.
Wrapper stolen from https://github.com/facebookresearch/phosa/blob/15f864d68ed3ed4536f019ad5713dda388d7c666/phosa/bodymocap.py
See: https://github.com/facebookresearch/frankmocap
"""

import os.path as osp
import numpy as np
import torch
import sys

sys.path.append('externals/frankmocap', )
sys.path.append('externals/frankmocap/detectors/body_pose_estimator/', )

from handmocap.hand_mocap_api import HandMocap

from torchvision.transforms import ToTensor
from ihoi.nnutils import image_utils, geom_utils

from ihoi.nnutils.hand_utils import ManopthWrapper

def get_handmocap_predictor(
        mocap_dir='externals/frankmocap',
        checkpoint_hand='extra_data/hand_module/pretrained_weights/pose_shape_best.pth', 
        smpl_dir='extra_data/smpl/',
    ):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    hand_mocap = HandMocap(osp.join(mocap_dir, checkpoint_hand), 
        osp.join(mocap_dir, smpl_dir), device = device)
    return hand_mocap



def process_mocap_predictions(mocap_predictions, image, hand_wrapper=None, mask=None, bbox_len=0):
    if hand_wrapper is None:
        hand_wrapper = ManopthWrapper().to('cpu')
    one_hand = mocap_predictions[0]['right_hand']
    root_2d = one_hand['pred_joints_img'][0]
    pose = torch.FloatTensor(one_hand['pred_hand_pose'])
    rot, hA = pose[..., :3], pose[..., 3:]
    hA = hA + hand_wrapper.hand_mean

    x1, y1 = one_hand['bbox_top_left']
    if bbox_len == 0:
        bbox_len = 360 / one_hand['bbox_scale_ratio']
    x2, y2 = x1 + bbox_len, y1 + bbox_len
    
    hand_bbox = np.array([x1,y1, x2, y2])
    hoi_bbox = image_utils.joint_bbox(hand_bbox)
    hoi_bbox = image_utils.square_bbox(hoi_bbox, 1, scale=0.7)

    verts2d = (one_hand['pred_vertices_img'][:, :2] - np.array([hoi_bbox[0], hoi_bbox[1]])) / (hoi_bbox[2] - hoi_bbox[0]) * 256.
    
    cTh, cam_f, cam_p, new_center, cTh_orig = get_camera(one_hand['pred_camera'], one_hand['bbox_top_left'], one_hand['bbox_scale_ratio'], hoi_bbox, hand_wrapper, hA, rot)
    crop = image_utils.crop_resize(image, hoi_bbox, return_np=False)
    crop = ToTensor()(crop)[None] * 2 - 1

    if mask is None:
        mask = torch.ones([1, 1, crop.shape[-2], crop.shape[-1]])
    else:
        mask = image_utils.crop_resize(mask, hoi_bbox, return_np=False)
        mask = ToTensor()(mask)[None]
    cTh_se3 = geom_utils.matrix_to_se3(cTh)

    data = {
        'cTh_mat': cTh,
        'cTh_mat_orig': cTh_orig,
        'cTh': cTh_se3,
        'hA': hA,
        'image': crop,
        'obj_mask': mask,
        'cam_f': cam_f,
        'cam_p': cam_p,
        'crop_center': new_center,
        'hoi_bbox': hoi_bbox,
        'root_2d': root_2d,
        'joints_2d': one_hand['pred_joints_img'],
        'verts_2d': verts2d,
        'mano_pose': pose,
        'rot': rot
    }
    return data


def get_handmocap_detector(view_type='ego_centric'):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bbox_detector =  HandBboxDetector(view_type, device)
    return bbox_detector


def get_camera(pred_cam, hand_bbox_tl, bbox_scale, bbox, hand_wrapper, hA, rot, fx=10):
    new_center = (bbox[0:2] + bbox[2:4]) / 2
    new_size = max(bbox[2:4] - bbox[0:2])
    s_orig, tx_orig, ty_orig = pred_cam
    cam, topleft, scale = image_utils.crop_weak_cam(
        pred_cam, hand_bbox_tl, bbox_scale, new_center, new_size)
    s, tx, ty = cam
    
    f = torch.FloatTensor([[fx, fx]])
    p = torch.FloatTensor([[0, 0]])

    translate = torch.FloatTensor([[tx, ty, fx/s]])
    translate_orig = torch.FloatTensor([[tx_orig, ty_orig, fx/s_orig]])
    
    _, joints = hand_wrapper(
        geom_utils.matrix_to_se3(geom_utils.axis_angle_t_to_matrix(rot)), 
        hA)
    
    cTh = geom_utils.axis_angle_t_to_matrix(
        rot, translate - joints[:, 5] + joints[:, 0])
    
    cTh_orig = geom_utils.axis_angle_t_to_matrix(
        rot, translate_orig - joints[:, 5] + joints[:, 0])
    return cTh, f, p, new_center, cTh_orig

