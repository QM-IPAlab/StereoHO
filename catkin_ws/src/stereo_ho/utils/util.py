from __future__ import print_function

import os
import random

import trimesh
from scipy.spatial import cKDTree as KDTree

import numpy as np
from PIL import Image
from einops import rearrange

import torch
import torchvision.utils as vutils

from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler

# -b-0.32,-0.36,-0.27,0.2,0.16,0.25
# SDF_MULTIPLIER = 50.0
SDF_MULTIPLIER = 50.0
EXTENDED_RATIO = 0.5

BBOX_ORIG_X = SDF_MULTIPLIER * ((0.2+(-0.32))/2.0)
BBOX_ORIG_Y = SDF_MULTIPLIER * ((0.16+(-0.36))/2.0)
BBOX_ORIG_Z = SDF_MULTIPLIER * ((0.25+(-0.27))/2.0)
BBOX_SIZE_X = SDF_MULTIPLIER * (0.2-(-0.32))
BBOX_SIZE_Y = SDF_MULTIPLIER * (0.16-(-0.36))
BBOX_SIZE_Z = SDF_MULTIPLIER * (0.25-(-0.27))

HAND_RGB = [1, 1, 1]
OBJ_RGB = [1, 1, 1]
# HAND_RGB = [67/256., 205/256., 220/256.]
# OBJ_RGB = [230/256., 227/256., 67/256.]

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    # image_numpy = image_tensor[0].cpu().float().numpy()
    # if image_numpy.shape[0] == 1:
    #     image_numpy = np.tile(image_numpy, (3, 1, 1))
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # return image_numpy.astype(imtype)

    n_img = min(image_tensor.shape[0], 16)
    image_tensor = image_tensor[:n_img]

    if image_tensor.shape[1] == 1:
        image_tensor = image_tensor.repeat(1, 3, 1, 1)

    # if image_tensor.shape[1] == 4:
        # import pdb; pdb.set_trace()

    image_tensor = vutils.make_grid( image_tensor, nrow=4 )

    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = ( np.transpose( image_numpy, (1, 2, 0) ) + 1) / 2.0 * 255.
    return image_numpy.astype(imtype)

def to_variable(numpy_data, volatile=False):
    numpy_data = numpy_data.astype(np.float32)
    torch_data = torch.from_numpy(numpy_data).float()
    variable = Variable(torch_data, volatile=volatile)
    return variable

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def seed_everything(seed):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def iou(x_gt, x, thres):
    thres_gt = 0.0

    # compute iou
    # > 0 free space, < 0 occupied
    x_gt_mask = x_gt.clone().detach()
    x_gt_mask[x_gt > thres_gt] = 0.
    x_gt_mask[x_gt <= thres_gt] = 1.

    x_mask = x.clone().detach()
    x_mask[x > thres] = 0.
    x_mask[x <= thres] = 1.

    inter = torch.logical_and(x_gt_mask, x_mask)
    union = torch.logical_or(x_gt_mask, x_mask)
    inter = rearrange(inter, 'b c d h w -> b (c d h w)')
    union = rearrange(union, 'b c d h w -> b (c d h w)')

    iou = inter.sum(1) / (union.sum(1) + 1e-12)
    return iou


# Noam Learning rate schedule.
# From https://github.com/tugstugi/pytorch-saltnet/blob/master/utils/lr_scheduler.py
class NoamLR(_LRScheduler):
	
	def __init__(self, optimizer, warmup_steps):
		self.warmup_steps = warmup_steps
		super().__init__(optimizer)

	def get_lr(self):
		last_epoch = max(1, self.last_epoch)
		scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
		return [base_lr * scale for base_lr in self.base_lrs]

def evaluate_score(gt_hand_mesh_path, gt_obj_mesh_path, pred_hand_mesh_path, pred_obj_mesh_path):
    # Ref https://github.com/zerchen/gSDF/blob/master/datasets/obman/obman.py

    if not os.path.exists(pred_hand_mesh_path):
        chamfer_hand = None
        fscore_hand_1 = None
        fscore_hand_5 = None
    else:
        pred_hand_mesh = trimesh.load(pred_hand_mesh_path, process=False)
        gt_hand_mesh = trimesh.load(gt_hand_mesh_path, process=False)
        pred_hand_points, _ = trimesh.sample.sample_surface(pred_hand_mesh, 30000) # m
        gt_hand_points, _ = trimesh.sample.sample_surface(gt_hand_mesh, 30000) # m
        pred_hand_points *= 100. # cm
        gt_hand_points *= 100. # cm

        # one direction
        gen_points_kd_tree = KDTree(pred_hand_points)
        one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_hand_points)
        gt_to_gen_chamfer = np.mean(np.square(one_distances))
        # other direction
        gt_points_kd_tree = KDTree(gt_hand_points)
        two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_hand_points)
        gen_to_gt_chamfer = np.mean(np.square(two_distances))
        chamfer_hand = gt_to_gen_chamfer + gen_to_gt_chamfer

        threshold = 0.1 # 1 mm
        precision_1 = np.mean(one_distances < threshold).astype(np.float32)
        precision_2 = np.mean(two_distances < threshold).astype(np.float32)
        fscore_hand_1 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

        threshold = 0.5 # 5 mm
        precision_1 = np.mean(one_distances < threshold).astype(np.float32)
        precision_2 = np.mean(two_distances < threshold).astype(np.float32)
        fscore_hand_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

    if not os.path.exists(pred_obj_mesh_path):
        chamfer_obj = None
        fscore_obj_5 = None
        fscore_obj_10 = None
    else:
        pred_obj_mesh = trimesh.load(pred_obj_mesh_path, process=False)
        gt_obj_mesh = trimesh.load(gt_obj_mesh_path, process=False)
        pred_obj_points, _ = trimesh.sample.sample_surface(pred_obj_mesh, 30000)
        gt_obj_points, _ = trimesh.sample.sample_surface(gt_obj_mesh, 30000)
        pred_obj_points *= 100.
        gt_obj_points *= 100.

        # one direction
        gen_points_kd_tree = KDTree(pred_obj_points)
        one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_obj_points)
        gt_to_gen_chamfer = np.mean(np.square(one_distances))
        # other direction
        gt_points_kd_tree = KDTree(gt_obj_points)
        two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_obj_points)
        gen_to_gt_chamfer = np.mean(np.square(two_distances))
        chamfer_obj = gt_to_gen_chamfer + gen_to_gt_chamfer

        threshold = 0.5 # 5 mm
        precision_1 = np.mean(one_distances < threshold).astype(np.float32)
        precision_2 = np.mean(two_distances < threshold).astype(np.float32)
        fscore_obj_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

        threshold = 1.0 # 10 mm
        precision_1 = np.mean(one_distances < threshold).astype(np.float32)
        precision_2 = np.mean(two_distances < threshold).astype(np.float32)
        fscore_obj_10 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)
    
    error_dict = {}
    # error_dict['id'] = sample_idx
    error_dict['chamfer_hand'] = chamfer_hand
    error_dict['fscore_hand_1'] = fscore_hand_1
    error_dict['fscore_hand_5'] = fscore_hand_5
    error_dict['chamfer_obj'] = chamfer_obj
    error_dict['fscore_obj_5'] = fscore_obj_5
    error_dict['fscore_obj_10'] = fscore_obj_10
    error_dict['gt_hand_mesh_path'] = gt_hand_mesh_path
    error_dict['gt_obj_mesh_path'] = gt_obj_mesh_path
    error_dict['pred_hand_mesh_path'] = pred_hand_mesh_path
    error_dict['pred_obj_mesh_path'] = pred_obj_mesh_path

    return error_dict