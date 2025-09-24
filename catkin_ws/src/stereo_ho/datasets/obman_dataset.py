"""
    adopted from: https://github.com/shubhtuls/PixelTransformer/blob/03b65b8612fe583b3e35fc82b446b5503dd7b6bd/data/shapenet.py
"""
import os.path
import json
import pickle

import h5py
import numpy as np
from PIL import Image
import cv2
from termcolor import colored, cprint

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from datasets.base_dataset import BaseDataset

from utils.util_3d import read_sdf
from utils import binvox_rw
import utils.util as util
from configs.paths import dataroot

from tqdm import tqdm
import time
import random


def get_code_setting(opt):
    code_setting = f'{opt.vq_model}-{opt.vq_dset}-{opt.vq_cat}-T{opt.trunc_thres}'
    if opt.vq_note != 'default':
        code_setting = f'{code_setting}-{opt.vq_note}'
    return code_setting

# from https://github.com/laughtervv/DISN/blob/master/preprocessing/info.json
class ObManDataset(BaseDataset):

    def initialize(self, opt, phase='train'):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.mlp_decoder = opt.mlp_decoder
        self.num_samples = opt.num_samples
        self.phase = phase
        self.dataset_dir = opt.dataset_dir
        self.dataset_dir = os.path.join(self.dataset_dir, phase)
        self.mesh_dir_h = os.path.join(self.dataset_dir, 'mesh_hand_normalise')
        self.mesh_dir_o = os.path.join(self.dataset_dir, 'mesh_obj_normalise')


        if phase == 'train':
            self.split_list_path = 'datasets/splits/obman/train_87k_fixed.json'
        else:
            self.split_list_path = 'datasets/splits/obman/test_6k.json'

        self.input_list = []
        self.target_list = []

        if self.opt.ho_mode == 'hand':
            self.grid_sdf_path = os.path.join(self.dataset_dir, "sdf_hand/grid/128")
            self.sparse_sdf_path = os.path.join(self.dataset_dir, "sdf_hand/sparse")
        elif self.opt.ho_mode == 'object':
            self.grid_sdf_path = os.path.join(self.dataset_dir, "sdf_obj/grid/128")
            self.sparse_sdf_path = os.path.join(self.dataset_dir, "sdf_obj/sparse")

        with open(self.split_list_path, 'r') as f:
            split_list = json.load(f)
        
        for fn in split_list:
            sparse_sdf_path = os.path.join(self.sparse_sdf_path, "{}.npz".format(fn))
            input_sdf_file = os.path.join(self.grid_sdf_path, "{}.npz".format(fn))
            if not os.path.exists(input_sdf_file):
                continue
            self.input_list.append(input_sdf_file)
            self.target_list.append(sparse_sdf_path)

        
        self.input_list = self.input_list[:self.max_dataset_size]
        self.target_list = self.target_list[:self.max_dataset_size]

        cprint('[*] %d samples loaded.' % (len(self.input_list)), 'yellow')

        self.N = len(self.input_list)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __getitem__(self, index):

        input_file_path = self.input_list[index]
        fn = input_file_path.split('/')[-1].replace('.npz', '')
        
        input_sdf = load_grid_sdf(input_file_path)
        # input_sdf = input_sdf[:,::2,::2,::2]


        if self.mlp_decoder == 0 or self.phase in ['test', 'val']:
            target_samples = torch.zeros(1)
            target_sdf_file = ""
        else:

            target_sdf_file = self.target_list[index]
            target_samples = load_sparse_sdf(target_sdf_file, self.opt.ho_mode, self.num_samples)

        # GT mesh path for visualization
        mesh_path_h = os.path.join(self.mesh_dir_h, '{}.obj'.format(fn))
        mesh_path_o = os.path.join(self.mesh_dir_o, '{}.obj'.format(fn))
        
        ret = {
            'input_sdf': input_sdf,
            'target_sdf': target_samples,
            'mesh_path_h': mesh_path_h,
            'mesh_path_o': mesh_path_o,
            'fn': fn
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'SDFDataset'


class ObManCodeDataset(BaseDataset):

    def initialize(self, opt, phase='train'):
        self.opt = opt
        self.ratio = opt.ratio
        self.max_dataset_size = opt.max_dataset_size
        
        self.code_dir_h = os.path.join(opt.dataset_dir, phase, 'code_hand', 'cb256')
        self.code_dir_o = os.path.join(opt.dataset_dir, phase, 'code_obj', 'cb256')

        if phase == 'train':
            self.split_list_path = 'datasets/splits/obman/train_87k_fixed.json'
        else:
            self.split_list_path = 'datasets/splits/obman/test_6k.json'
        
        self.split_list = []
        with open(self.split_list_path, 'r') as f:
            self.split_list = json.load(f)

        # self.split_list = os.listdir(self.code_dir_h)

        self.fn_list = []
        for _, fn in tqdm(enumerate(self.split_list)):
            code_path_h = os.path.join(self.code_dir_h, fn)
            code_path_o = os.path.join(self.code_dir_o, fn)
            if not os.path.exists(code_path_h) or not os.path.exists(code_path_o):
                continue
            self.fn_list.append(fn)

        self.fn_list = self.fn_list[:self.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.fn_list)), 'yellow')

        self.N = len(self.fn_list)

    def __getitem__(self, index):

        fn = self.fn_list[index]
        
        path_code_h = os.path.join(self.code_dir_h, fn, 'code.npy')
        path_codeidx_h = os.path.join(self.code_dir_h, fn, 'codeix.npy')
        path_code_o = os.path.join(self.code_dir_o, fn, 'code.npy')
        path_codeidx_o = os.path.join(self.code_dir_o, fn, 'codeix.npy')

        # Read code
        code_h = None
        codeidx_h = None
        code_o = None
        codeidx_o = None
        if self.opt.ho_mode == 'hand' or self.opt.ho_mode == 'joint':
            code_h = torch.from_numpy(np.load(path_code_h))
            codeidx_h = torch.from_numpy(np.load(path_codeidx_h))
        if self.opt.ho_mode == 'object' or self.opt.ho_mode == 'joint':
            code_o = torch.from_numpy(np.load(path_code_o))
            codeidx_o = torch.from_numpy(np.load(path_codeidx_o))

        ret = {
            'code_h': code_h,
            'codeidx_h': codeidx_h,
            'code_o': code_o,
            'codeidx_o': codeidx_o,
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'ObManCodeDataset'


class ObManImgDataset(BaseDataset):

    def initialize(self, opt, phase='train'):

        self.opt = opt
        self.ratio = opt.ratio
        self.max_dataset_size = opt.max_dataset_size
        self.phase = phase
        self.load_code = opt.load_code
        # self.load_grid_sdf = opt.load_grid_sdf
        self.load_sparse_sdf = opt.load_sparse_sdf
        self.debug = opt.debug

        if phase == 'train':
            self.split_list_path = 'datasets/splits/obman/train_87k_fixed.json'
        else:
            self.split_list_path = 'datasets/splits/obman/test_6k.json'
            self.load_sparse_sdf = False

        self.split_list = []
        with open(self.split_list_path, 'r') as f:
            self.split_list = json.load(f)

        base_dir = os.path.join(opt.dataset_dir, phase)
        self.code_dir_h = os.path.join(base_dir, 'code_hand', 'cb256')
        self.code_dir_o = os.path.join(base_dir, 'code_obj', 'cb256')
        self.img_dir = os.path.join(base_dir, 'rgb')
        self.hand_pose_dir = os.path.join(base_dir, 'hand_pose')
        self.mesh_dir_h = os.path.join(base_dir, 'mesh_hand_normalise')
        self.mesh_dir_o = os.path.join(base_dir, 'mesh_obj_normalise')
        self.sparse_sdf_path_h = os.path.join(base_dir, 'sdf_hand', 'sparse')
        self.sparse_sdf_path_o = os.path.join(base_dir, 'sdf_obj', 'sparse')

        # assert os.path.exists(self.code_dir_h), f'{self.code_dir_h} should exist.'
        # assert os.path.exists(self.code_dir_o), f'{self.code_dir_o} should exist.'
        assert os.path.exists(self.img_dir), f'{self.img_dir} should exist.'
        assert os.path.exists(self.hand_pose_dir), f'{self.hand_pose_dir} should exist.'
        # assert os.path.exists(self.mesh_dir_h), f'{self.mesh_dir_h} should exist.'
        # assert os.path.exists(self.mesh_dir_o), f'{self.mesh_dir_o} should exist.'
        
        self.fn_list = []
        for _, fn in tqdm(enumerate(self.split_list)):
            if self.load_code:
                # code_path_h = os.path.join(self.code_dir_h, fn)
                # code_path_o = os.path.join(self.code_dir_o, fn)
                # if not os.path.exists(code_path_h) or not os.path.exists(code_path_o):
                img_path = os.path.join(self.img_dir, '{}.jpg'.format(fn))
                if not os.path.exists(img_path):
                    continue
            # if self.load_sparse_sdf:
            #     sdf_path_h = os.path.join(self.sparse_sdf_path_h, '{}.npz'.format(fn))
            #     sdf_path_o = os.path.join(self.sparse_sdf_path_o, '{}.npz'.format(fn))
            #     if not os.path.exists(sdf_path_h) or not os.path.exists(sdf_path_o):
            #         continue
            self.fn_list.append(fn)
                
        # if phase == 'train':
            # random.shuffle(self.fn_list)

        self.fn_list = self.fn_list[:self.max_dataset_size]

        ds_start = opt.resnet_test_split_start
        ds_end = opt.resnet_test_split_end
        if phase == 'test' and ds_start != None and ds_end != None:
            ds_start_id = int(len(self.fn_list) * ds_start)
            ds_end_id = int(len(self.fn_list) * ds_end)
            self.fn_list = self.fn_list[ds_start_id:ds_end_id]

        cprint('[*] %d samples loaded.' % (len(self.fn_list)), 'yellow')

        self.N = len(self.fn_list)

        # See gSDF or IHOI for transforms
        self.to_tensor = transforms.ToTensor()
        self.transforms_resize = transforms.Resize((256, 256))
        self.to_PIL = transforms.ToPILImage()
        # self.transforms_color = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
        # self.transforms_affine = transforms.RandomAffine(0, scale=(0.7, 1.25), interpolation=InterpolationMode.BILINEAR)
        # self.transforms_rot = transforms.RandomRotation(45, interpolation=InterpolationMode.BILINEAR)
        self.rot_factor = 90.0
        color_factor = 0.2
        self.c_up = 1.0 + color_factor
        self.c_low = 1.0 - color_factor
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.transforms_norm = transforms.Normalize(mean, std)
        self.random_erase = transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False)


    def __getitem__(self, index):
        fn = self.fn_list[index]

        # # Read extracted code
        # path_codeidx_h = os.path.join(self.code_dir_h, fn, 'codeix.npy')
        # path_codeidx_o = os.path.join(self.code_dir_o, fn, 'codeix.npy')
        # # path_code_h = os.path.join(self.code_dir_h, fn, 'code.npy')
        # # path_code_o = os.path.join(self.code_dir_o, fn, 'code.npy')
        # if self.load_code:
        #     codeidx_h = torch.from_numpy(np.load(path_codeidx_h))
        #     codeidx_o = torch.from_numpy(np.load(path_codeidx_o))
        #     # code_h = torch.from_numpy(np.load(path_code_h))
        #     # code_o = torch.from_numpy(np.load(path_code_o))
        # else:
        codeidx_h = torch.zeros((8,8,8))
        codeidx_o = torch.zeros((8,8,8))
        #     # code_h = torch.zeros(1)
        #     # code_o = torch.zeros(1)
        
        # # Read SDF
        # if self.load_sparse_sdf:
        #     sdf_path_h = os.path.join(self.sparse_sdf_path_h, '{}.npz'.format(fn))
        #     sdf_path_o = os.path.join(self.sparse_sdf_path_o, '{}.npz'.format(fn))
        #     sdf_h = load_sparse_sdf(sdf_path_h, 'hand', self.opt.num_samples)
        #     sdf_o = load_sparse_sdf(sdf_path_o, 'object', self.opt.num_samples)
        # else:
        sdf_h = torch.zeros(1)
        sdf_o = torch.zeros(1)
            
        # Read image
        img_path = os.path.join(self.img_dir, '{}.jpg'.format(fn))
        # img = Image.open(img_path).convert('RGB')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.phase == 'train':
            # rot = random.uniform(-self.transforms_rot_angle, self.transforms_rot_angle)
            rot = np.clip(np.random.randn(), -2.0, 2.0) * self.rot_factor if random.random() <= 0.6 else 0
            # rotate with cv2
            h, w, _ = img.shape
            center = (w / 2, h / 2)
            rot_mat = cv2.getRotationMatrix2D(center, rot, 1.0)
            img = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
            # img = TF.rotate(img, rot, InterpolationMode.BILINEAR)
            rot_aug_mat = torch.from_numpy(np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0, 0],
                                                     [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0, 0],
                                                     [0, 0, 1, 0],
                                                     [0, 0, 0, 1]], dtype=np.float32))
            color_scale = [random.uniform(self.c_low, self.c_up), random.uniform(self.c_low, self.c_up), random.uniform(self.c_low, self.c_up)]
            for i in range(3):
                img[:, :, i] = img[:, :, i] * color_scale[i]
            img = np.clip(img, 0, 255)
            img = img.astype(np.uint8)
        
        img = self.to_PIL(img)
        img = self.to_tensor(img)
        if self.phase == 'train':
            img = self.random_erase(img)
        img = self.transforms_norm(img)


        # Read hand pose
        hand_pose_path = os.path.join(self.hand_pose_dir, '{}'.format(fn))
        with open(hand_pose_path, 'rb') as f:
            hand_pose = pickle.load(f)['cTh']
        if self.phase == 'train':
            # hand_pose is pose in camera frame
            hand_pose = torch.matmul(rot_aug_mat, hand_pose)
        
        # GT mesh path for visualization
        mesh_path_h = os.path.join(self.mesh_dir_h, '{}.obj'.format(fn))
        mesh_path_o = os.path.join(self.mesh_dir_o, '{}.obj'.format(fn))

        ret = {

            'sdf_h': sdf_h,
            'sdf_o': sdf_o,
            'mesh_path_h': mesh_path_h,
            'mesh_path_o': mesh_path_o,
            'codeidx_h': codeidx_h,
            'codeidx_o': codeidx_o,
            # 'code_h': code_h,
            # 'code_o': code_o,
            'img': img,
            'hand_pose': hand_pose,
            'fn': fn
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'ObManImageDataset'


# From grasping field
def unpack_sdf_samples(filename, subsample=None, hand=True, clamp=None, filter_dist=False):
    npz = np.load(filename)
    if subsample is None:
        return npz
    try:
        pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
        neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
        pos_sdf_other = torch.from_numpy(npz["pos_other"])
        neg_sdf_other = torch.from_numpy(npz["neg_other"])
        if hand:
            lab_pos_tensor = torch.from_numpy(npz["lab_pos"])
            lab_neg_tensor = torch.from_numpy(npz["lab_neg"])
        else:
            lab_pos_tensor = torch.from_numpy(npz["lab_pos_other"])
            lab_neg_tensor = torch.from_numpy(npz["lab_neg_other"])
    except Exception as e:
        print("fail to load {}, {}".format(filename, e))
    ### make it (x,y,z,sdf_to_hand,sdf_to_obj)
    if hand:
        pos_tensor = torch.cat([pos_tensor, pos_sdf_other], 1)
        neg_tensor = torch.cat([neg_tensor, neg_sdf_other], 1)
    else:
        xyz_pos = pos_tensor[:, :3]
        sdf_pos = pos_tensor[:, 3].unsqueeze(1)
        pos_tensor = torch.cat([xyz_pos, pos_sdf_other, sdf_pos], 1)

        xyz_neg = neg_tensor[:, :3]
        sdf_neg = neg_tensor[:, 3].unsqueeze(1)
        neg_tensor = torch.cat([xyz_neg, neg_sdf_other, sdf_neg], 1)

    # split the sample into half
    half = int(subsample / 2)

    if filter_dist:
        pos_tensor, lab_pos_tensor = filter_invalid_sdf(pos_tensor, lab_pos_tensor, 2.0)
        neg_tensor, lab_neg_tensor = filter_invalid_sdf(neg_tensor, lab_neg_tensor, 2.0)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    # label
    sample_pos_lab = torch.index_select(lab_pos_tensor, 0, random_pos)
    sample_neg_lab = torch.index_select(lab_neg_tensor, 0, random_neg)

    # hand part label
    # 0-finger corase, 1-finger fine, 2-contact, 3-sealed wrist
    hand_part_pos = sample_pos_lab[:, 0]
    hand_part_neg = sample_neg_lab[:, 0]
    samples = torch.cat([sample_pos, sample_neg], 0)
    labels = torch.cat([hand_part_pos, hand_part_neg], 0)

    if clamp:
        labels[samples[:, 3] < -clamp] = -1
        labels[samples[:, 3] > clamp] = -1

    if not hand:
        labels[:] = -1
    return samples, labels

def unpack_normal_params(filename):
    npz = np.load(filename)
    scale = torch.from_numpy(npz["scale"])
    offset = torch.from_numpy(npz["offset"])
    return scale, offset

def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]

def filter_invalid_sdf(tensor, lab_tensor, dist):
    keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
    return tensor[keep, :], lab_tensor[keep, :]

def load_sparse_sdf(path, ho_mode='hand', num_samples=20000):
    try:
        target_data = np.load(path)
    except:
        print("Failed to load target_sdf_file {}".format(path))

    target_samples = target_data['target_samples']

    num_samples = int(num_samples)

    target_samples = torch.from_numpy(target_samples).float()
    if ho_mode == 'hand':
        target_sdf = target_samples[:, 3]
    elif ho_mode == 'object':
        target_sdf = target_samples[:, 4]
    target_sdf = target_sdf.unsqueeze(1)
    target_samples = torch.cat([target_samples[:, :3], target_sdf], dim=1)
    target_samples = (target_samples/100.0) * util.SDF_MULTIPLIER # prepared SDF is metric x100.0
    
    # Sample
    pos_tensor = target_samples[target_samples[:, 3] >= 0]
    neg_tensor = target_samples[target_samples[:, 3] < 0]
    half = int(num_samples/2)
    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    target_samples = torch.cat([sample_pos, sample_neg], 0)
    target_samples[:, 3:] = torch.clamp(target_samples[:, 3:], -1.0, 1.0)

    return target_samples

def load_grid_sdf(path):
    sdf = np.load(path)['input_sdf']
    sdf = torch.from_numpy(sdf).float()
    sdf = sdf * util.SDF_MULTIPLIER
    sdf = torch.clamp(sdf, min=-1.0, max=1.0)
    # sdf = torch.clamp(sdf, min=-0.2, max=0.2)
    
    return sdf