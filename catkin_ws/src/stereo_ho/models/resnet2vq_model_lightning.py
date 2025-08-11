import os
from collections import OrderedDict

import numpy as np
import cv2
import einops
import mcubes
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import rearrange, repeat
from tqdm import tqdm
from pytorch3d.io import IO
import pytorch3d

import torch
import torch.nn.functional as F
from torch import nn, optim

import torchvision.utils as vutils
import torchvision.transforms as transforms

from models.base_model import BaseModel
from models.networks.resnet2vq_net import ResNet2VQ
from models.networks.pvqvae_networks.auto_encoder import PVQVAE
from models.networks.losses import FocalLoss

import utils
from utils.util_3d import init_mesh_renderer, render_sdf, sdf_to_mesh, ho_sdf_to_mesh, load_mesh
from utils.qual_util import save_mesh_as_gif, save_mesh_as_img

import lightning.pytorch as pl

from datasets.obman_dataset import load_grid_sdf

class ResNet2VQModelLightning(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        assert opt.vq_cfg is not None

        self.opt = opt
        self.debug = int(opt.debug)
        self.focal_loss = int(opt.focal_loss)
        self.label_smoothing = float(opt.label_smoothing)
        vq_conf = OmegaConf.load(opt.vq_cfg)
        tf_conf = OmegaConf.load(opt.tf_cfg)
        ntokens = tf_conf.model.params.ntokens

        # define loss functions
        self.empty_idx_h = opt.empty_idx_h
        self.empty_idx_o = opt.empty_idx_o
        alpha = 0.25
        weight_h = torch.ones(ntokens).to(opt.device) * (1.0 - alpha)
        weight_h[self.empty_idx_h] = alpha
        weight_o = torch.ones(ntokens).to(opt.device) * (1.0 - alpha)
        weight_o[self.empty_idx_o] = alpha
        self.criterion_nce_h = nn.CrossEntropyLoss(weight=weight_h)
        self.criterion_nce_o = nn.CrossEntropyLoss(weight=weight_o)
        
        alpha_h = weight_h.tolist()
        alpha_o = weight_o.tolist()
        self.criterion_focal_h = FocalLoss(gamma=2.0, alpha=alpha_h)
        self.criterion_focal_o = FocalLoss(gamma=2.0, alpha=alpha_o)
        self.criterion_l1 = nn.L1Loss()
        self.criterion_l2 = nn.MSELoss()

        # init resnet2vq network
        self.net = ResNet2VQ(opt, tf_conf, self.empty_idx_h, self.empty_idx_o)
        
        # init vqvae for decoding shapes
        mparam = vq_conf.model.params
        n_embed = mparam.n_embed
        embed_dim = mparam.embed_dim
        ddconfig = mparam.ddconfig
        decmlpconfig = mparam.decmlpconfig
        self.load_vqvae(ddconfig, decmlpconfig, n_embed, embed_dim)

        self.resolution = ddconfig['resolution']

        # hyper-parameters for SDF
        nC = self.resolution
        assert nC == 128, 'right now, only trained with sdf resolution = 128'
        self.down_size = 8   # x: res, x_cube: res//8
        self.cube_size = nC // self.down_size    # size per cube. nC=64, down_size=8 -> size=8 for each smaller cube
        self.stride = nC // self.down_size
        self.ncubes_per_dim = nC // self.cube_size

        # grid size
        self.grid_size = 8

        # setup renderer
        dist, elev, azim = 0.8, 20, 20   
        self.renderer = init_mesh_renderer(image_size=512, dist=dist, elev=elev, azim=azim, device=self.opt.device)


    def configure_optimizers(self):
        optimizer = optim.Adam([p for p in self.net.parameters() if p.requires_grad == True], lr=self.opt.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 1.0)
        
        return [optimizer], [scheduler]


    def load_vqvae(self, ddconfig, decmlpconfig, n_embed, embed_dim):
        assert type(self.opt.vq_ckpt_h) == str
        self.vqvae_h = PVQVAE(ddconfig, decmlpconfig, n_embed, embed_dim, self.opt.mlp_decoder)

        state_dict = torch.load(self.opt.vq_ckpt_h)['state_dict']
        # Remove vqvae from name
        state_dict = {k.replace('vqvae.', ''): v for k, v in state_dict.items()}
        self.vqvae_h.load_state_dict(state_dict)
        print(colored('[*] VQVAE hand: weight successfully load from: %s' % self.opt.vq_ckpt_h, 'blue'))
        
        self.vqvae_h.eval()
        for param in self.vqvae_h.parameters():
            param.requires_grad = False
    
        assert type(self.opt.vq_ckpt_o) == str
        self.vqvae_o = PVQVAE(ddconfig, decmlpconfig, n_embed, embed_dim, self.opt.mlp_decoder)

        state_dict = torch.load(self.opt.vq_ckpt_o)['state_dict']
        # Remove vqvae from name
        state_dict = {k.replace('vqvae.', ''): v for k, v in state_dict.items()}
        self.vqvae_o.load_state_dict(state_dict)
        print(colored('[*] VQVAE object: weight successfully load from: %s' % self.opt.vq_ckpt_o, 'blue'))
        
        self.vqvae_o.eval()
        for param in self.vqvae_o.parameters():
            param.requires_grad = False
    

    def compute_acc(self, outp, tgt, empty_idx):
        # top-1 accuracy
        pred = outp.argmax(dim=-1)
        correct = pred.eq(tgt).sum().item()
        acc = correct / float(tgt.numel())

        # top-1 accuracy no empty
        mask = tgt != empty_idx
        acc_noempty = torch.sum(pred[mask] == tgt[mask]).float() / mask.sum()

        # binary accuracy of occupancy
        pred_bin = pred != empty_idx
        tgt_bin = tgt != empty_idx
        acc_bin_pos = torch.sum(pred_bin[tgt_bin] == tgt_bin[tgt_bin]).float() / tgt_bin.sum()
        acc_bin_neg = torch.sum(pred_bin[~tgt_bin] == tgt_bin[~tgt_bin]).float() / (~tgt_bin).sum()

        return acc, acc_noempty, acc_bin_pos, acc_bin_neg


    def training_step(self, batch, batch_idx):
        self.vqvae_h.eval()
        self.vqvae_o.eval()
        self.net.train()

        bs = batch['img'].shape[0]
        img = batch['img']
        sdf_h = batch['sdf_h']
        sdf_o = batch['sdf_o']
        mesh_path_h = batch['mesh_path_h']
        mesh_path_o = batch['mesh_path_o']
        codeidx_h = batch['codeidx_h']
        codeidx_o = batch['codeidx_o']
        hand_pose = batch['hand_pose']

        idx_seq_h = rearrange(codeidx_h, 'bs dz hz wz -> (dz hz wz) bs').contiguous() # to (T, B)
        idx_seq_o = rearrange(codeidx_o, 'bs dz hz wz -> (dz hz wz) bs').contiguous()

        outp, proj_2d, proj_axis = self.forward(img, hand_pose) # outp is (B, cls, D, H, W)
        
        # Compute latent loss
        outp_h, outp_o = outp
        outp_h = rearrange(outp_h, 'bs cls d h w -> (d h w) bs cls').contiguous() # to (T, B, cls)
        outp_h = rearrange(outp_h, 'seq bs cls -> (seq bs) cls')
        outp_o = rearrange(outp_o, 'bs cls d h w -> (d h w) bs cls').contiguous() # to (T, B, cls)
        outp_o = rearrange(outp_o, 'seq bs cls -> (seq bs) cls')
        tgt_h = rearrange(idx_seq_h, 'seq bs -> (seq bs)')
        tgt_o = rearrange(idx_seq_o, 'seq bs -> (seq bs)')

        if self.focal_loss >= 0 and self.current_epoch >= self.focal_loss:
            print("Using focal loss")
            loss_latent_h = self.criterion_focal_h(outp_h, tgt_h)
            loss_latent_o = self.criterion_focal_o(outp_o, tgt_o)
        else:
            loss_latent_h = self.criterion_nce_h(outp_h, tgt_h)
            loss_latent_o = self.criterion_nce_o(outp_o, tgt_o)

        loss = 0.5 * loss_latent_h + 0.5 * loss_latent_o


        acc_h, acc_noempty_h, acc_bin_pos_h, acc_bin_neg_h = self.compute_acc(outp_h, tgt_h, self.empty_idx_h)
        acc_o, acc_noempty_o, acc_bin_pos_o, acc_bin_neg_o = self.compute_acc(outp_o, tgt_o, self.empty_idx_o)
        acc = (acc_h + acc_o) / 2.0

        self.log('train/loss', loss, sync_dist=True)
        self.log('train/loss_h', loss_latent_h, sync_dist=True)
        self.log('train/loss_o', loss_latent_o, sync_dist=True)
        self.log('train/acc', acc, sync_dist=True)
        self.log('train/acc_h', acc_h, sync_dist=True)
        self.log('train/acc_o', acc_o, sync_dist=True)
        self.log('train/acc_h_noempty', acc_noempty_h, sync_dist=True)
        self.log('train/acc_o_noempty', acc_noempty_o, sync_dist=True)
        self.log("train/acc_bin_pos_h", acc_bin_pos_h, sync_dist=True)
        self.log("train/acc_bin_pos_o", acc_bin_pos_o, sync_dist=True)
        self.log("train/acc_bin_neg_h", acc_bin_neg_h, sync_dist=True)
        self.log("train/acc_bin_neg_o", acc_bin_neg_o, sync_dist=True)

        # Only vis on rank 0
        if self.global_rank == 0:
            print("global_step", self.global_step)
            try:
                if self.global_step % self.opt.display_freq == 0:
                    print("Saving gif")
                    # Save input image
                    gif_dir = os.path.join(self.logger.log_dir, "gif", "train")
                    save_path = os.path.join(gif_dir, 'e_{:05d}_b_{:05d}_img.png'.format(self.current_epoch, batch_idx))
                    vutils.save_image(img[0], save_path, normalize=True)
                    
                    # Visualise reconsturction, argmax to get the index
                    recon_code_h, recon_code_o = self.get_recon_code(outp_h, outp_o, batch_size=bs)
                    x_recon_h, x_recon_o = self.generate_vis(recon_code_h, recon_code_o, device=img.device)
                    # x_recon_h, x_recon_o = self.generate_vis(quant_h, quant_o, device=img.device, enc_indices=False)

                    mesh_h = load_mesh(mesh_path_h[0], color=utils.util.HAND_RGB).to(img.device)
                    mesh_o = load_mesh(mesh_path_o[0], color=utils.util.OBJ_RGB).to(img.device)
                    self.save_vis(mesh_h, mesh_o, x_recon_h, x_recon_o,
                                gif_dir=gif_dir, epoch=self.current_epoch, step=batch_idx)

                    # Draw proj 2d points
                    img_save = cv2.imread(save_path)
                    proj_2d = proj_2d[0].cpu().numpy()
                    for i in range(proj_2d.shape[0]):
                        x, y = proj_2d[i]
                        cv2.circle(img_save, (int(x), int(y)), 2, (0, 0, 255), -1)
                    save_path = os.path.join(gif_dir, 'e_{:05d}_b_{:05d}_proj.png'.format(self.current_epoch, batch_idx))
                    cv2.imwrite(save_path, img_save)
                    # Draw proj axis
                    img_save = cv2.imread(save_path)
                    proj_axis = proj_axis[0].cpu().numpy()
                    pairs = [[0,1], [0,2], [0,3]]
                    colors = [(0,0,255), (0,255,0), (255,0,0)]
                    for i, pair in enumerate(pairs):
                        x1, y1 = proj_axis[pair[0]]
                        x2, y2 = proj_axis[pair[1]]
                        cv2.line(img_save, (int(x1), int(y1)), (int(x2), int(y2)), colors[i], 2)
                    save_path = os.path.join(gif_dir, 'e_{:05d}_b_{:05d}_proj_axis.png'.format(self.current_epoch, batch_idx))
                    cv2.imwrite(save_path, img_save)
            except Exception as e:
                print(e)
                print("Error in saving gif")

        return loss
    
    def validation_step(self, batch, batch_idx):
        self.vqvae_h.eval()
        self.vqvae_o.eval()
        self.net.eval()
        
        bs = batch['img'].shape[0]
        img = batch['img']
        sdf_h = batch['sdf_h']
        sdf_o = batch['sdf_o']
        mesh_path_h = batch['mesh_path_h']
        mesh_path_o = batch['mesh_path_o']
        codeidx_h = batch['codeidx_h']
        codeidx_o = batch['codeidx_o']
        hand_pose = batch['hand_pose']

        idx_seq_h = rearrange(codeidx_h, 'bs dz hz wz -> (dz hz wz) bs').contiguous() # to (T, B)
        idx_seq_o = rearrange(codeidx_o, 'bs dz hz wz -> (dz hz wz) bs').contiguous()

        outp, proj_2d, proj_axis = self.forward(img, hand_pose)

        # Compute latent loss
        outp_h, outp_o = outp

        outp_h = rearrange(outp_h, 'bs cls d h w -> (d h w) bs cls').contiguous() # to (T, B, cls)
        outp_h = rearrange(outp_h, 'seq bs cls -> (seq bs) cls')
        outp_o = rearrange(outp_o, 'bs cls d h w -> (d h w) bs cls').contiguous() # to (T, B, cls)
        outp_o = rearrange(outp_o, 'seq bs cls -> (seq bs) cls')
        tgt_h = rearrange(idx_seq_h, 'seq bs -> (seq bs)')
        tgt_o = rearrange(idx_seq_o, 'seq bs -> (seq bs)')

        loss_latent_h = self.criterion_nce_h(outp_h, tgt_h)
        loss_latent_o = self.criterion_nce_o(outp_o, tgt_o)
        loss = 0.5 * loss_latent_h + 0.5 * loss_latent_o

        acc_h, acc_noempty_h, acc_bin_pos_h, acc_bin_neg_h = self.compute_acc(outp_h, tgt_h, self.empty_idx_h)
        acc_o, acc_noempty_o, acc_bin_pos_o, acc_bin_neg_o = self.compute_acc(outp_o, tgt_o, self.empty_idx_o)
        acc = (acc_h + acc_o) / 2.0

        self.log('val/loss', loss, sync_dist=True, on_epoch=True)
        self.log('val/loss_h', loss_latent_h, sync_dist=True, on_epoch=True)
        self.log('val/loss_o', loss_latent_o, sync_dist=True, on_epoch=True)
        self.log('val/acc', acc, sync_dist=True, on_epoch=True)
        self.log('val/acc_h', acc_h, sync_dist=True, on_epoch=True)
        self.log('val/acc_o', acc_o, sync_dist=True, on_epoch=True)
        self.log('val/acc_h_noempty', acc_noempty_h, sync_dist=True, on_epoch=True)
        self.log('val/acc_o_noempty', acc_noempty_o, sync_dist=True, on_epoch=True)
        self.log("val/acc_bin_pos_h", acc_bin_pos_h, sync_dist=True, on_epoch=True)
        self.log("val/acc_bin_pos_o", acc_bin_pos_o, sync_dist=True, on_epoch=True)
        self.log("val/acc_bin_neg_h", acc_bin_neg_h, sync_dist=True, on_epoch=True)
        self.log("val/acc_bin_neg_o", acc_bin_neg_o, sync_dist=True, on_epoch=True)

        # Only vis on rank 0
        if self.global_rank == 0 and self.debug == 0:
            try:
                if self.current_epoch % 10 == 0 and batch_idx % 5 == 0:
                    # Save input image
                    gif_dir = os.path.join(self.logger.log_dir, "gif", "val")
                    save_path = os.path.join(gif_dir, 'e_{:05d}_b_{:05d}_img.png'.format(self.current_epoch, batch_idx))
                    vutils.save_image(img[0], save_path, normalize=True)

                    # Visualise reconsturction, argmax to get the index
                    recon_code_h, recon_code_o = self.get_recon_code(outp_h, outp_o, batch_size=bs)
                    x_recon_h, x_recon_o = self.generate_vis(recon_code_h, recon_code_o, device=img.device)

                    mesh_h = load_mesh(mesh_path_h[0], color=utils.util.HAND_RGB).to(img.device)
                    mesh_o = load_mesh(mesh_path_o[0], color=utils.util.OBJ_RGB).to(img.device)
                    self.save_vis(mesh_h, mesh_o, x_recon_h, x_recon_o,
                                gif_dir=gif_dir, epoch=self.current_epoch, step=batch_idx)

                    # Draw proj 2d points
                    img_save = cv2.imread(save_path)
                    proj_2d = proj_2d[0].cpu().numpy()
                    for i in range(proj_2d.shape[0]):
                        x, y = proj_2d[i]
                        cv2.circle(img_save, (int(x), int(y)), 2, (0, 0, 255), -1)
                    save_path = os.path.join(gif_dir, 'e_{:05d}_b_{:05d}_proj.png'.format(self.current_epoch, batch_idx))
                    cv2.imwrite(save_path, img_save)
                    # Draw proj axis
                    img_save = cv2.imread(save_path)
                    proj_axis = proj_axis[0].cpu().numpy()
                    pairs = [[0,1], [0,2], [0,3]]
                    colors = [(0,0,255), (0,255,0), (255,0,0)]
                    for i, pair in enumerate(pairs):
                        x1, y1 = proj_axis[pair[0]]
                        x2, y2 = proj_axis[pair[1]]
                        cv2.line(img_save, (int(x1), int(y1)), (int(x2), int(y2)), colors[i], 2)
                    save_path = os.path.join(gif_dir, 'e_{:05d}_b_{:05d}_proj_axis.png'.format(self.current_epoch, batch_idx))
                    cv2.imwrite(save_path, img_save)
            except Exception as e:
                print(e)
                print("Error in saving gif")

    def forward(self, img, hand_pose=None, proj_mode='WeakPerspective', K=None, bbox=None):        
        outp, proj_2d, proj_axis = self.net(img, hand_pose, proj_mode, K, bbox)
        return outp, proj_2d, proj_axis
    
    def get_recon_code(self, outp_h, outp_o, batch_size=1, seq_len=512, cls=512):
        # outp_h is 32768, 512
        # reshape to cube
        recon_code_h = rearrange(outp_h, '(seq bs) cls -> seq bs cls', seq=seq_len, bs=batch_size)
        recon_code_h = rearrange(recon_code_h, '(d h w) bs cls -> bs d h w cls', d=self.grid_size, h=self.grid_size, w=self.grid_size, bs=batch_size)
        recon_code_o = rearrange(outp_o, '(seq bs) cls -> seq bs cls', seq=seq_len, bs=batch_size)
        recon_code_o = rearrange(recon_code_o, '(d h w) bs cls -> bs d h w cls', d=self.grid_size, h=self.grid_size, w=self.grid_size, bs=batch_size)

        # get the max index
        # Preserve gradients
        recon_code_h = torch.argmax(recon_code_h, dim=-1)
        recon_code_o = torch.argmax(recon_code_o, dim=-1)

        recon_code_h = rearrange(recon_code_h, 'bs d h w -> (d h w) bs')
        recon_code_o = rearrange(recon_code_o, 'bs d h w -> (d h w) bs')
        
        return recon_code_h, recon_code_o
    
    
    def decode_sdf(self, z_q, vis_xyz, ho_mode='hand', enc_indices=False):
        # Decode hand recon
        if enc_indices:
            z_q = z_q[:,0].unsqueeze(1)
        else:
            z_q = z_q[0].unsqueeze(0)
        vis_batch_size = 65000
        x_recon = []
        for i in range(0, vis_xyz.shape[1], vis_batch_size):
            start = i
            end = min(i+vis_batch_size, vis_xyz.shape[1])
            if ho_mode == 'hand':
                if enc_indices:
                    recon = self.vqvae_h.decode_enc_idices(z_q, z_spatial_dim=self.grid_size, xyz=vis_xyz[:,start:end])
                else:
                    recon = self.vqvae_h.decode(z_q, vis_xyz[:,start:end])
            elif ho_mode == 'object':
                if enc_indices:
                    recon = self.vqvae_o.decode_enc_idices(z_q, z_spatial_dim=self.grid_size, xyz=vis_xyz[:,start:end])
                else:
                    recon = self.vqvae_o.decode(z_q, vis_xyz[:,start:end])
            x_recon.append(recon)
            del recon
        x_recon = torch.cat(x_recon, dim=1)

        # Shape back into cube
        vgn_resolution = 40
        x_recon = rearrange(x_recon, 'b (p1 p2 p3) -> b p1 p2 p3', p1=vgn_resolution, p2=vgn_resolution, p3=vgn_resolution).unsqueeze(1)
        x_recon = x_recon / utils.util.SDF_MULTIPLIER

        return x_recon
    
    def generate_vis(self, code_h, code_o, device=None, enc_indices=True):
        assert device is not None

        with torch.no_grad():
            # Decode extracted latent cube to SDF
            vis_x = torch.linspace(-utils.util.BBOX_SIZE_X/2.0, utils.util.BBOX_SIZE_X/2.0, 40+1)[0:-1].to(device)
            vis_y = torch.linspace(-utils.util.BBOX_SIZE_Y/2.0, utils.util.BBOX_SIZE_Y/2.0, 40+1)[0:-1].to(device)
            vis_z = torch.linspace(-utils.util.BBOX_SIZE_Z/2.0, utils.util.BBOX_SIZE_Z/2.0, 40+1)[0:-1].to(device)

            vis_xyz = torch.meshgrid(vis_x, vis_y, vis_z)
            vis_xyz = torch.stack(vis_xyz, dim=-1).to(device)
            # Flatten
            vis_xyz = rearrange(vis_xyz, 'd h w c -> (d h w) c').unsqueeze(0)

            x_recon_h = self.decode_sdf(code_h, vis_xyz, ho_mode='hand', enc_indices=enc_indices)
            x_recon_o = self.decode_sdf(code_o, vis_xyz, ho_mode='object', enc_indices=enc_indices)
        
        return x_recon_h, x_recon_o
    
    def save_vis(self, mesh_h, mesh_o, x_recon_h, x_recon_o, level=0.0, gif_dir=None, save_gif=True, save_mesh=False,
                 epoch=0, step=0):
        with torch.no_grad():
            if mesh_h is not None and mesh_h is not None:
                try:
                    if save_gif:
                        p3d_mesh = pytorch3d.structures.join_meshes_as_scene([mesh_h, mesh_o])
                        if p3d_mesh is not None:
                            save_path = os.path.join(gif_dir, 'e_{:05d}_b_{:05d}_gt.gif'.format(epoch, step))
                            save_mesh_as_gif(self.renderer, p3d_mesh, nrow=1, out_name=save_path)
                except Exception as e:
                    print("Error saving original mesh")
                    print(e)
            if x_recon_h is not None and x_recon_o is not None:
                try:
                    p3d_mesh_h = sdf_to_mesh(x_recon_h, level=level, color=utils.util.HAND_RGB)
                    p3d_mesh_o = sdf_to_mesh(x_recon_o, level=level, color=utils.util.OBJ_RGB)
                    if save_mesh:
                        # Reconstructed from input codes
                        # Save p3d mesh as .obj
                        save_path_h = os.path.join(gif_dir, 'e_{:05d}_b_{:05d}_recon_h.obj'.format(epoch, step))
                        save_path_o = os.path.join(gif_dir, 'e_{:05d}_b_{:05d}_recon_o.obj'.format(epoch, step))
                        if p3d_mesh_h is not None:
                            IO().save_mesh(p3d_mesh_h, save_path_h)
                        if p3d_mesh_o is not None:
                            IO().save_mesh(p3d_mesh_o, save_path_o)
                    if save_gif:
                        p3d_mesh = pytorch3d.structures.join_meshes_as_scene([p3d_mesh_h, p3d_mesh_o])
                        if p3d_mesh is not None:
                            save_path = os.path.join(gif_dir, 'e_{:05d}_b_{:05d}_recon.gif'.format(epoch, step))
                            save_mesh_as_gif(self.renderer, p3d_mesh, nrow=1, out_name=save_path)
                except Exception as e:
                    print("Error saving recon mesh")
                    print(e)
    
    def test_recon(self, outp_h, outp_o, level=0.0, save_path=None, img_size=256, hand_pose=None, save_mesh=False, save_vis=False):
        outp_h = rearrange(outp_h, 'bs cls d h w -> (d h w) bs cls').contiguous() # to (T, B, cls)
        outp_h = rearrange(outp_h, 'seq bs cls -> (seq bs) cls')
        outp_o = rearrange(outp_o, 'bs cls d h w -> (d h w) bs cls').contiguous() # to (T, B, cls)
        outp_o = rearrange(outp_o, 'seq bs cls -> (seq bs) cls')

        recon_code_h, recon_code_o = self.get_recon_code(outp_h, outp_o, batch_size=1)

        p3d_mesh = None
        rendered_img = None
        tsdf_h = None
        tsdf_o = None

        if recon_code_h is not None and recon_code_o is not None:
            x_recon_h, x_recon_o = self.generate_vis(recon_code_h, recon_code_o, device=outp_h.device)
            tsdf_h = x_recon_h[0].unsqueeze(0)
            tsdf_o = x_recon_o[0].unsqueeze(0)

            if save_mesh:
                dist, elev, azim = 1.0, 0.0, 0.0
                img_renderer = init_mesh_renderer(img_size, dist=dist, elev=elev, azim=azim, device=self.opt.device)
                p3d_mesh_h = sdf_to_mesh(x_recon_h, level=level, color=utils.util.HAND_RGB)
                p3d_mesh_o = sdf_to_mesh(x_recon_o, level=level, color=utils.util.OBJ_RGB)
                p3d_mesh = pytorch3d.structures.join_meshes_as_scene([p3d_mesh_h, p3d_mesh_o])
            if save_vis and (p3d_mesh is not None):
                save_mesh_as_gif(self.renderer, p3d_mesh, nrow=1, out_name=save_path)
                rendered_img = save_mesh_as_img(img_renderer, p3d_mesh, hand_pose, save_path.replace('.gif', '_render.png'))
        
        return p3d_mesh, rendered_img, tsdf_h, tsdf_o

