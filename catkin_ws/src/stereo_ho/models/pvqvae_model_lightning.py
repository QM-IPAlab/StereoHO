import os
from collections import OrderedDict
import random as rand

import numpy as np
import einops
import mcubes
import omegaconf
from termcolor import colored
from einops import rearrange
from tqdm import tqdm
from pytorch3d.io import IO

import torch
from torch import nn, optim
from torch.profiler import record_function

import torchvision.utils as vutils
import torchvision.transforms as transforms

from models.base_model import BaseModel
from models.networks.pvqvae_networks.auto_encoder import PVQVAE
from models.networks.losses import VQLoss

import utils.util
import time
from utils.util_3d import init_mesh_renderer, render_sdf, sdf_to_mesh
from utils.qual_util import save_mesh_as_gif

import lightning.pytorch as pl


class PVQVAEModelLightning(pl.LightningModule):
    def __init__(self, opt, visualizer=None):
        super().__init__()

        # Read configs
        self.opt = opt
        self.mlp_decoder = opt.mlp_decoder
        configs = omegaconf.OmegaConf.load(opt.vq_cfg)
        mparam = configs.model.params
        n_embed = mparam.n_embed
        embed_dim = mparam.embed_dim
        ddconfig = mparam.ddconfig
        if self.mlp_decoder:
            decmlpconfig = mparam.decmlpconfig
        else:
            decmlpconfig = None

        # Setup hyper-params
        n_down = len(ddconfig.ch_mult) - 1
        resolution = configs.model.params.ddconfig['resolution']
        self.resolution = resolution
        nC = resolution
        self.cube_size = 2 ** n_down # patch_size
        self.stride = self.cube_size
        self.ncubes_per_dim = nC // self.cube_size
        # assert nC == 128, 'right now, only trained with sdf resolution = 128'
        # assert nC == 256, 'right now, only trained with sdf resolution = 256'
        assert (nC % self.cube_size) == 0, 'nC should be divisable by cube_size'

        # setup renderer
        dist, elev, azim = 0.8, 20, 20   
        self.renderer = init_mesh_renderer(image_size=512, dist=dist, elev=elev, azim=azim, device=self.opt.device)

        # Define model
        self.vqvae = PVQVAE(ddconfig, decmlpconfig, n_embed, embed_dim, self.mlp_decoder)

        # Define loss
        lossconfig = configs.lossconfig
        lossparams = lossconfig.params
        self.loss_vq = VQLoss(**lossparams).to(opt.device)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.vqvae.parameters(), lr=self.opt.lr, betas=(0.5, 0.9))
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.opt.lr_decay_epochs, 0.5,)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, self.opt.lr_decay_epochs, 1.0,)
        return [optimizer], [scheduler]

    @staticmethod
    def unfold_to_cubes(x, cube_size=8, stride=8):
        """ 
            assume x.shape: b, c, d, h, w 
            return: x_cubes: (b cubes)
        """
        x_cubes = x.unfold(2, cube_size, stride).unfold(3, cube_size, stride).unfold(4, cube_size, stride)
        x_cubes = rearrange(x_cubes, 'b c p1 p2 p3 d h w -> b c (p1 p2 p3) d h w')
        x_cubes = rearrange(x_cubes, 'b c p d h w -> (b p) c d h w')

        return x_cubes

    @staticmethod
    def fold_to_voxels(x_cubes, batch_size, ncubes_per_dim):
        x = rearrange(x_cubes, '(b p) c d h w -> b p c d h w', b=batch_size) 
        x = rearrange(x, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                        p1=ncubes_per_dim, p2=ncubes_per_dim, p3=ncubes_per_dim)
        return x

    def get_visxyz(self, res=40):
        # Decode extracted latent cube to SDF
        vis_x = torch.linspace(-utils.util.BBOX_SIZE_X/2.0, utils.util.BBOX_SIZE_X/2.0, res+1)[0:-1]
        vis_y = torch.linspace(-utils.util.BBOX_SIZE_Y/2.0, utils.util.BBOX_SIZE_Y/2.0, res+1)[0:-1]
        vis_z = torch.linspace(-utils.util.BBOX_SIZE_Z/2.0, utils.util.BBOX_SIZE_Z/2.0, res+1)[0:-1]

        vis_xyz = torch.meshgrid(vis_x, vis_y, vis_z)
        vis_xyz = torch.stack(vis_xyz, dim=-1)
        # Flatten
        vis_xyz = rearrange(vis_xyz, 'd h w c -> (d h w) c').unsqueeze(0)

        return vis_xyz

    def generate_vis(self, target_xyz, zq_voxels, res=40, vis_batch_size=64000):
        self.vqvae.eval()
        with torch.no_grad():
            # print("======== Generating vis...")
            # recon for visualization
            vis_xyz = self.get_visxyz(res=res).to(target_xyz.device)

            # # Separate into small batches of size 20000 and pass to decoder
            # vis_batch_size = 32000
            self.x_recon_vis = []
            for i in range(0, vis_xyz.shape[1], vis_batch_size):
                start = i
                end = min(i+vis_batch_size, vis_xyz.shape[1])
                recon = self.vqvae.decode(zq_voxels, vis_xyz[:,start:end])
                self.x_recon_vis.append(recon)
                del recon
            self.x_recon_vis = torch.cat(self.x_recon_vis, dim=1)

            # Shape back into cube
            self.x_recon_vis = rearrange(self.x_recon_vis, 'b (p1 p2 p3) -> b p1 p2 p3', p1=res, p2=res, p3=res).unsqueeze(1)
            self.x_recon_vis = self.x_recon_vis / utils.util.SDF_MULTIPLIER
        self.vqvae.train()

        return self.x_recon_vis

    def save_vis(self, input_sdf, recon_vis, level=0.00, gif_dir=None, save_gif=True, epoch=0, step=0):
        with torch.no_grad():
            self.image = torch.zeros(1, 4, 512, 512).to(input_sdf.device)
            self.image_recon = torch.zeros(1, 4, 512, 512).to(input_sdf.device)

            print("input_sdf.shape: ", input_sdf.shape)
            print("recon_vis.shape: ", recon_vis.shape)
            # Upsample to 256 using tri-linear interpolation
            input_sdf = nn.functional.interpolate(input_sdf, size=256, mode='trilinear', align_corners=False)
            recon_vis = nn.functional.interpolate(recon_vis, size=256, mode='trilinear', align_corners=False)
            print("input_sdf.shape: ", input_sdf.shape)
            print("recon_vis.shape: ", recon_vis.shape)

            p3d_mesh = sdf_to_mesh(input_sdf/utils.util.SDF_MULTIPLIER, level=level)
            # Save p3d mesh as .obj
            save_path = os.path.join(gif_dir, 'e_{:05d}_b_{:05d}_orig.obj'.format(epoch, step))
            if p3d_mesh is not None:
                IO().save_mesh(p3d_mesh, save_path)
            if save_gif:
                if p3d_mesh is not None:
                    save_path = os.path.join(gif_dir, 'e_{:05d}_b_{:05d}_orig.gif'.format(epoch, step))
                    save_mesh_as_gif(self.renderer, p3d_mesh, nrow=1, out_name=save_path)

            # Reconstructed
            p3d_mesh = sdf_to_mesh(recon_vis/utils.util.SDF_MULTIPLIER, level=level)
            # Save p3d mesh as .obj
            save_path = os.path.join(gif_dir, 'e_{:05d}_b_{:05d}_recon.obj'.format(epoch, step))
            if p3d_mesh is not None:
                IO().save_mesh(p3d_mesh, save_path)
            if save_gif:
                if p3d_mesh is not None:
                    save_path = os.path.join(gif_dir, 'e_{:05d}_b_{:05d}_recon.gif'.format(epoch, step))
                    save_mesh_as_gif(self.renderer, p3d_mesh, nrow=1, out_name=save_path)
    
    def get_codebook_weight(self):
        ret = self.vqvae.quantize.embedding.state_dict()
        # self.vqvae.quantize.embedding.cuda()
        return ret
    
    def forward(self, input_sdf_cubes, target_xyz, cur_bs, cb_restart=False):
        recon_sdf, zq_voxels, qloss, qinfo = self.vqvae(input_sdf_cubes, target_xyz, cur_bs, self.ncubes_per_dim, cb_restart=cb_restart)
        return recon_sdf, zq_voxels, qloss, qinfo

    def extract_code(self, batch):
        self.vqvae.eval()
        print('==============================')
        # print("batch_idx: ", batch_idx)
        input_sdf = batch['input_sdf'].cuda()
        cur_bs = input_sdf.shape[0]
        print("input_sdf.shape: ", input_sdf.shape) # B x C x D x H x W
        input_sdf_cubes = self.unfold_to_cubes(input_sdf, self.cube_size, self.stride)
        print("input_sdf_cubes.device: ", input_sdf_cubes.device)
        print("input_sdf_cubes.shape: ", input_sdf_cubes.shape) # Cube_id x C x CubeD x CubeH x CubeW
        # make sure it has the same name as forward
        with torch.no_grad():
            zq_cubes, _, qinfo = self.vqvae.encode(input_sdf_cubes, mix_cb=False, cb_restart=False)
            zq_voxels = self.fold_to_voxels(zq_cubes, batch_size=cur_bs, ncubes_per_dim=self.ncubes_per_dim)

        self.vqvae.train()
        return zq_voxels, qinfo

    def training_step(self, batch, batch_idx):
        input_sdf = batch['input_sdf']
        cur_bs = input_sdf.shape[0]
        input_sdf_cubes = self.unfold_to_cubes(input_sdf, self.cube_size, self.stride)
        print("================DEBUG================")
        print("input_sdf_cubes.shape: ", input_sdf_cubes.shape)
        print("input_sdf.shape: ", input_sdf.shape)
        print("input_sdf max: ", input_sdf.max())
        print("input_sdf min: ", input_sdf.min())
        print("input_sdf mean: ", input_sdf.mean())
        # Targets
        if self.mlp_decoder:
            target_xyz = batch['target_sdf'][:,:,0:3]
            target_sdf = batch['target_sdf'][:,:,3]
        else:
            target_xyz = None
            target_sdf = input_sdf
        print("target_sdf.shape: ", target_sdf.shape)
        print("target_sdf max: ", target_sdf.max())
        print("target_sdf min: ", target_sdf.min())

        # Forward
        recon_sdf, zq_voxels, qloss, qinfo = self.forward(input_sdf_cubes, target_xyz, cur_bs, cb_restart=False)
        aeloss, loss_log = self.loss_vq(qloss, target_sdf, recon_sdf)

        self.log("train/aeloss", aeloss, sync_dist=True)
        self.log("train/qloss", qloss, sync_dist=True)
        self.log("train/loss_rec", loss_log['loss_rec'], sync_dist=True)
        self.log("train/num_unique", qinfo[3], sync_dist=True)
        self.log("train/total_usage", qinfo[5], sync_dist=True)


        # Only on rank 0
        print("Global rank: ", self.global_rank)
        if self.global_rank == 0:
            idx = rand.randint(0, cur_bs-1)
            if self.global_step % self.opt.display_freq == 0 and self.global_step > 0:
                if self.mlp_decoder:
                    recon_vis = self.generate_vis(target_xyz, zq_voxels[0].unsqueeze(0))
                else:
                    recon_vis = recon_sdf[idx].unsqueeze(0)
                    print("recon_vis: ", recon_vis)
                    print("input_sdf: ", input_sdf[idx].unsqueeze(0))
                    print("recon_vis.shape: ", recon_vis.shape)
                    print("recon_vis max: ", recon_vis.max())
                    print("recon_vis min: ", recon_vis.min())
                    print("recon_vis mean: ", recon_vis.mean())
                    print("input_sdf.shape: ", input_sdf[idx].unsqueeze(0).shape)
                    print("input_sdf max: ", input_sdf[idx].unsqueeze(0).max())
                    print("input_sdf min: ", input_sdf[idx].unsqueeze(0).min())
                    print("input_sdf mean: ", input_sdf[idx].unsqueeze(0).mean())
                
                gif_dir = os.path.join(self.logger.log_dir, "gif",  "train")
                self.save_vis(input_sdf[idx].unsqueeze(0), recon_vis, gif_dir=gif_dir, epoch=self.current_epoch, step=batch_idx)

        return aeloss

    def validation_step(self, batch, batch_idx):
        start = time.time()
        input_sdf = batch['input_sdf'] # B x C x D x H x W
        cur_bs = input_sdf.shape[0]
        input_sdf_cubes = self.unfold_to_cubes(input_sdf, self.cube_size, self.stride)

        # Targets
        if self.mlp_decoder:
            # target_xyz = batch['target_sdf'][:,:,0:3]
            # target_sdf = batch['target_sdf'][:,:,3]
            target_xyz = self.get_visxyz(res=self.resolution)
            print("target_xyz.shape: ", target_xyz.shape)
            target_xyz = target_xyz.repeat(cur_bs, 1, 1).to(input_sdf.device)
            target_sdf = rearrange(input_sdf, 'b c d h w -> b (d h w) c').squeeze(-1)
            target_xyz = target_xyz[:,::4,:]
            target_sdf = target_sdf[:,::4]
            print("target_sdf.shape: ", target_sdf.shape)
            print("target_xyz.shape: ", target_xyz.shape)

        else:
            target_xyz = None
            target_sdf = input_sdf
        
        print("target_sdf.shape: ", target_sdf.shape)

        # Forward
        recon_sdf, zq_voxels, qloss, qinfo = self.forward(input_sdf_cubes, target_xyz, cur_bs, cb_restart=False)
        print("Time to forward pass: ", time.time() - start)
        print("recon_sdf.shape: ", recon_sdf.shape)
        start = time.time()
        aeloss, loss_log = self.loss_vq(qloss, target_sdf, recon_sdf)
        print("Time to compute loss: ", time.time() - start)
        self.log("val/aeloss", aeloss, sync_dist=True)
        self.log("val/qloss", qloss, sync_dist=True)
        self.log("val/loss_rec", loss_log['loss_rec'], sync_dist=True)
        self.log("val/num_unique", qinfo[3], sync_dist=True)
        self.log("val/total_usage", qinfo[5], sync_dist=True)


        # Only on rank 0
        print("Global rank: ", self.global_rank)
        if self.global_rank == 0:
            if self.current_epoch % 1 == 0 and batch_idx % 20 == 0:
                # time the forward pass in seconds
                start = time.time()
                if self.mlp_decoder:
                    recon_vis = self.generate_vis(target_xyz, zq_voxels[0].unsqueeze(0))
                else:
                    recon_vis = recon_sdf[0].unsqueeze(0)
                print("Time to generate vis: ", time.time() - start)
                gif_dir = os.path.join(self.logger.log_dir, "gif", "val")
                start = time.time()
                self.save_vis(input_sdf[0].unsqueeze(0), recon_vis, gif_dir=gif_dir, epoch=self.current_epoch, step=batch_idx)
                print("Time to save vis: ", time.time() - start)

        return aeloss
        

