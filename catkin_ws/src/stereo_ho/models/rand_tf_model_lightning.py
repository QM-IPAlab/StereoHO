import os
from collections import OrderedDict
import traceback

import numpy as np
import einops
import mcubes
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import rearrange, repeat
from tqdm import tqdm
from pytorch3d.io import IO
from pytorch3d.transforms import Transform3d
import pytorch3d
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn.functional as F
from torch import nn, optim

import torchvision.utils as vutils
import torchvision.transforms as transforms

from models.base_model import BaseModel
from models.networks.transformer_networks.rand_transformer import RandTransformer, RandTransformerPE
from models.networks.pvqvae_networks.auto_encoder import PVQVAE

import utils.util
import time
from utils.util_3d import init_mesh_renderer, render_sdf, sdf_to_mesh, ho_sdf_to_mesh, load_mesh
from utils.qual_util import save_mesh_as_gif
from datasets.obman_dataset import load_grid_sdf
from models.networks.losses import FocalLoss

import lightning.pytorch as pl


class RandTransformerModelLightning(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        assert opt.tf_cfg is not None
        assert opt.vq_cfg is not None

        # load configs for tf and vq
        self.opt = opt
        self.two_stage = opt.two_stage
        tf_conf = OmegaConf.load(opt.tf_cfg)
        vq_conf = OmegaConf.load(opt.vq_cfg)
        self.empty_idx_h = opt.empty_idx_h
        self.empty_idx_o = opt.empty_idx_o
        self.grid_h_val = -0.7
        self.grid_o_val = 0.7
        self.grid_val_noise = 0.2
        self.grid_xyz_noise = 0.25/8.0

        # init tf model
        self.tf = RandTransformer(tf_conf, vq_conf=vq_conf, two_stage=opt.two_stage,
                                  empty_idx_h=self.empty_idx_h, empty_idx_o=self.empty_idx_o)
        # self.tf = RandTransformerPE(tf_conf, vq_conf=vq_conf)

        # init vqvae for decoding shapes
        mparam = vq_conf.model.params
        n_embed = mparam.n_embed
        embed_dim = mparam.embed_dim
        ddconfig = mparam.ddconfig
        decmlpconfig = mparam.decmlpconfig
        self.cb_size = n_embed

        n_down = len(ddconfig.ch_mult) - 1

        self.load_vqvae(ddconfig, decmlpconfig, n_embed, embed_dim)

        # modify the tf's embedding to be the codebook learned from vqvae
        if self.opt.ho_mode == 'joint' or self.opt.ho_mode == 'hand':
            self.embedding_encoder_h = nn.Embedding(n_embed, embed_dim)
            self.embedding_encoder_h.load_state_dict(self.vqvae_h.quantize.embedding.state_dict())
            # self.embedding_encoder_h.requires_grad = False
            for param in self.embedding_encoder_h.parameters():
                param.requires_grad = False
        if self.opt.ho_mode == 'joint' or self.opt.ho_mode == 'object':
            self.embedding_encoder_o = nn.Embedding(n_embed, embed_dim)
            self.embedding_encoder_o.load_state_dict(self.vqvae_o.quantize.embedding.state_dict())
            # self.embedding_encoder_o.requires_grad = False
            for param in self.embedding_encoder_o.parameters():
                param.requires_grad = False
        
        # Embedding for start token
        self.embedding_start = nn.Embedding(1, embed_dim)
        self.embedding_start.weight.data.uniform_(-0.01, 0.01)

        # define loss functions
        self.criterion_ce = nn.CrossEntropyLoss()

        # define loss functions
        alpha = opt.focal_alpha
        weight_h = torch.ones(512).to(opt.device) * (1.0 - alpha)
        weight_h[self.empty_idx_h] = alpha
        weight_o = torch.ones(512).to(opt.device) * (1.0 - alpha)
        weight_o[self.empty_idx_o] = alpha
        self.criterion_cce_h = nn.CrossEntropyLoss(weight=weight_h)
        self.criterion_cce_o = nn.CrossEntropyLoss(weight=weight_o)
        w_empty = opt.binary_empty_weight
        weight_occp = torch.tensor([w_empty, 1-w_empty]).to(opt.device)
        self.criterion_cce_occp = nn.CrossEntropyLoss(weight=weight_occp)
        
        alpha_h = weight_h.tolist()
        alpha_o = weight_o.tolist()
        self.criterion_focal_h = FocalLoss(gamma=opt.focal_gamma, alpha=alpha_h)
        self.criterion_focal_o = FocalLoss(gamma=opt.focal_gamma, alpha=alpha_o)

        if opt.loss == 'ce':
            self.criterion_h = self.criterion_ce
            self.criterion_o = self.criterion_ce
        elif opt.loss == 'cce':
            self.criterion_h = self.criterion_cce_h
            self.criterion_o = self.criterion_cce_o
        elif opt.loss == 'focal':
            self.criterion_h = self.criterion_focal_h
            self.criterion_o = self.criterion_focal_o
        # elif opt.loss == 'hybrid':
        #     self.criterion_h = self.criterion_ce
        #     self.criterion_o = self.criterion_cce_o

        resolution = tf_conf.data.resolution
        self.resolution = resolution

        # start token
        self.sos = 0
        self.counter = 0

        # init grid for lookup
        pe_conf = tf_conf.pe
        self.grid_size = pe_conf.zq_dim
        self.grid_table = self.init_grid(pos_dim=pe_conf.pos_dim, zq_dim=self.grid_size)

        # setup hyper-params 
        nC = resolution
        self.cube_size = 2 ** n_down # patch_size
        self.stride = self.cube_size
        self.ncubes_per_dim = nC // self.cube_size
        assert nC == 128, 'right now, only trained with sdf resolution = 64'
        assert (nC % self.cube_size) == 0, 'nC should be divisable by cube_size'

        # setup renderer
        dist, elev, azim = 0.8, 20, 20   
        self.renderer = init_mesh_renderer(image_size=512, dist=dist, elev=elev, azim=azim, device=self.opt.device)


    def load_vqvae(self, ddconfig, decmlpconfig, n_embed, embed_dim):

        if self.opt.ho_mode == 'joint' or self.opt.ho_mode == 'hand':
            assert type(self.opt.vq_ckpt_h) == str
            self.vqvae_h = PVQVAE(ddconfig, decmlpconfig, n_embed, embed_dim, self.opt.mlp_decoder)

            state_dict = torch.load(self.opt.vq_ckpt_h)['state_dict']
            # Remove vqvae from name
            state_dict = {k.replace('vqvae.', ''): v for k, v in state_dict.items()}
            self.vqvae_h.load_state_dict(state_dict)
            print(colored('[*] VQVAE: weight successfully load from: %s' % self.opt.vq_ckpt_h, 'blue'))
            
            self.vqvae_h.eval()
            for param in self.vqvae_h.parameters():
                param.requires_grad = False
        
        if self.opt.ho_mode == 'joint' or self.opt.ho_mode == 'object':
            assert type(self.opt.vq_ckpt_o) == str
            self.vqvae_o = PVQVAE(ddconfig, decmlpconfig, n_embed, embed_dim, self.opt.mlp_decoder)

            state_dict = torch.load(self.opt.vq_ckpt_o)['state_dict']
            # Remove vqvae from name
            state_dict = {k.replace('vqvae.', ''): v for k, v in state_dict.items()}
            self.vqvae_o.load_state_dict(state_dict)
            print(colored('[*] VQVAE: weight successfully load from: %s' % self.opt.vq_ckpt_o, 'blue'))
            
            self.vqvae_o.eval()
            for param in self.vqvae_o.parameters():
                param.requires_grad = False


    def init_grid(self, pos_dim=3, zq_dim=8):
        x = torch.linspace(-1, 1, zq_dim)
        y = torch.linspace(-1, 1, zq_dim)
        if pos_dim == 3:
            z = torch.linspace(-1, 1, zq_dim)
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
            grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
            grid_table = grid.view(-1, pos_dim)
            pos_sos = torch.tensor([-1., -1., -1-2/zq_dim]).float().unsqueeze(0)
        elif pos_dim == 4:
            z = torch.linspace(-1, 1, zq_dim)
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
            grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
            grid_h = grid.clone()
            grid_h = grid_h.view(-1, 3)
            grid_h_val = torch.ones(grid_h.shape[0], 1)*self.grid_h_val
            grid_h = torch.cat([grid_h, grid_h_val], dim=-1)
            grid_o = grid.clone()
            grid_o = grid_o.view(-1, 3)
            grid_o_val = torch.ones(grid_o.shape[0], 1)*self.grid_o_val
            grid_o = torch.cat([grid_o, grid_o_val], dim=-1)
            grid_table = torch.cat([grid_h, grid_o], dim=0)
            pos_sos = torch.tensor([-1., -1., -1., -1-2/zq_dim]).float().unsqueeze(0)
        else:
            grid_x, grid_y = torch.meshgrid(x, y)
            grid = torch.stack([grid_x, grid_y], dim=-1)
            grid_table = grid.view(-1, pos_dim)
            pos_sos = torch.tensor([-1., -1-2/zq_dim]).float().unsqueeze(0)

        grid_table = torch.cat([pos_sos, grid_table], dim=0)
        return grid_table


    def configure_optimizers(self):
        tf_params = [p for p in self.tf.parameters() if p.requires_grad == True]
        encoder_start_params = [p for p in self.embedding_start.parameters() if p.requires_grad == True]
        all_params = tf_params + encoder_start_params
        optimizer = optim.Adam(all_params, lr=self.opt.lr)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, 50, 0.5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 50, 1.0)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 1.0)

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
    

    def decode_sdf(self, z_q, vis_xyz, ho_mode='hand', vis_batch_size=20000, enc_indices=False):
        assert enc_indices == False, 'enc_indices not implemented'
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
        x_recon = rearrange(x_recon, 'b (p1 p2 p3) -> b p1 p2 p3', p1=self.resolution, p2=self.resolution, p3=self.resolution).unsqueeze(1)
        x_recon = x_recon / utils.util.SDF_MULTIPLIER

        return x_recon
    

    def get_visxyz(self):
        # Decode extracted latent cube to SDF
        vis_x = torch.linspace(-utils.util.BBOX_SIZE_X/2.0, utils.util.BBOX_SIZE_X/2.0, self.resolution+1)[0:-1]
        vis_y = torch.linspace(-utils.util.BBOX_SIZE_Y/2.0, utils.util.BBOX_SIZE_Y/2.0, self.resolution+1)[0:-1]
        vis_z = torch.linspace(-utils.util.BBOX_SIZE_Z/2.0, utils.util.BBOX_SIZE_Z/2.0, self.resolution+1)[0:-1]

        vis_xyz = torch.meshgrid(vis_x, vis_y, vis_z)
        vis_xyz = torch.stack(vis_xyz, dim=-1)
        # Flatten
        vis_xyz = rearrange(vis_xyz, 'd h w c -> (d h w) c').unsqueeze(0)

        return vis_xyz


    def inference(self, T, B, O, inp_pos, tgt_pos, gen_order, img_cond=None, seq_len=1, topk=30, alpha_h=0.5, alpha_o=0.5, img_cond_len=0):
        def top_k_logits(logits, k=5):
            v, ix = torch.topk(logits, k)
            out = logits.clone()
            out[out < v[:, :, [-1]]] = -float('Inf')
            return out

        self.freeze_all()
        
        with torch.no_grad():
            # auto-regressively gen
            pred_idx = torch.zeros((T, B), dtype=torch.long).to(O.device)
            pred = O
            occp_cnt = 0
            for t in tqdm(range(seq_len, T), total=T-seq_len, desc='[*] autoregressively inferencing...'):
                _, _, outp = self.tf(pred, inp_pos[:t], tgt_pos[:t])
                outp_t = outp[-1:]
                
                outp_t = F.log_softmax(outp_t, dim=-1)

                ho_mode = 'hand' if tgt_pos[t-1][3] < 0.0 else 'object'
                if img_cond is not None:
                    # scale alpha from alpha_high to alpha_low
                    # alpha = alpha_low + (alpha_high-alpha_low) * (1. - (float(t)/float(T-seq_len)))
                    # alpha = alpha_h if tgt_pos[t-1][3] < 0.0 else alpha_o
                    # pred_img = img_cond[t-1:t,:,:]
                    # pred_img = F.softmax(pred_img, dim=-1)
                    # pred_img = torch.argmax(pred_img, dim=-1)
                    # pred_img = pred_img.squeeze(0)
                    # if ho_mode == 'hand':
                    #     if pred_img == self.empty_idx_h:
                    #         alpha = 1.0
                    #     else:
                    #         alpha = alpha_h
                    # elif ho_mode == 'object':
                    #     if pred_img == self.empty_idx_o:
                    #         alpha = 1.0
                    #     else:
                    #         alpha = alpha_o
                    if occp_cnt > img_cond_len:
                        # print("t: ", t)
                        if ho_mode == 'hand':
                            alpha = alpha_h
                        elif ho_mode == 'object':
                            alpha = alpha_o
                    else:
                        alpha = 1.0

                    outp_t = (1-alpha)*outp_t + alpha*img_cond[t-1:t,:,:]

                if topk is not None:
                    outp_t = top_k_logits(outp_t, k=topk)

                outp_t = F.softmax(outp_t, dim=-1) # compute prob
                outp_t = rearrange(outp_t, 't b nc -> (t b) nc')
                pred_t = torch.multinomial(outp_t, num_samples=1).squeeze(1)
                pred_t = rearrange(pred_t, '(t b) -> t b', t=1, b=B)
                pred_idx[t-1] = pred_t
                if ho_mode == 'hand':
                    if img_cond is not None and pred_t != self.empty_idx_h:
                        occp_cnt += 1
                    pred_t = self.embedding_encoder_h(pred_t)
                elif ho_mode == 'object':
                    if img_cond is not None and pred_t != self.empty_idx_o:
                        occp_cnt += 1
                    pred_t = self.embedding_encoder_o(pred_t)
                pred = torch.cat([pred, pred_t], dim=0)
        # Separate hand and object
        pred = pred[1:][torch.argsort(gen_order)] # exclude pred[0] since it's <sos>
        pred = pred[:,0].unsqueeze(1) ### ????????????????
        pred_h = pred[:int((T-1)/2), :]
        pred_o = pred[int((T-1)/2):, :]
        pred_h = rearrange(pred_h, '(d1 d2 d3) b c -> b c d1 d2 d3', d1=self.grid_size, d2=self.grid_size, d3=self.grid_size)
        pred_o = rearrange(pred_o, '(d1 d2 d3) b c -> b c d1 d2 d3', d1=self.grid_size, d2=self.grid_size, d3=self.grid_size)
        pred_idx = pred_idx[torch.argsort(gen_order)]
        # print("occp_cnt: ", occp_cnt)

        return pred_h, pred_o, pred_idx


    def generate_vis(self, code_h, code_o, vis_xyz):
        # Expected shape: B x C x D x H x W
        # only visualize the first one
        code_h = code_h[0].unsqueeze(0)
        code_o = code_o[0].unsqueeze(0)

        x_recon_h = self.decode_sdf(code_h, vis_xyz, ho_mode='hand')
        x_recon_o = self.decode_sdf(code_o, vis_xyz, ho_mode='object')

        return x_recon_h, x_recon_o
    

    def image_recon(self, img_cond, output_dir, img, id, mesh_path_h, mesh_path_o, hand_pose=None, alpha_h=0.5, alpha_o=0.5,
                    topk=1, seq_len=1, sample_method='hand_pose', save_gif=False, codeidx_h=None, codeidx_o=None, img_cond_len=5):

        self.freeze_all()

        output_dir = os.path.join(output_dir, id)
        os.makedirs(output_dir, exist_ok=True)
        # save image
        vutils.save_image(img, os.path.join(output_dir, 'img.jpg'), normalize=True, nrow=1)

        # generate SDF
        # img_cond: B x 2*T x D x H x W
        img_cond_orig = img_cond.clone()
        device = img.device
        T = 2*(img_cond.shape[2] * img_cond.shape[3] * img_cond.shape[4]) + 1
        B = img_cond.shape[0]

        if sample_method=='hand_pose' and hand_pose is not None:
            # To order according to camera pose
            # Points closer to camera are reconstructed first
            print('Ordering according to hand pose')
            gen_order = self.get_gen_order(T-1, mode='hand_pose', hand_pose=hand_pose, device=device)

        elif sample_method=='confidence':
            # To order according to confidence
            # Points with higher confidence are reconstructed first
            print('Ordering according to confidence')
            cond_h = img_cond[:,:self.cb_size,:,:,:]
            cond_o = img_cond[:,self.cb_size:,:,:,:]
            cond_h = rearrange(cond_h, 'b t d h w -> (d h w) (b t)')
            cond_o = rearrange(cond_o, 'b t d h w -> (d h w) (b t)')
            cond_all = torch.cat([cond_h, cond_o], dim=0)
            cond_all = F.softmax(cond_all, dim=1)
            cond_all = torch.max(cond_all, dim=1)[0]

            # Ascending variance
            # var = torch.var(cond_all, dim=1)
            # gen_order = torch.argsort(cond_all, descending=True).to(device)

            # Descending confidence
            confidence, gen_order = torch.sort(cond_all, descending=True)
        else:
            # Random order
            print('Ordering randomly')
            gen_order = self.get_gen_order(T-1, mode='random')

        # Get log prob from image condition
        img_cond = rearrange(img_cond, 'bs cb dz hz wz -> (dz hz wz) bs cb').contiguous() # 512 x B x cb_size
        img_cond_h = img_cond[:,:,:self.cb_size] # 512 x B x cb_size
        img_cond_o = img_cond[:,:,self.cb_size:] # 512 x B x cb_size
        img_cond_joint = torch.cat([img_cond_h, img_cond_o], dim=0) # 1024 x B x cb_size
        img_cond = img_cond_joint[gen_order, :, :] # 1024 x B x cb_size
        img_cond = F.log_softmax(img_cond, dim=-1)

        # Fill inp val with placeholders
        sos = torch.LongTensor(1, B).fill_(self.sos).to(device)
        code_sos = self.embedding_start(sos)
        inp_val = code_sos.repeat(T-1, 1, 1)
        O = inp_val[:seq_len]

        # Build inp pos and tgt pos according to gen_order
        grid_table = self.grid_table.clone().to(device)
        grid_table = torch.cat([grid_table[:1], grid_table[1:][gen_order]], dim=0)
        inp_pos = grid_table[:-1].clone() # T x 4 (sos, h, o)
        tgt_pos = grid_table[1:].clone()

        # GT
        codeidx_h = rearrange(codeidx_h, 'bs d h w -> (d h w) bs').contiguous()
        codeidx_o = rearrange(codeidx_o, 'bs d h w -> (d h w) bs').contiguous()
        gt = torch.cat([codeidx_h, codeidx_o], dim=0)

        with torch.no_grad():

            vis_xyz = self.get_visxyz().to(device)
            
            pred_h, pred_o, pred_idx = self.inference(T, B, O, inp_pos, tgt_pos, gen_order, img_cond=img_cond, seq_len=seq_len, topk=topk, alpha_h=alpha_h, alpha_o=alpha_o, img_cond_len=img_cond_len)
            x_recon_tf_h, x_recon_tf_o = self.generate_vis(pred_h, pred_o, vis_xyz)

            # Get image only SDF
            # Decode extracted latent cube to SDF
            outp = rearrange(img_cond_orig, 'bs cls d h w -> (d h w) bs cls').contiguous() # (1, 1024, 8, 8, 8) -> (512, 1, 1024)
            outp = rearrange(outp, 'seq bs cls -> (seq bs) cls') # (512, 1024)
            outp_h = outp[:, :self.cb_size] # (512, 512)
            outp_o = outp[:, self.cb_size:] # (512, 512)
            recon_code_h = rearrange(outp_h, '(seq bs) cls -> seq bs cls', seq=int((T-1)/2), bs=B) # (512, 1, 512)
            recon_code_h = rearrange(recon_code_h, '(d h w) bs cls -> bs d h w cls', d=self.grid_size, h=self.grid_size, w=self.grid_size, bs=B) # (1, 8, 8, 8, 512)
            recon_code_o = rearrange(outp_o, '(seq bs) cls -> seq bs cls', seq=int((T-1)/2), bs=B)
            recon_code_o = rearrange(recon_code_o, '(d h w) bs cls -> bs d h w cls', d=self.grid_size, h=self.grid_size, w=self.grid_size, bs=B)
            # get the max index
            recon_code_h = torch.argmax(recon_code_h, dim=-1) # (1, 8, 8, 8)
            recon_code_o = torch.argmax(recon_code_o, dim=-1)
            recon_code_h = rearrange(recon_code_h, 'bs d h w -> (d h w) bs') # (512, 1)
            recon_code_o = rearrange(recon_code_o, 'bs d h w -> (d h w) bs')

            # Accuracy
            recon_code = torch.cat([recon_code_h, recon_code_o], dim=0)
            acc = torch.sum(recon_code == gt).float() / recon_code.shape[0]
            print("image acc: ", acc)
            acc = torch.sum(pred_idx == gt).float() / pred_idx.shape[0]
            print("combined acc: ", acc)
            # get the embedding
            recon_code_h = self.embedding_encoder_h(recon_code_h) # (512, 1, 512)
            recon_code_o = self.embedding_encoder_o(recon_code_o)
            # rearrange to (bs, c, d, h, w)
            recon_code_h = rearrange(recon_code_h, '(d h w) bs c -> bs c d h w', bs=B, d=self.grid_size, h=self.grid_size, w=self.grid_size) # (1, 512, 8, 8, 8)
            recon_code_o = rearrange(recon_code_o, '(d h w) bs c -> bs c d h w', bs=B, d=self.grid_size, h=self.grid_size, w=self.grid_size)
            # get the sdf
            x_recon_h, x_recon_o = self.generate_vis(recon_code_h, recon_code_o, vis_xyz)
        
            # Save visualizations
            if save_gif:
                mesh_h = load_mesh(mesh_path_h, color=utils.util.HAND_RGB).to(img.device)
                mesh_o = load_mesh(mesh_path_o, color=utils.util.OBJ_RGB).to(img.device)
                self.save_mesh_vis(mesh_h, mesh_o, output_dir)
            self.save_sdf_vis(x_recon_h, x_recon_o, save_dir=output_dir, save_gif=save_gif, save_mesh=True, prefix='recon')
            self.save_sdf_vis(x_recon_tf_h, x_recon_tf_o, save_dir=output_dir, save_gif=save_gif, save_mesh=True, prefix='recon_tf')
    

    def save_mesh_vis(self, mesh_h, mesh_o, save_dir=None, save_gif=True, prefix='gt'):
        with torch.no_grad():
            try:
                p3d_mesh = pytorch3d.structures.join_meshes_as_scene([mesh_h, mesh_o])

                if save_gif:
                    save_path = os.path.join(save_dir, '{}.gif'.format(prefix))
                    save_mesh_as_gif(self.renderer, p3d_mesh, nrow=1, out_name=save_path)

            except Exception as e:
                print("Error saving original mesh")
                print(e)


    def save_sdf_vis(self, x_recon_h, x_recon_o, level=0.0, save_dir=None, save_gif=True, save_mesh=False, prefix='recon'):
        with torch.no_grad():
            try:
                # Reconstructed from image codes
                p3d_mesh_h = sdf_to_mesh(x_recon_h, level=level)
                p3d_mesh_o = sdf_to_mesh(x_recon_o, level=level)
                
                # Save p3d mesh as .obj
                if save_mesh:
                    save_path_h = os.path.join(save_dir, '{}_h.obj'.format(prefix))
                    save_path_o = os.path.join(save_dir, '{}_o.obj'.format(prefix))
                    IO().save_mesh(p3d_mesh_h, save_path_h)
                    IO().save_mesh(p3d_mesh_o, save_path_o)

                if save_gif:
                    p3d_mesh = pytorch3d.structures.join_meshes_as_scene([p3d_mesh_h, p3d_mesh_o])
                    save_path = os.path.join(save_dir, '{}.gif'.format(prefix))
                    save_mesh_as_gif(self.renderer, p3d_mesh, nrow=1, out_name=save_path)

            except Exception as e:
                print("Error saving {} mesh".format(prefix))
                print(e)


    def conditional_generation(self, sdf_h, sdf_o, cond="hand", step=0, gif_dir=None, extra_cond=0):
        import time
        bs = sdf_h.shape[0]
        sdf_h = self.unfold_to_cubes(sdf_h, self.cube_size, self.stride)
        sdf_o = self.unfold_to_cubes(sdf_o, self.cube_size, self.stride)
        
        # Encode SDF
        enc_h, _, info_h = self.vqvae_h.encode(sdf_h)
        enc_o, _, info_o = self.vqvae_o.encode(sdf_o)
        
        # Prepare input for transformer, transformer should only see condition (hand or object)
        if cond == "hand":
            inp_h = enc_h.squeeze(2).squeeze(2).squeeze(2).unsqueeze(1)
            inp_o = enc_o.squeeze(2).squeeze(2).squeeze(2).unsqueeze(1)
        elif cond == "object":
            inp_o = enc_o.squeeze(2).squeeze(2).squeeze(2).unsqueeze(1)
            inp_h = enc_h.squeeze(2).squeeze(2).squeeze(2).unsqueeze(1)
        
        seq_len = 512
        if cond == "hand":
            gen_order = torch.cat([torch.range(0, seq_len-1).long(), torch.range(seq_len, seq_len*2-1).long()[torch.randperm(seq_len)]], dim=0)
        elif cond == "object":
            gen_order = torch.cat([torch.range(seq_len, seq_len*2-1).long(), torch.range(0, seq_len-1).long()[torch.randperm(seq_len)]], dim=0)
        assert gen_order.shape[0] == seq_len*2

        sos = torch.LongTensor(1, bs).fill_(self.sos).to(sdf_h.device)
        code_sos = self.embedding_start(sos)
        
        inp_val = torch.cat([code_sos, inp_h, inp_o], dim=0).clone()
        inp_val = torch.cat([inp_val[:1], inp_val[1:][gen_order]], dim=0)[:-1]

        grid_table = self.grid_table.clone().to(sdf_h.device)
        grid_table = torch.cat([grid_table[:1], grid_table[1:][gen_order]], dim=0)
        
        inp_pos = grid_table[:-1].clone() # T x 4
        tgt_pos = grid_table[1:].clone()
        
        code_h = enc_h.squeeze(2).squeeze(2).squeeze(2)
        code_o = enc_o.squeeze(2).squeeze(2).squeeze(2)
        code_h = rearrange(code_h, '(bs d h w) c-> bs c d h w', bs=bs, d=self.grid_size, h=self.grid_size, w=self.grid_size)
        code_o = rearrange(code_o, '(bs d h w) c-> bs c d h w', bs=bs, d=self.grid_size, h=self.grid_size, w=self.grid_size)
        sdf_h = self.fold_to_voxels(sdf_h, bs, self.ncubes_per_dim)
        sdf_o = self.fold_to_voxels(sdf_o, bs, self.ncubes_per_dim)
        
        context_len = seq_len+extra_cond
        # context_len = seq_len+256
        x_recon_h, x_recon_o, x_recon_tf_h, x_recon_tf_o = self.generate_vis(sdf_h, sdf_o, code_h, code_o,
                                                                            inp_val, inp_pos, tgt_pos, gen_order, seq_len=context_len)
        sdf_h = sdf_h[0].unsqueeze(0)
        sdf_o = sdf_o[0].unsqueeze(0)
        if step != 0:
            sdf_h = None
            sdf_o = None
            x_recon_h = None
            x_recon_o = None
        self.save_vis(None, None, x_recon_h, x_recon_o, x_recon_tf_h, x_recon_tf_o,
                                gif_dir=gif_dir, epoch=0, step=step, save_mesh=False)

        return None
    

    def unconditional_generation(self):
        return None


    def get_gen_order(self, sz, mode='random', hand_pose=None, device=None):
        if mode == 'random':
            # Random order
            gen_order = torch.randperm(sz)
            if device is not None:
                gen_order = gen_order.to(device)
                
        elif mode == 'hand_pose':
            # To order according to camera pose
            # Points closer to camera are reconstructed first
            # Sample random hand pose
            if hand_pose is None:
                trans = np.array([0.0, 0.0, 0.7]) + (np.random.rand(3)-0.5)*0.3
                rot_xyz = (np.random.rand(3)-0.5)*(2.*np.pi)
                rot_mat = R.from_euler('xyz', rot_xyz).as_matrix()
                hand_pose = np.eye(4)
                hand_pose[:3, :3] = rot_mat
                hand_pose[:3, 3] = trans.T
                hand_pose = torch.from_numpy(hand_pose).float().unsqueeze(0)

            # gen_order_half = torch.arange(sz//2)
            grid_table_coords = self.grid_table.clone()
            if device is not None:
                grid_table_coords = grid_table_coords.to(device)
            grid_table_coords = grid_table_coords[1:513, :3]
            bbox_size = utils.util.BBOX_SIZE_X / utils.util.SDF_MULTIPLIER
            grid_table_coords = grid_table_coords * (bbox_size/2.0)
            bbox_center = torch.tensor([utils.util.BBOX_ORIG_X, utils.util.BBOX_ORIG_Y, utils.util.BBOX_ORIG_Z]).float()
            if device is not None:
                bbox_center = bbox_center.to(device)
            bbox_center = bbox_center / utils.util.SDF_MULTIPLIER
            grid_table_coords = grid_table_coords + bbox_center
            grid_table_coords = grid_table_coords.unsqueeze(0)

            trans = Transform3d(matrix=hand_pose.transpose(1,2))
            grid_table_coords = trans.transform_points(grid_table_coords)
            grid_table_coords = grid_table_coords.squeeze(0)
            grid_table_coords = grid_table_coords[:, :3]

            # calculate distance from camera
            dist = torch.norm(grid_table_coords, dim=-1)
            # sort gen_order by ascending distance
            indices = torch.argsort(dist)
            # duplicate each element
            indices = indices.repeat_interleave(2)
            # add 512 to even elements
            indices[1::2] += (sz//2)
            gen_order = indices

        return gen_order


    def forward(self, inp_val, inp_pos, tgt_pos):
        outp = self.tf(inp_val, inp_pos, tgt_pos)
        return outp
    

    def freeze_vqvae(self):
        if self.opt.ho_mode == 'joint' or self.opt.ho_mode == 'hand':
            self.vqvae_h.eval()
            self.embedding_encoder_h.eval()
        if self.opt.ho_mode == 'joint' or self.opt.ho_mode == 'object':
            self.vqvae_o.eval()
            self.embedding_encoder_o.eval()
    

    def freeze_tf(self):
        self.tf.eval()
        self.embedding_start.eval()


    def freeze_all(self):
        self.freeze_vqvae()
        self.freeze_tf()
        

    def process_batch(self, batch):
        code_h = batch['code_h']
        code_o = batch['code_o']
        codeidx_h = batch['codeidx_h']
        codeidx_o = batch['codeidx_o']
        return code_h, code_o, codeidx_h, codeidx_o


    def serialize(self, codeidx_h, codeidx_o, code_h, code_o):
        if self.opt.ho_mode == 'joint':
            idx_seq_h = rearrange(codeidx_h, 'bs dz hz wz -> (dz hz wz) bs').contiguous() # to (T, B)
            idx_seq_o = rearrange(codeidx_o, 'bs dz hz wz -> (dz hz wz) bs').contiguous()
            idx_seq = torch.cat([idx_seq_h, idx_seq_o], dim=0)

            code_seq_h = rearrange(code_h, 'bs cz dz hz wz -> (dz hz wz) bs cz').contiguous() # to (T, B, C)
            code_seq_o = rearrange(code_o, 'bs cz dz hz wz -> (dz hz wz) bs cz').contiguous()
            code_seq = torch.cat([code_seq_h, code_seq_o], dim=0)

        elif self.opt.ho_mode == 'hand':
            idx_seq = rearrange(codeidx_h, 'bs dz hz wz -> (dz hz wz) bs').contiguous()
            code_seq = rearrange(code_h, 'bs cz dz hz wz -> (dz hz wz) bs cz').contiguous()
        elif self.opt.ho_mode == 'object':
            idx_seq = rearrange(codeidx_o, 'bs dz hz wz -> (dz hz wz) bs').contiguous()
            code_seq = rearrange(code_o, 'bs cz dz hz wz -> (dz hz wz) bs cz').contiguous()
        
        return idx_seq, code_seq
    

    def prepare_tf_input(self, idx_seq, code_seq, mask_prob=0.0, add_grid_noise=False):
        T, B = idx_seq.shape[:2]
        
        gen_order = self.get_gen_order(T, mode='hand_pose')

        # Encode input and targets
        sos = torch.LongTensor(1, B).fill_(self.sos).to(idx_seq.device)
        code_sos = self.embedding_start(sos)
        
        inp_val = torch.cat([code_sos, code_seq[gen_order][:-1]], dim=0).clone()
        # Randomly mask out some input
        if mask_prob > 0.0:
            mask = torch.rand(inp_val.shape[0], inp_val.shape[1]).to(inp_val.device)
            mask = mask < self.opt.mask_prob
            # Don't mask out sos
            mask[0] = False
            # noise = (torch.rand_like(inp_val).to(inp_val.device) - 0.5) * 2 * 0.01
            noise = torch.randint(0, self.cb_size, idx_seq.shape).to(inp_val.device)
            noise_h = self.embedding_encoder_h(noise[:int((T-1)/2), :])
            noise_o = self.embedding_encoder_o(noise[int((T-1)/2):, :])
            noise = torch.cat([noise_h, noise_o], dim=0)[gen_order][:-1]
            noise = torch.cat([code_sos, noise], dim=0)
            inp_val[mask] = noise[mask]
        tgt_idx = idx_seq[gen_order]

        grid_table = self.grid_table.clone().to(idx_seq.device)
        if add_grid_noise:
            grid_table[1:,3] = grid_table[1:,3] + (torch.rand(grid_table[1:,3].shape)-0.5).to(grid_table.device)*self.grid_val_noise
            grid_table[1:,:3] = grid_table[1:,:3] + (torch.rand(grid_table[1:,:3].shape)-0.5).to(grid_table.device)*self.grid_xyz_noise
        pos_shuffled = torch.cat([grid_table[:1], grid_table[1:][gen_order]], dim=0)   # T+1 x 4, <sos> should always at start.
        inp_pos = pos_shuffled[:-1].clone() # T x 4
        tgt_pos = pos_shuffled[1:].clone()

        return inp_val, inp_pos, tgt_idx, tgt_pos, gen_order
    

    def separate_ho(self, gen_order, outp, tgt, outp_occp=None):
        tgt_ordered = tgt[torch.argsort(gen_order)]
        outp_ordered = outp[torch.argsort(gen_order)]

        tgt_h, tgt_o = torch.split(tgt_ordered, int(tgt_ordered.shape[0]/2), dim=0)
        outp_h, outp_o = torch.split(outp_ordered, int(outp_ordered.shape[0]/2), dim=0)
        tgt_h = rearrange(tgt_h, 't b -> (t b)')
        tgt_o = rearrange(tgt_o, 't b -> (t b)')
        outp_h = rearrange(outp_h, 't b cls -> (t b) cls')
        outp_o = rearrange(outp_o, 't b cls -> (t b) cls')

        outp_occp_h = None
        outp_occp_o = None
        tgt_occp_h = None
        tgt_occp_o = None
        if self.two_stage:
            outp_occp = outp_occp[torch.argsort(gen_order)]
            outp_occp_h, outp_occp_o = torch.split(outp_occp, int(outp_occp.shape[0]/2), dim=0)
            outp_occp_h = rearrange(outp_occp_h, 't b cls -> (t b) cls')
            outp_occp_o = rearrange(outp_occp_o, 't b cls -> (t b) cls')

            tgt_occp_h = tgt_h != self.empty_idx_h
            tgt_occp_o = tgt_o != self.empty_idx_o

        return tgt_h, tgt_o, outp_h, outp_o, tgt_occp_h, tgt_occp_o, outp_occp_h, outp_occp_o
    

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
        self.freeze_vqvae()
        
        code_h, code_o, codeidx_h, codeidx_o = self.process_batch(batch)

        idx_seq, code_seq = self.serialize(codeidx_h, codeidx_o, code_h, code_o)

        inp_val, inp_pos, tgt_idx, tgt_pos, gen_order = self.prepare_tf_input(idx_seq, code_seq, mask_prob=self.opt.mask_prob, add_grid_noise=True)

        # Forward
        if self.two_stage:
            outp, outp_occp, outp_masked = self.forward(inp_val, inp_pos, tgt_pos) # T x B x V
        else:
            outp, outp_occp, _ = self.forward(inp_val, inp_pos, tgt_pos) # T x B x V
        
        tgt_h, tgt_o, outp_h, outp_o, tgt_occp_h, tgt_occp_o, outp_occp_h, outp_occp_o = self.separate_ho(gen_order, outp, tgt_idx, outp_occp)

        if self.two_stage:
            loss_occp_h = self.criterion_cce_occp(outp_occp_h, tgt_occp_h.long())
            loss_occp_o = self.criterion_cce_occp(outp_occp_o, tgt_occp_o.long())
            loss_occp = loss_occp_h + loss_occp_o

            l_tgt_h = tgt_h[tgt_occp_h]
            l_tgt_o = tgt_o[tgt_occp_o]
            l_outp_h = outp_h[tgt_occp_h]
            l_outp_o = outp_o[tgt_occp_o]

            loss_hand = self.criterion_h(l_outp_h, l_tgt_h)
            loss_obj = self.criterion_o(l_outp_o, l_tgt_o)

            loss = loss_hand + loss_obj + loss_occp

        else:
            loss_hand = self.criterion_h(outp_h, tgt_h)
            loss_obj = self.criterion_o(outp_o, tgt_o)

            # loss = (loss_hand * 0.25 + loss_obj * 0.75) * 2.0
            loss = loss_hand + loss_obj

        with torch.no_grad():
            if self.two_stage:
                outp_masked = outp_masked[torch.argsort(gen_order)]
                outp_masked_h, outp_masked_o = torch.split(outp_masked, int(outp_masked.shape[0]/2), dim=0)
                outp_h = rearrange(outp_masked_h, 't b cls -> (t b) cls')
                outp_o = rearrange(outp_masked_o, 't b cls -> (t b) cls')

            acc_h, acc_noempty_h, acc_bin_pos_h, acc_bin_neg_h = self.compute_acc(outp_h, tgt_h, self.empty_idx_h)
            acc_o, acc_noempty_o, acc_bin_pos_o, acc_bin_neg_o = self.compute_acc(outp_o, tgt_o, self.empty_idx_o)
            
            acc = (acc_h + acc_o) / 2.0
        
        self.log("train/loss_ce", loss, sync_dist=True)
        self.log("train/loss_hand", loss_hand, sync_dist=True)
        self.log("train/loss_obj", loss_obj, sync_dist=True)
        if self.two_stage:
            self.log("train/loss_occp", loss_occp, sync_dist=True)
            self.log("train/loss_occp_h", loss_occp_h, sync_dist=True)
            self.log("train/loss_occp_o", loss_occp_o, sync_dist=True)

        self.log("train/acc", acc, sync_dist=True)
        self.log("train/acc_h", acc_h, sync_dist=True)
        self.log("train/acc_o", acc_o, sync_dist=True)
        self.log("train/acc_h_noempty", acc_noempty_h, sync_dist=True)
        self.log("train/acc_o_noempty", acc_noempty_o, sync_dist=True)
        self.log("train/acc_bin_pos_h", acc_bin_pos_h, sync_dist=True)
        self.log("train/acc_bin_pos_o", acc_bin_pos_o, sync_dist=True)
        self.log("train/acc_bin_neg_h", acc_bin_neg_h, sync_dist=True)
        self.log("train/acc_bin_neg_o", acc_bin_neg_o, sync_dist=True)

        # Only vis on rank 0
        if self.global_rank == 0:
            print("global_step", self.global_step)
            try:
                if self.global_step % self.opt.display_freq == 0 and self.global_step != 0:
                    seq_len = 1
                    T = inp_val.shape[0] + seq_len
                    B = inp_val.shape[1]
                    O = inp_val[:seq_len]
                    pred_h, pred_o, _ = self.inference(T, B, O, inp_pos, tgt_pos, gen_order, topk=1)

                    vis_xyz = self.get_visxyz().to(code_h.device)
                    x_recon_h, x_recon_o = self.generate_vis(code_h, code_o, vis_xyz)
                    x_recon_tf_h, x_recon_tf_o = self.generate_vis(pred_h, pred_o, vis_xyz)

                    save_dir = os.path.join(self.logger.log_dir, "gif", "train")
                    prefix="e_{:05d}_b_{:05d}_recon".format(self.current_epoch, batch_idx)
                    self.save_sdf_vis(x_recon_h, x_recon_o, save_dir=save_dir, save_mesh=False, save_gif=True, prefix=prefix)
                    prefix="e_{:05d}_b_{:05d}_recon_tf".format(self.current_epoch, batch_idx)
                    self.save_sdf_vis(x_recon_tf_h, x_recon_tf_o, save_dir=save_dir, save_mesh=False, save_gif=True, prefix=prefix)
            except Exception as e:
                print(e)
                print("Error in saving gif")
                traceback.print_exc()
        return loss
    

    def validation_step(self, batch, batch_idx):
        
        self.freeze_vqvae()
        
        code_h, code_o, codeidx_h, codeidx_o = self.process_batch(batch)

        idx_seq, code_seq = self.serialize(codeidx_h, codeidx_o, code_h, code_o)

        inp_val, inp_pos, tgt_idx, tgt_pos, gen_order = self.prepare_tf_input(idx_seq, code_seq)

        # Forward
        if self.two_stage:
            outp, outp_occp, outp_masked = self.forward(inp_val, inp_pos, tgt_pos) # T x B x V
        else:
            outp, outp_occp, _ = self.forward(inp_val, inp_pos, tgt_pos) # T x B x V
        
        tgt_h, tgt_o, outp_h, outp_o, tgt_occp_h, tgt_occp_o, outp_occp_h, outp_occp_o = self.separate_ho(gen_order, outp, tgt_idx, outp_occp)

        with torch.no_grad():
            if self.two_stage:
                outp_masked = outp_masked[torch.argsort(gen_order)]
                outp_masked_h, outp_masked_o = torch.split(outp_masked, int(outp_masked.shape[0]/2), dim=0)
                outp_h = rearrange(outp_masked_h, 't b cls -> (t b) cls')
                outp_o = rearrange(outp_masked_o, 't b cls -> (t b) cls')

            acc_h, acc_noempty_h, acc_bin_pos_h, acc_bin_neg_h = self.compute_acc(outp_h, tgt_h, self.empty_idx_h)
            acc_o, acc_noempty_o, acc_bin_pos_o, acc_bin_neg_o = self.compute_acc(outp_o, tgt_o, self.empty_idx_o)
            
            acc = (acc_h + acc_o) / 2.0

        self.log("val/acc", acc, sync_dist=True)
        self.log("val/acc_h", acc_h, sync_dist=True)
        self.log("val/acc_o", acc_o, sync_dist=True)
        self.log("val/acc_h_noempty", acc_noempty_h, sync_dist=True)
        self.log("val/acc_o_noempty", acc_noempty_o, sync_dist=True)
        self.log("val/acc_bin_pos_h", acc_bin_pos_h, sync_dist=True)
        self.log("val/acc_bin_pos_o", acc_bin_pos_o, sync_dist=True)
        self.log("val/acc_bin_neg_h", acc_bin_neg_h, sync_dist=True)
        self.log("val/acc_bin_neg_o", acc_bin_neg_o, sync_dist=True)

        # Only vis on rank 0
        if self.global_rank == 0:
            print("global_step", self.global_step)
            try:
                if self.current_epoch % 10 == 0 and batch_idx % 10 == 0:
                    seq_len = 1
                    T = inp_val.shape[0] + seq_len
                    B = inp_val.shape[1]
                    O = inp_val[:seq_len]
                    pred_h, pred_o, _ = self.inference(T, B, O, inp_pos, tgt_pos, gen_order, topk=1)

                    vis_xyz = self.get_visxyz().to(code_h.device)
                    x_recon_h, x_recon_o = self.generate_vis(code_h, code_o, vis_xyz)
                    x_recon_tf_h, x_recon_tf_o = self.generate_vis(pred_h, pred_o, vis_xyz)
                    
                    save_dir = os.path.join(self.logger.log_dir, "gif", "val")
                    prefix="e_{:05d}_b_{:05d}_recon".format(self.current_epoch, batch_idx)
                    self.save_sdf_vis(x_recon_h, x_recon_o, save_dir=save_dir, save_mesh=False, save_gif=True, prefix=prefix)
                    prefix="e_{:05d}_b_{:05d}_recon_tf".format(self.current_epoch, batch_idx)
                    self.save_sdf_vis(x_recon_tf_h, x_recon_tf_o, save_dir=save_dir, save_mesh=False, save_gif=True, prefix=prefix)
            except Exception as e:
                print(e)
                print("Error in saving gif")
                traceback.print_exc()
        return None
