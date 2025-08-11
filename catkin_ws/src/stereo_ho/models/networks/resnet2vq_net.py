
import torch
import torch.nn as nn
import cv2

from einops import rearrange

from models.networks.resnet import model_urls, resnet18, resnet50
from models.networks.pvqvae_networks.modules import ResnetBlock as PVQVAEResnetBlock
from models.networks.pvqvae_networks.modules import Upsample, AttnBlock, Normalize, nonlinearity
from models.networks.transformer_networks.pos_embedding import PEPixelTransformer

import utils
import numpy as np
from pytorch3d.transforms import Transform3d
from pytorch3d.renderer import PerspectiveCameras

class PredictionHead(nn.Module):
    def __init__(self, in_channels_visual, in_channels_pos, out_channels, hidden_dim=512, dropout_prob=0.5):
        super(PredictionHead, self).__init__()
        self.in_channels_visual = in_channels_visual
        self.in_channels_pos = in_channels_pos
        self.bn_visual = nn.BatchNorm1d(in_channels_visual)
        self.lin_in_visual = nn.Linear(in_channels_visual, in_channels_visual)

        self.lin1 = nn.Linear(in_channels_visual+in_channels_pos, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x_pos = x[:,:self.in_channels_pos]
        x_visual = x[:,self.in_channels_pos:]

        x_visual = self.bn_visual(x_visual)
        x_visual = self.lin_in_visual(x_visual)

        x = torch.cat([x_visual, x_pos], dim=-1)

        x = self.lin1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.lin_out(x)

        return x

class PredictionHead3DConv(nn.Module):
    def __init__(self, in_channels, out_channels, nblocks=3, use_attn=True, dropout_prob=0.5, empty_idx=0):
        super(PredictionHead3DConv, self).__init__()
        convt_layers = []
        in_c = in_channels
        for i in range(nblocks):
            out_c = min(in_c // 2, out_channels)
            print(f'convt {i}: {in_c} -> {out_c}')
            convt_layers.append(
                PVQVAEResnetBlock(in_channels=in_c, out_channels=out_c, temb_channels=0, dropout=dropout_prob)
            )
            if use_attn:
                convt_layers.append( AttnBlock(out_c) )
            in_c = out_c

        self.convt_layers = nn.Sequential(*convt_layers)

        self.convt = PVQVAEResnetBlock(in_channels=in_c, out_channels=in_c, temb_channels=0, dropout=dropout_prob)
        self.attn = AttnBlock(in_c)

        self.norm_out = Normalize(in_c)
        self.conv_out = torch.nn.Conv3d(in_c, out_channels, 1)


    def forward(self, x):
        x = self.convt_layers(x)
        x = self.convt(x)
        x = self.attn(x)
        x = self.norm_out(x)
        x = nonlinearity(x)
        x = self.conv_out(x)

        return x

class ResNet2VQ(nn.Module):
    
    def __init__(self, opt, tf_conf=None, empty_idx_h=0, empty_idx_o=0):
        super(ResNet2VQ, self).__init__()
        
        self.resnet = resnet18(pretrained=True)
        input_dims = 3
        input_dims += 1
        # extra channels for object segmentation
        orig_layer = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(input_dims, orig_layer.out_channels, kernel_size=orig_layer.kernel_size,
                                    stride=orig_layer.stride, padding=orig_layer.padding, bias=orig_layer.bias)
        self.resnet.conv1.weight.data[:,:3,:,:] = orig_layer.weight.data

        self.proj_hand = True

        self.pos_emb = False
        self.conv_3d = True
        
        self.twoBranch = False
        self.use_global = True
        self.use_local = True

        self.dropout_prob = 0.5

        pe_conf = tf_conf.pe
        ntoken = tf_conf.model.params.ntokens
        self.dz = self.hz = self.wz = 8

        if not(self.proj_hand):
            # AutoSDF
            self.linear_to3d = nn.Linear(8 ** 2, self.dz * self.hz * self.wz)
            in_c_convt1 = self.resnet.block.expansion * 512
            self.pred_ho = PredictionHead3DConv(in_channels=in_c_convt1, out_channels=ntoken*2, dropout_prob=self.dropout_prob)
        else:
            bbox_scale = utils.util.SDF_MULTIPLIER
            bbox_center = [utils.util.BBOX_ORIG_X, utils.util.BBOX_ORIG_Y, utils.util.BBOX_ORIG_Z]
            bbox_center = np.array(bbox_center)/bbox_scale
            bbox_size = [utils.util.BBOX_SIZE_X, utils.util.BBOX_SIZE_Y, utils.util.BBOX_SIZE_Z]
            bbox_size = np.array(bbox_size)/bbox_scale
            bbox_top_left = bbox_center + bbox_size / 2.0
            bbox_bottom_right = bbox_center - bbox_size / 2.0
            bbox_xyz = np.meshgrid(np.linspace(bbox_bottom_right[0], bbox_top_left[0], 8, endpoint=False),
                                        np.linspace(bbox_bottom_right[1], bbox_top_left[1], 8, endpoint=False),
                                        np.linspace(bbox_bottom_right[2], bbox_top_left[2], 8, endpoint=False))
            # add half voxel size to get voxel centers
            bbox_xyz = np.stack(bbox_xyz, axis=-1)
            bbox_xyz = bbox_xyz.reshape(-1, 3)
            voxel_size = bbox_size / 8.0
            bbox_xyz += voxel_size / 2.0
            self.bbox_xyz = torch.from_numpy(bbox_xyz).float().unsqueeze(0)
            self.bbox_xyz_norm = self.bbox_xyz - torch.tensor(bbox_center).unsqueeze(0).unsqueeze(0)
            self.bbox_xyz_norm = self.bbox_xyz_norm / (bbox_size[0]/2.0)

            self.img_size = 256
            cam_f = np.array([10., 10.])
            cam_p = np.array([0., 0.])
            
            self.cam_f = torch.from_numpy(cam_f).float().unsqueeze(0)
            self.cam_p = torch.from_numpy(cam_p).float().unsqueeze(0)
            
            res_c = 512
            res_local_c = 512+256+128+64
            x_global_c = 512
            x_local_c = 512
            hidden_c = 512
            in_c = 0
            if self.pos_emb:
                self.pos_embedding = PEPixelTransformer(pe_conf=pe_conf)
                in_c += pe_conf.pos_embed_dim

            if self.use_global:
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.global_head = nn.Conv2d(res_c, x_global_c, 1)
            if self.use_local:
                self.local_head = nn.Conv2d(res_local_c, x_local_c, 1)
            if self.use_local or self.use_global:
                in_c += x_local_c

            print(f'pos_emb in_c: {in_c}')
            if self.pos_emb:
                if self.twoBranch:
                    self.pred_h = PredictionHead(in_channels_visual=x_local_c, in_channels_pos=pe_conf.pos_embed_dim, out_channels=ntoken, hidden_dim=hidden_c, dropout_prob=self.dropout_prob)
                    self.pred_o = PredictionHead(in_channels_visual=x_local_c, in_channels_pos=pe_conf.pos_embed_dim, out_channels=ntoken, hidden_dim=hidden_c, dropout_prob=self.dropout_prob)
                else:
                    self.pred_ho = PredictionHead(in_channels_visual=in_c, in_channels_pos=pe_conf.pos_embed_dim, out_channels=ntoken*2, hidden_dim=hidden_c, dropout_prob=self.dropout_prob)

            elif self.conv_3d:

                self.pred_h = PredictionHead3DConv(in_channels=x_local_c, out_channels=ntoken, dropout_prob=self.dropout_prob, empty_idx=empty_idx_h)
                self.pred_o = PredictionHead3DConv(in_channels=x_local_c, out_channels=ntoken, dropout_prob=self.dropout_prob, empty_idx=empty_idx_o)

        
    def forward(self, x, hand_pose=None, proj_mode='WeakPerspective', K=None, bbox=None):
        proj_2d, proj_axis = None, None
        local_x, global_x = self.resnet(x)
        B, C, SH, SW = x.shape
        ntoken = 512

        if not(self.proj_hand):
            x = rearrange(x, 'b c h w -> (b c) (h w)')
            x = self.linear_to3d(x)
            x = rearrange(x, '(b c) (d h w) -> b c d h w', b=B, c=C, d=self.dz, h=self.hz, w=self.wz) # 512, 8x8x8
            x = self.pred_ho(x)
            x_h = x[:,:ntoken,:,:,:]
            x_o = x[:,ntoken:,:,:,:]
            
        else:

            if self.use_local:
                # hand pose is B x 4 x 4
                if proj_mode == 'WeakPerspective':
                    trans = Transform3d(matrix=hand_pose.transpose(1,2)).to(x.device)
                    camera = PerspectiveCameras(self.cam_f, self.cam_p, image_size=(self.img_size, self.img_size)).to(x.device)

                    bbox_xyz_batched = self.bbox_xyz.repeat(B, 1, 1).to(x.device) # B x 512 x 3
                    bbox_xyz_batched = trans.transform_points(bbox_xyz_batched)
                    bbox_xyz_batched = rearrange(bbox_xyz_batched, 'b n xyz -> (b n) xyz')
                    index_2d = camera.transform_points_screen(bbox_xyz_batched)
                    index_2d = index_2d[0][:,:2]
                    index_2d = rearrange(index_2d, '(b n) xyz -> b n xyz', b=B)
                    index_2d = self.img_size-index_2d

                    proj_2d = index_2d.long().clone()
                    axis = torch.from_numpy(np.array([[0,0,0],[0.1,0,0],[0,0.1,0],[0,0,0.1]])).float().unsqueeze(0).to(x.device)
                    axis = axis.repeat(B, 1, 1) # B x 4 x 3
                    axis_trans = trans.transform_points(axis)
                    axis_trans = rearrange(axis_trans, 'b n xyz -> (b n) xyz')
                    axis_2d = camera.transform_points_screen(axis_trans)
                    axis_2d = axis_2d[0][:,:2]
                    axis_2d = rearrange(axis_2d, '(b n) xyz -> b n xyz', b=B)
                    axis_2d = self.img_size-axis_2d
                    axis_2d = axis_2d.long()
                    proj_axis = axis_2d.clone() # B x 4 x 2

                else:
                    assert K is not None
                    assert bbox is not None
                    bbox_xyz_batched = self.bbox_xyz.repeat(B, 1, 1).to(x.device) # B x 512 x 3
                    bbox_xyz_batched = rearrange(bbox_xyz_batched, 'b n xyz -> (b n) xyz')
                    bbox_xyz_batched = bbox_xyz_batched.cpu().numpy()
                    rvec = cv2.Rodrigues(hand_pose[:3, :3])[0]
                    tvec = hand_pose[:3, 3]
                    dist_coeffs = np.zeros((4,1))
                    index_2d = cv2.projectPoints(bbox_xyz_batched, rvec, tvec, K, dist_coeffs)[0]
                    index_2d = np.squeeze(index_2d, axis=1)/2.0
                    index_2d = (index_2d - np.array([bbox[0], bbox[1]])) / (bbox[2]-bbox[0]) * 256.0
                    index_2d = torch.from_numpy(index_2d).float().to(x.device)
                    index_2d = rearrange(index_2d, '(b n) xyz -> b n xyz', b=B)

                    proj_2d = index_2d.long().clone()
                    axis = np.array([[0,0,0],[0.1,0,0],[0,0.1,0],[0,0,0.1]]).astype(np.float32).reshape(4,3)
                    axis_2d = cv2.projectPoints(axis, rvec, tvec, K, dist_coeffs)[0]
                    axis_2d = np.squeeze(axis_2d, axis=1)/2.0
                    axis_2d = (axis_2d - np.array([bbox[0], bbox[1]])) / (bbox[2]-bbox[0]) * 256.0
                    axis_2d = torch.from_numpy(axis_2d).float().to(x.device)
                    axis_2d = axis_2d.unsqueeze(0).repeat(B, 1, 1)
                    proj_axis = axis_2d.clone() # B x 4 x 2

                # Need to form 3D latent cube by indexing into x_local
                # Combine latents
                align_corners = True
                latent_sz = local_x[0].shape[-2:]
                for i, lat in enumerate(local_x):
                    local_x[i] = nn.functional.interpolate(
                        local_x[i],
                        latent_sz,
                        mode='bilinear',
                        align_corners=align_corners,
                    )
                local_x = torch.cat(local_x, dim=1)

                x_2d = self.local_head(local_x)
                x_2d = rearrange(x_2d, 'b c h w -> b h w c')
                # Then index into x_2d
                index_2d = (index_2d/256.0) * latent_sz[0]
                index_2d = torch.clamp(index_2d, 0, latent_sz[0]-1)
                index_2d = index_2d.long()

                x_3d = x_2d[torch.arange(B).unsqueeze(1), index_2d[:,:,0], index_2d[:,:,1], :]
                

            if self.use_global:
                # x is B x 512 x 8 x 8
                x_g = self.avgpool(global_x)
                x_g = self.global_head(x_g)
                
                

            if self.pos_emb:
                bbox_xyz_batched_norm = self.bbox_xyz_norm.repeat(B, 1, 1).to(x.device) # B x 512 x 3
                bbox_xyz_batched_norm = bbox_xyz_batched_norm.float()
                pe_xyz = rearrange(bbox_xyz_batched_norm, 'b n xyz -> (b n) xyz')
                pe = self.pos_embedding(pe_xyz) # BT, 128
                x_combined = pe

                if self.use_local:
                    x_3d = rearrange(x_3d, 'b T c -> (b T) c')
                if self.use_global:
                    x_g = rearrange(x_g, 'b c h w -> b (c h w)')
                    x_g = x_g.unsqueeze(1).repeat(1, self.dz * self.hz * self.wz, 1)
                    x_g = rearrange(x_g, 'b T c -> (b T) c')

                if self.use_local and self.use_global:
                    x_visual = x_g + x_3d
                    x_combined = torch.cat([x_combined, x_visual], dim=-1)
                elif self.use_global:
                    x_combined = torch.cat([x_combined, x_g], dim=-1)
                elif self.use_local:
                    x_combined = torch.cat([x_combined, x_3d], dim=-1)
                
                if self.twoBranch:
                    x_h = self.pred_h(x_combined)
                    x_h = rearrange(x_h, '(b d h w) c -> b c d h w', b=B, d=self.dz, h=self.hz, w=self.wz)

                    x_o = self.pred_o(x_combined)
                    x_o = rearrange(x_o, '(b d h w) c -> b c d h w', b=B, d=self.dz, h=self.hz, w=self.wz)

                else:
                    x = self.pred_ho(x_combined)
                    x = rearrange(x, '(b d h w) c -> b c d h w', b=B, d=self.dz, h=self.hz, w=self.wz) # B, 1024, 8x8x8
                    x_h = x[:,:ntoken,:,:,:]
                    x_o = x[:,ntoken:,:,:,:]

            elif self.conv_3d:
                
                if self.use_local:
                    x_3d_grid = rearrange(x_3d, 'b (d h w) c -> b c d h w', b=B, d=self.dz, h=self.hz, w=self.wz)
                if self.use_global:
                    x_g_grid = x_g.unsqueeze(-1).repeat(1, 1, self.dz, self.hz, self.wz)

                if self.use_local and self.use_global:
                    x_3d_grid = x_3d_grid + x_g_grid
                elif self.use_global:
                    x_3d_grid = x_g_grid
                elif self.use_local:
                    x_3d_grid = x_3d_grid
                x_h = self.pred_h(x_3d_grid)
                x_o = self.pred_o(x_3d_grid)

        return (x_h, x_o), proj_2d, proj_axis
