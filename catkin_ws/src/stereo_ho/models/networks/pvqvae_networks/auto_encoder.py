# adopt from: 
# - VQVAE: https://github.com/nadavbh12/VQ-VAE
# - Encoder: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py

from __future__ import print_function

import random

import torch
import torch.utils.data
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from einops import rearrange

from models.networks.pvqvae_networks.modules import Encoder3D, Decoder3D, DecoderMLP, DecoderMLP2
from models.networks.pvqvae_networks.quantizer import VectorQuantizer

def init_weights(net, init_type='normal', gain=0.01):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

    # propagate to children
    for m in net.children():
        m.apply(init_func)


class PVQVAE(nn.Module):
    def __init__(self,
                 ddconfig,
                 decmlpconfig,
                 n_embed,
                 embed_dim,
                 mlp_decoder=False,
                 ckpt_path=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super(PVQVAE, self).__init__()
        
        self.ddconfig = ddconfig
        self.decmlpconfig = decmlpconfig
        # self.lossconfig = lossconfig
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.mlp_decoder = mlp_decoder
        self.use_codebook = True

        # mostly from taming
        self.encoder = Encoder3D(**ddconfig)
        if self.mlp_decoder:
            self.decoder = DecoderMLP2(latent_size=ddconfig.z_channels, **decmlpconfig, **ddconfig)
        else:
            self.decoder = Decoder3D(**ddconfig)

        # self.loss = VQLoss(lossconfig)
        if self.use_codebook:
            self.quantize = VectorQuantizer(n_embed, embed_dim, beta=1.0,
                                            remap=remap, sane_index_shape=sane_index_shape, legacy=False)
        self.quant_conv = torch.nn.Conv3d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)

        init_weights(self.encoder, 'normal', 0.02)
        init_weights(self.decoder, 'normal', 0.02)
        # init_weights(self.decoder.module_list, 'normal', 0.02)
        # init_weights(self.decoder.fc_map, 'normal', 0.02)
        # init_weights(self.decoder.mid, 'normal', 0.02)
        # init_weights(self.decoder.conv_in, 'normal', 0.02)
        # init_weights(self.decoder.conv_out, 'normal', 0.02)
        # init_weights(self.decoder.norm_out, 'normal', 0.02)
        init_weights(self.quant_conv, 'normal', 0.02)
        init_weights(self.post_quant_conv, 'normal', 0.02)

        # Print encoder and decoder size
        print('Encoder size: ', sum(p.numel() for p in self.encoder.parameters() if p.requires_grad) + 
                                sum(p.numel() for p in self.quant_conv.parameters() if p.requires_grad))
        print('Decoder size: ', sum(p.numel() for p in self.decoder.parameters() if p.requires_grad) +
                                sum(p.numel() for p in self.post_quant_conv.parameters() if p.requires_grad))
        # print('Decoder mid size: ', sum(p.numel() for p in self.decoder.mid.parameters() if p.requires_grad))
        # print('Decoder conv_in size: ', sum(p.numel() for p in self.decoder.conv_in.parameters() if p.requires_grad))
        # print('Decoder conv_out size: ', sum(p.numel() for p in self.decoder.conv_out.parameters() if p.requires_grad))

    def encode(self, x, mix_cb=False, cb_restart=False):
        h = self.encoder(x)
        h = self.quant_conv(h)
        if self.use_codebook:
            quant, emb_loss, info = self.quantize(h, is_voxel=True, cb_restart=cb_restart)
            # print("quantize")
        else:
            quant = h
            emb_loss = torch.zeros(1, device=h.device)
            info = (0,0,0,0,0,0)
        return quant, emb_loss, info

    def decode(self, quant, xyz=None, use_extended=False):
        quant = self.post_quant_conv(quant)

        if self.mlp_decoder:
            dec, _ = self.decoder(quant, xyz, use_extended)
        else:
            dec = self.decoder(quant)
        
        return dec

    def decode_from_quant(self,quant_code):
        embed_from_code = self.quantize.embedding(quant_code)
        return embed_from_code
    
    def decode_enc_idices(self, enc_indices, z_spatial_dim=8, xyz=None):
        # for transformer
        enc_indices = rearrange(enc_indices, 't bs -> (bs t)')
        z_q = self.quantize.embedding(enc_indices) # (bs t) zd#
        z_q = rearrange(z_q, '(bs d1 d2 d3) zd -> bs zd d1 d2 d3', d1=z_spatial_dim, d2=z_spatial_dim, d3=z_spatial_dim)
        quant = self.post_quant_conv(z_q)
        if self.mlp_decoder:
            dec, _ = self.decoder(quant, xyz)
        else:
            dec = self.decoder(quant)
        # dec = self.decode(z_q)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec
    
    @staticmethod
    def fold_to_voxels(x_cubes, batch_size, ncubes_per_dim):
        x = rearrange(x_cubes, '(b p) c d h w -> b p c d h w', b=batch_size) 
        x = rearrange(x, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                        p1=ncubes_per_dim, p2=ncubes_per_dim, p3=ncubes_per_dim)
        return x

    def forward(self, input_sdf_cubes, target_xyz, cur_bs, ncubes_per_dim, verbose=False, cb_restart=False):
        # Forward
        zq_cubes, qloss, qinfo = self.encode(input_sdf_cubes, mix_cb=False, cb_restart=cb_restart) # zq_cubes: ncubes X zdim X 1 X 1 X 1
        # if batch_idx % 32 == 31:
        #     codebook_usage = qinfo[4]
        #     cb_file = os.path.join(self.logger.log_dir, "cb_usage", "cb_usage_{}.npy".format(self.trainer.proc_rank))
        #     np.save(cb_file, codebook_usage.cpu().numpy())
        
        zq_voxels = self.fold_to_voxels(zq_cubes, batch_size=cur_bs, ncubes_per_dim=ncubes_per_dim) # zq_voxels: bs X zdim X ncubes_per_dim X ncubes_per_dim X ncubes_per_dim
        if self.mlp_decoder:
            recon_sdf = self.decode(zq_voxels, target_xyz)
        else:
            recon_sdf = self.decode(zq_voxels)
        
        return recon_sdf, zq_voxels, qloss, qinfo

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()
