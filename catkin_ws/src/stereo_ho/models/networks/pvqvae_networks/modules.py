""" adopted from: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py """
# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
import utils.util as util
import torch.nn.functional as F

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels):
    if in_channels <= 32:
        num_groups = in_channels // 4
    else:
        num_groups = 32

    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            # x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
            x = torch.nn.functional.max_pool3d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, d, h, w = q.shape
        q = q.reshape(b, c, d*h*w)
        q = q.permute(0,2,1)   # b,dhw,c
        k = k.reshape(b, c, d*h*w) # b,c,dhw
        w_ = torch.bmm(q,k)     # b,dhw,dhw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,d*h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, d, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class Encoder3D(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv3d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    print('[*] Enc has Attn at i_level, i_block: %d, %d' % (i_level, i_block))
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        # hs = [self.conv_in(x)]
        print("x shape: ", x.shape)
        h = self.conv_in(x)
        print("h shape: ", h.shape)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                # h = self.down[i_level].block[i_block](hs[-1], temb)
                h = self.down[i_level].block[i_block](h, temb)

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                # hs.append(h)
            if i_level != self.num_resolutions-1:
                # hs.append(self.down[i_level].downsample(hs[-1]))
                h = self.down[i_level].downsample(h)
            print("h shape: ", h.shape)

        # middle
        # h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Decoder3D(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv3d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            # for i_block in range(self.num_res_blocks+1):
            for i_block in range(self.num_res_blocks): # change this to align with encoder
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    print('[*] Dec has Attn at i_level, i_block: %d, %d' % (i_level, i_block))
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        print("h shape: ", h.shape)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            # for i_block in range(self.num_res_blocks+1):
            for i_block in range(self.num_res_blocks): # change this to align encoder
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
            print("h shape: ", h.shape)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DecoderMLP(nn.Module):
    def __init__(
        self,
        latent_size,
        *,
        dims,
        num_class,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        use_classifier=False,
        ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
        attn_resolutions, resamp_with_conv=True, in_channels,
        resolution, z_channels, give_pre_end=False,
        **ignorekwargs
    ):
        super(DecoderMLP, self).__init__()

        def make_sequence():
            return []

        self.use_FiLM = False
        if self.use_FiLM:
            dims = [latent_size] + dims + [1]
        else:
            dims = [latent_size + 3] + dims + [1]  # <<<< 2 outputs instead of 1.

        self.num_layers = len(dims)
        self.num_class = num_class
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm
        self.use_classifier = use_classifier

        
        self.film_size = 256
        if self.use_FiLM:
            self.film_generator = nn.Linear(3, self.film_size*(self.num_layers-3))
        print("Dims: ", dims)
        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

            # classifier
            if self.use_classifier and layer == self.num_layers - 2:
                self.classifier_head = nn.Linear(dims[layer], self.num_class)

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        # self.softplus = nn.Softplus(beta=100)

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        # z to block_in
        print("z_channels", z_channels)
        print("block_in", block_in)
        self.conv_in = torch.nn.Conv3d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=0.0)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=0.0)
        # self.mid.attn_2 = AttnBlock(block_in)
        # self.mid.block_3 = ResnetBlock(in_channels=block_in,
        #                                out_channels=block_in,
        #                                temb_channels=self.temb_ch,
        #                                dropout=0.0)
        # Normalize
        self.norm_out = Normalize(block_in)

        self.conv_out = torch.nn.Conv3d(block_in,
                                        block_in,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        
        self.latent_map = nn.Linear(block_in, latent_size)

        
    # input: N x (L+3)
    def forward(self, latent, xyz, use_extended=False):
        # timestep embedding
        temb = None

        # z to block_in
        h = latent

        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # h = self.mid.attn_2(h)
        # h = self.mid.block_3(h, temb)

        h = self.conv_in(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        latent = h.permute(0, 2, 3, 4, 1)

        # Find BBOX dimensions
        num_patches = latent.shape[2]
        patch_size_x = (util.BBOX_SIZE_X / num_patches)
        patch_size_y = (util.BBOX_SIZE_Y / num_patches)
        patch_size_z = (util.BBOX_SIZE_Z / num_patches)

        # Separate xyz into normal and extended
        num_points = xyz.shape[1]
        if use_extended:
            extended_ratio = util.EXTENDED_RATIO
        else:
            extended_ratio = 0.0
        xyz_normal = xyz[:, int(num_points*extended_ratio):, :]
        xyz_extended = xyz[:, :int(num_points*extended_ratio), :]

        # Put origin at the bottom corner
        xyz_idx_coors_normal = xyz_normal + torch.Tensor([util.BBOX_SIZE_X/2.0, util.BBOX_SIZE_Y/2.0, util.BBOX_SIZE_Z/2.0]).repeat(xyz_normal.shape[0], xyz_normal.shape[1], 1).to(xyz.device)
        xyz_idx_coors_extended = xyz_extended + torch.Tensor([util.BBOX_SIZE_X/2.0, util.BBOX_SIZE_Y/2.0, util.BBOX_SIZE_Z/2.0]).repeat(xyz_extended.shape[0], xyz_extended.shape[1], 1).to(xyz.device)

        # Get index of patch for each xyz point
        patch_idx_x_normal = (xyz_idx_coors_normal[:, :, 0] / patch_size_x).long()
        patch_idx_y_normal = (xyz_idx_coors_normal[:, :, 1] / patch_size_y).long()
        patch_idx_z_normal = (xyz_idx_coors_normal[:, :, 2] / patch_size_z).long()
        patch_idx_x_normal = torch.clamp(patch_idx_x_normal, 0, num_patches-1)
        patch_idx_y_normal = torch.clamp(patch_idx_y_normal, 0, num_patches-1)
        patch_idx_z_normal = torch.clamp(patch_idx_z_normal, 0, num_patches-1)

        patch_idx_x_extended = (xyz_idx_coors_extended[:, :, 0] / patch_size_x).long()
        patch_idx_y_extended = (xyz_idx_coors_extended[:, :, 1] / patch_size_y).long()
        patch_idx_z_extended = (xyz_idx_coors_extended[:, :, 2] / patch_size_z).long()
        patch_idx_x_extended = patch_idx_x_extended + torch.randint_like(patch_idx_x_extended, -1, 2)
        patch_idx_y_extended = patch_idx_y_extended + torch.randint_like(patch_idx_y_extended, -1, 2)
        patch_idx_z_extended = patch_idx_z_extended + torch.randint_like(patch_idx_z_extended, -1, 2)
        patch_idx_x_extended = torch.clamp(patch_idx_x_extended, 0, num_patches-1)
        patch_idx_y_extended = torch.clamp(patch_idx_y_extended, 0, num_patches-1)
        patch_idx_z_extended = torch.clamp(patch_idx_z_extended, 0, num_patches-1)
        
        # Normlise translation of xyz to each patch orgin
        patch_origin_normal = torch.cat([(patch_idx_x_normal*patch_size_x).unsqueeze(2), (patch_idx_y_normal*patch_size_y).unsqueeze(2), (patch_idx_z_normal*patch_size_z).unsqueeze(2)], dim=2)
        patch_origin_normal = patch_origin_normal - torch.Tensor([util.BBOX_SIZE_X/2.0, util.BBOX_SIZE_Y/2.0, util.BBOX_SIZE_Z/2.0]).repeat(xyz_normal.shape[0], xyz_normal.shape[1], 1).to(xyz.device)
        xyz_normal_norm = (xyz_normal - patch_origin_normal)
        xyz_normal_norm = ((xyz_normal_norm / patch_size_x) - 0.5) * 2.0

        patch_origin_extended = torch.cat([(patch_idx_x_extended*patch_size_x).unsqueeze(2), (patch_idx_y_extended*patch_size_y).unsqueeze(2), (patch_idx_z_extended*patch_size_z).unsqueeze(2)], dim=2)
        patch_origin_extended = patch_origin_extended - torch.Tensor([util.BBOX_SIZE_X/2.0, util.BBOX_SIZE_Y/2.0, util.BBOX_SIZE_Z/2.0]).repeat(xyz_extended.shape[0], xyz_extended.shape[1], 1).to(xyz.device)
        xyz_extended_norm = (xyz_extended - patch_origin_extended)
        xyz_extended_norm = ((xyz_extended_norm / patch_size_x) - 0.5) * 2.0

        xyz_norm = torch.cat([xyz_extended_norm, xyz_normal_norm], dim=1)

        # Output dim: B x NUM_POINTS x Z
        latent_vecs = torch.zeros((latent.shape[0], xyz.shape[1], latent.shape[4])).to(latent.device)
        latent_vecs[:, :int(num_points*extended_ratio), :] = latent[:, patch_idx_x_extended, patch_idx_y_extended, patch_idx_z_extended]
        latent_vecs[:, int(num_points*extended_ratio):, :] = latent[:, patch_idx_x_normal, patch_idx_y_normal, patch_idx_z_normal]
        input = torch.cat([latent_vecs, xyz_norm], dim=2).squeeze(0)
        sample_size = input.shape[0]

        latent_vecs = input[:, :-3]
        latent_vecs = self.latent_map(latent_vecs)
        xyz = input[:, -3:]
        input = torch.cat([latent_vecs, xyz], 1)


        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        if self.use_FiLM:
            film_vector = self.film_generator(xyz)
            x = latent_vecs.view(-1, latent_vecs.shape[2])

        for layer in range(0, self.num_layers - 1):
            # classify
            if self.use_classifier and layer == self.num_layers - 2:
                predicted_class = self.classifier_head(x)

            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)

                if layer > 0 and layer < self.num_layers - 3 and self.use_FiLM:
                    beta = film_vector[:, (layer-1)*self.film_size:(layer)*self.film_size]
                    x = x + beta

                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = self.th(x)
        x_sdf = x.squeeze(1).unsqueeze(0)

        # hand, object, class label
        if self.use_classifier:
            return x_sdf, predicted_class
        else:
            return x_sdf, torch.Tensor([0]).cuda()

class DecoderMLP2(nn.Module):
    def __init__(
        self,
        latent_size,
        *,
        dims,
        num_class,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        use_classifier=False,
        ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
        attn_resolutions, resamp_with_conv=True, in_channels,
        resolution, z_channels, give_pre_end=False,
        **ignorekwargs
    ):
        super(DecoderMLP2, self).__init__()

        block_in = 128 / (2 ** (len(ch_mult) - 1))
        self.latent_size = latent_size
        self.temb_ch = 0


        self.conv_in = torch.nn.Conv3d(self.latent_size,
                                       self.latent_size,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        
        self.top = nn.Module()
        self.top.block_1 = ResnetBlock(in_channels=self.latent_size,
                                        out_channels=self.latent_size,
                                        temb_channels=self.temb_ch,
                                        dropout=0.0)
        self.top.attn_1 = AttnBlock(self.latent_size)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=self.latent_size,
                                       out_channels=self.latent_size,
                                       temb_channels=self.temb_ch,
                                       dropout=0.0)
        self.mid.attn_1 = AttnBlock(self.latent_size)
        self.mid.block_2 = ResnetBlock(in_channels=self.latent_size,
                                       out_channels=self.latent_size,
                                       temb_channels=self.temb_ch,
                                       dropout=0.0)

        mlp_latent_size = self.latent_size

        self.use_FiLM = False
        self.film_size = dims[0]
        if self.use_FiLM:
            dims = [mlp_latent_size] + dims + [1]
        else:
            dims = [mlp_latent_size + 3] + dims + [1]  # <<<< 2 outputs instead of 1.

        self.num_layers = len(dims)
        self.num_class = num_class
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm
        self.use_classifier = use_classifier

        
        if self.use_FiLM:
            self.film_generator_beta = nn.Sequential(nn.Linear(3, self.film_size*(self.num_layers-3)))
            latent_in = []

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

            # classifier
            if self.use_classifier and layer == self.num_layers - 2:
                self.classifier_head = nn.Linear(dims[layer], self.num_class)

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

        self.upsample_mode = "trilinear" # "conv" or "trilinear"
        self.upsample_scale = 128.0
        self.latent_scale = 8.0
        
        if self.upsample_mode == "conv":
            k = self.latent_scale
            self.upsample_conv = nn.ModuleList()
            while k < self.upsample_scale:
                conv = torch.nn.Conv3d(self.latent_size,
                                    self.latent_size,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
                self.upsample_conv.append(conv)
                k *= 2

        self.latent_bn = nn.BatchNorm1d(self.latent_size)
        self.latent_map = nn.Linear(self.latent_size, self.latent_size)

            

    def forward(self, z, xyz, use_extended=False):
        # z: B x latent_dim x 16 x 16 x 16
        bs = z.shape[0]
        xyz = xyz / (util.SDF_MULTIPLIER)
        xyz = xyz / (0.52/2.0)

        temb = None

        # z to block_in
        z = self.conv_in(z)

        # top
        z = self.top.block_1(z, temb)
        z = self.top.attn_1(z)
        # z = self.top.block_2(z, temb)

        # middle
        z = self.mid.block_1(z, temb)
        z = self.mid.attn_1(z)
        z = self.mid.block_2(z, temb)

        # upsampling
        if self.upsample_mode == "trilinear":
            z = torch.nn.functional.interpolate(z, scale_factor=int(self.upsample_scale/z.shape[2]), mode="trilinear")
        else:
            for m in self.upsample_conv:
                z = torch.nn.functional.interpolate(z, scale_factor=2.0, mode="nearest")
                z = m(z)
        
        pos_x = (xyz[:, :, 0].unsqueeze(2) + 1.0)/2.0
        pos_y = (xyz[:, :, 1].unsqueeze(2) + 1.0)/2.0
        pos_z = (xyz[:, :, 2].unsqueeze(2) + 1.0)/2.0
        latent_pos_x = (self.upsample_scale * pos_x).long()
        latent_pos_y = (self.upsample_scale * pos_y).long()
        latent_pos_z = (self.upsample_scale * pos_z).long()
        latent_pos_x = torch.clamp(latent_pos_x, 0, self.upsample_scale-1)
        latent_pos_y = torch.clamp(latent_pos_y, 0, self.upsample_scale-1)
        latent_pos_z = torch.clamp(latent_pos_z, 0, self.upsample_scale-1)

        z = z.permute(0, 2, 3, 4, 1)
        latent_codes = torch.zeros((bs, xyz.shape[1], self.latent_size)).to(z.device)
        for i in range(bs):
            latent_codes[i] = z[i, latent_pos_x[i], latent_pos_y[i], latent_pos_z[i]].squeeze(1)

        sample_size = xyz.shape[1]

        if self.use_FiLM:
            xyz = xyz.view(-1, xyz.shape[2])
            film_vector_beta = self.film_generator_beta(xyz)
            x = latent_codes.view(-1, latent_codes.shape[2])
        else:

            xyz = xyz.view(-1, xyz.shape[2])
            latent_codes = latent_codes.view(-1, latent_codes.shape[2])
            latent_codes = self.latent_map(self.latent_bn(latent_codes))
            input = torch.cat([latent_codes, xyz], 1)
            x = input

        for layer in range(0, self.num_layers - 1):
            # classify
            if self.use_classifier and layer == self.num_layers - 2:
                predicted_class = self.classifier_head(x)

            lin = getattr(self, "lin" + str(layer))
            if not(self.use_FiLM):
                if layer in self.latent_in:
                    x = torch.cat([x, input], 1)
                elif layer != 0 and self.xyz_in_all:
                    x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)

                if layer > 0 and layer < self.num_layers - 3 and self.use_FiLM:
                    beta = film_vector_beta[:, (layer-1)*self.film_size:(layer)*self.film_size]
                    x = x + beta

                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = self.th(x)
        # Reconstruct batch
        x_sdf = x.view(-1, sample_size)

        # hand, object, class label
        if self.use_classifier:
            return x_sdf, predicted_class
        else:
            return x_sdf, torch.Tensor([0]).cuda()
