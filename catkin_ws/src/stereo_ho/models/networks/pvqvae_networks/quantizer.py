""" adapted from: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange

class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1. / self.n_e, 1. / self.n_e)
        # self.embedding.weight.data.uniform_(-0.01, 0.01)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

        self.codebook_restart_step = 0
        self.codebook_restart_freq = 25
        self.codebook_usage = torch.zeros(self.n_e)
        self.prev_total_usage = 0.
        self.curr_total_usage = 0.

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False, is_voxel=False, cb_restart=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten

        
        if not is_voxel:
            z = rearrange(z, 'b c h w -> b h w c').contiguous()
        else:
            z = rearrange(z, 'b c d h w -> b d h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # print("------->Codebook restart step: ", self.codebook_restart_step)
        # if cb_restart:
        self.codebook_restart_step += 1
            # print("Codebook restart step: ", self.codebook_restart_step)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        # Get unique indices and their counts
        with torch.no_grad():
            unique_indices, counts = torch.unique(min_encoding_indices, return_counts=True)
            num_unique = unique_indices.shape[0]
            counts = counts.to(self.codebook_usage.device)
            unique_indices = unique_indices.to(self.codebook_usage.device)
            for idx, i in enumerate(unique_indices):
                self.codebook_usage[i] = 1
            total_usage = torch.sum(self.codebook_usage)/self.n_e
            

        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        if not is_voxel:
            z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        else:
            z_q = rearrange(z_q, 'b d h w c -> b c d h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            if not is_voxel:
                min_encoding_indices = min_encoding_indices.reshape(
                    z_q.shape[0], z_q.shape[2], z_q.shape[3])
            else:
                min_encoding_indices = min_encoding_indices.reshape(
                    z_q.shape[0], z_q.shape[2], z_q.shape[3], z_q.shape[4])

        with torch.no_grad():
            if self.codebook_restart_step % self.codebook_restart_freq == 0:
                # print("codebook_usage: ", self.codebook_usage)
                if cb_restart:
                    print("******************************[Codebook] Restarting codebook at step {}******************************".format(self.codebook_restart_step))
                    rs_cnt = 0
                    for i in range(self.n_e):
                        if self.codebook_usage[i] == 0:
                            # self.embedding.weight[i] = (torch.rand(self.e_dim).to(self.embedding.weight.device) - 0.5) * 0.02
                            self.embedding.weight[i] = torch.clone(z_flattened[torch.randint(0, z_flattened.shape[0], (1,))])
                            rs_cnt += 1
                    print(f"Restarted {rs_cnt} codebook entries out of {self.n_e} entries")
                    self.codebook_usage *= 0
                    self.codebook_restart_step = 0
                    # self.codebook_restart_freq += 100
                    self.codebook_restart_freq *= 2
                    # print("Total usage: ", total_usage)
        # if cb_restart:
        #     self.codebook_restart(z_flattened, min_encoding_indices, unique_indices, counts)

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices, num_unique, self.codebook_usage, total_usage)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
    
    def codebook_restart(self, z_flattened, min_encoding_indices, unique_indices, counts):
        with torch.no_grad():
            if self.codebook_restart_step % 256 == 0:
                print("******************************[Codebook] Restarting codebook******************************")
                # Find indx with max count, do not use max_idx for replacement
                # replace_indices = []
                # max_idx = unique_indices[torch.argmax(counts)]
                # for i in range(z_flattened.shape[0]):
                #     if min_encoding_indices[i] != max_idx:
                #         replace_indices.append(i)
                # print("max_idx: ", max_idx)
                # print("replace_indices: ", len(replace_indices))

                # rs_cnt = 0
                # for i in range(self.n_e):
                #     if self.codebook_usage[i] < 1:
                #         # self.embedding.weight[i] = (torch.randn(self.e_dim).to(self.embedding.weight.device) - 0.5) * 0.01
                #         self.embedding.weight[i] = torch.clone(z_flattened[torch.randint(0, z_flattened.shape[0], (1,))])
                #         rs_cnt += 1
                
                # Restart codebook if usage is below 0.5%
                # print("Top usage: ", torch.topk(self.codebook_usage, 10))
                rs_cnt = 0
                for i in range(self.n_e):
                    # if self.codebook_usage[i] < 0.00001:
                    if self.codebook_usage[i] < 1:
                        # self.embedding.weight[i] = (torch.randn(self.e_dim).to(self.embedding.weight.device) - 0.5) * 0.01
                        self.embedding.weight[i] = torch.clone(z_flattened[torch.randint(0, z_flattened.shape[0], (1,))])
                        # if len(replace_indices) == 0:
                        #     self.embedding.weight[i] = torch.clone(z_flattened[torch.randint(0, z_flattened.shape[0], (1,))])
                        # else:
                        #     self.embedding.weight[i] = torch.clone(z_flattened[replace_indices[torch.randperm(len(replace_indices))[0]]])
                        rs_cnt += 1
                print(f"Restarted {rs_cnt} codebook entries out of {self.n_e} entries")

                # Restart codebook of bottom 5% usage
                # usage_val, usage_idx = torch.sort(self.codebook_usage)
                # for i in range(int(self.n_e * 0.05)):
                #     idx = usage_idx[i]
                #     self.embedding.weight[idx] = torch.clone(z_flattened[usage_idx[torch.randint(int(self.n_e * 0.05), usage_idx.shape[0], (1,))]])

                self.codebook_restart_step = 0
                self.codebook_usage *= 0
            else:
                pass

