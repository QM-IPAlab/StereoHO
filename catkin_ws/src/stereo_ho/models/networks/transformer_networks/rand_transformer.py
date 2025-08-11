# Code for Transformer module
import math
import torch
import torch.nn as nn

# from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder, LayerNorm, TransformerDecoderLayer, TransformerDecoder
from torch.nn.modules.transformer import *

from einops import rearrange, repeat

from .pos_embedding import PEPixelTransformer, PEFourier

class RandTransformer(nn.Module):
    def __init__(self, tf_conf, vq_conf=None, two_stage=False, empty_idx_h=0, empty_idx_o=0):
        """init method"""
        super().__init__()

        # vqvae related params
        if vq_conf is not None:
            ntokens_vqvae = vq_conf.model.params.n_embed
            embed_dim_vqvae = vq_conf.model.params.embed_dim
        else:
            ntokens_vqvae = tf_conf.model.params.ntokens
            embed_dim_vqvae = tf_conf.model.params.embed_dim

        # pe
        pe_conf = tf_conf.pe
        pos_embed_dim = pe_conf.pos_embed_dim

        # tf
        mparam = tf_conf.model.params
        ntokens = mparam.ntokens
        d_tf = mparam.embed_dim
        nhead = mparam.nhead
        num_encoder_layers = mparam.nlayers_enc
        dim_feedforward = mparam.d_hid
        dropout = mparam.dropout
        self.ntokens_vqvae = ntokens_vqvae
        self.two_stage = two_stage
        self.empty_idx_h = empty_idx_h
        self.empty_idx_o = empty_idx_o
        self.empty_dist_h = torch.zeros(ntokens_vqvae)
        self.empty_dist_o = torch.zeros(ntokens_vqvae)
        self.empty_dist_h[empty_idx_h] = 1.0
        self.empty_dist_o[empty_idx_o] = 1.0

        # position embedding
        # self.pos_embedding = PEPixelTransformer(pe_conf=pe_conf)
        self.pos_embedding = PEFourier(pe_conf=pe_conf)
        self.fuse_linear = nn.Linear(embed_dim_vqvae+pos_embed_dim+pos_embed_dim, d_tf)

        # transformer
        encoder_layer = TransformerEncoderLayer(d_tf, nhead, dim_feedforward, dropout, activation='relu')
        encoder_norm = LayerNorm(d_tf)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        self.dec_linear = nn.Linear(d_tf, ntokens_vqvae)
        if self.two_stage:
            self.dec_occp_linear = nn.Linear(d_tf, 2)

        self.d_tf = d_tf

        self._init_weights()

    def _init_weights(self) -> None:
        """initialize the weights of params."""

        _init_range = 0.1

        self.fuse_linear.bias.data.normal_(0, 0.02)
        self.fuse_linear.weight.data.normal_(0, 0.02)

        self.dec_linear.bias.data.normal_(0, 0.02)
        self.dec_linear.weight.data.normal_(0, 0.02)

    def generate_square_subsequent_mask(self, sz, device):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)

    def generate_square_id_mask(self, sz, device):
        mask = torch.eye(sz)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)
        return mask

    def forward_transformer(self, src, src_mask=None):
        output = self.encoder(src, mask=src_mask)
        # output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return output

    def forward(self, inp_val, inp_posn, tgt_posn):
        """ Here we will have the full sequence of inp """
        device = inp_val.get_device()
        seq_len, bs = inp_val.shape[:2] # T, 1

        inp_posn_emb = repeat(self.pos_embedding(inp_posn), 't pos_d -> t bs pos_d', bs=bs) # T, 128 -> T, bs, 128
        tgt_posn_emb = repeat(self.pos_embedding(tgt_posn), 't pos_d -> t bs pos_d', bs=bs) # T, 128 -> T, bs, 128
        inp = torch.cat([inp_val, inp_posn_emb, tgt_posn_emb], dim=-1) # T, bs, 128+64+64

        # fusion
        inp = rearrange(inp, 't bs d -> (t bs) d') # T, 128+64+64
        inp = rearrange(self.fuse_linear(inp), '(t bs) d -> t bs d', t=seq_len, bs=bs) # T, 256 -> T, d_tf
        src_mask = self.generate_square_subsequent_mask(seq_len, device) # T x T diag 0

        outp_f = self.forward_transformer(inp, src_mask=src_mask) # T, bs, d_tf
        outp = self.dec_linear(outp_f) # T, bs, ntokens_vqvae
        if self.two_stage:
            empty_dist_h = self.empty_dist_h.to(device)
            empty_dist_o = self.empty_dist_o.to(device)
            outp_occp = self.dec_occp_linear(outp_f) # T, bs, 2
            occp_mask = torch.argmax(outp_occp, dim=-1) # T, bs
            empty_mask = occp_mask == 0 # T, bs
            tgt_posn_b = repeat(tgt_posn, 't pos_d -> t bs pos_d', bs=bs)
            h_mask = tgt_posn_b[:, :, 3] < 0.0 # T, bs
            o_mask = tgt_posn_b[:, :, 3] > 0.0 # T, bs
            occp_mask_h = torch.logical_and(empty_mask, h_mask) # T, bs
            occp_mask_o = torch.logical_and(empty_mask, o_mask) # T, bs
            outp_masked = outp.clone() # T, bs, ntokens_vqvae
            outp_masked[occp_mask_h] = empty_dist_h # T, bs, ntokens_vqvae
            outp_masked[occp_mask_o] = empty_dist_o # T, bs, ntokens_vqvae

            return outp, outp_occp, outp_masked
        else:
            return outp, None, outp



# From Pixel Transformer
class TransformerDecoderLayerNoSelfAttn(TransformerDecoderLayer):
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayerNoSelfAttn, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    
class TransformerEfficient(Transformer):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None) -> None:
        super(TransformerEfficient, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


class RandTransformerPE(nn.Module):
    def __init__(self, tf_conf, vq_conf=None):
        """init method"""
        super().__init__()
        # vqvae related params
        if vq_conf is not None:
            ntokens_vqvae = vq_conf.model.params.n_embed
            embed_dim_vqvae = vq_conf.model.params.embed_dim
        else:
            ntokens_vqvae = tf_conf.model.params.ntokens
            embed_dim_vqvae = tf_conf.model.params.embed_dim

        # pe
        pe_conf = tf_conf.pe
        pos_embed_dim = pe_conf.pos_embed_dim

        # tf
        mparam = tf_conf.model.params
        ntokens = mparam.ntokens
        d_tf = mparam.embed_dim
        nhead = mparam.nhead
        num_encoder_layers = mparam.nlayers_enc
        dim_feedforward = mparam.d_hid
        dropout = mparam.dropout
        self.ntokens_vqvae = ntokens_vqvae

        # position embedding
        # self.pos_embedding = PEPixelTransformer(pe_conf=pe_conf)
        self.pos_embedding = PEFourier(pe_conf=pe_conf)

        # transformer
        self.transformer = TransformerEfficient(d_tf, nhead, num_encoder_layers, num_encoder_layers, dim_feedforward, dropout)
        self.dec_linear = nn.Linear(d_tf, ntokens_vqvae)
    
    def generate_square_subsequent_mask(self, sz, device):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)
    
    def generate_square_id_mask(self, sz, device):
        mask = torch.eye(sz)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)
        return mask

    def forward(self, inp_val, inp_posn, tgt_posn):
        """ Here we will have the full sequence of inp """
        device = inp_val.get_device()
        seq_len, bs = inp_val.shape[:2]

        inp_posn_emb = repeat(self.pos_embedding(inp_posn), 't pos_d -> t bs pos_d', bs=bs) # T, 128 -> T, bs, 128
        tgt_posn_emb = repeat(self.pos_embedding(tgt_posn), 't pos_d -> t bs pos_d', bs=bs) # T, 128 -> T, bs, 128

        src = torch.cat([inp_val, inp_posn_emb], dim=-1)
        tgt = torch.cat([torch.zeros_like(inp_val), tgt_posn_emb], dim=-1)
        print("src.shape", src.shape)
        print("tgt.shape", tgt.shape)
        src_mask = self.generate_square_subsequent_mask(src.shape[0], device)
        tgt_mask = self.generate_square_id_mask(tgt.shape[0], device) # T x T diag 0
        print("src_mask.shape", src_mask.shape)
        print("tgt_mask.shape", tgt_mask.shape)

        outp = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask) # T, bs, d_tf
        print("outp.shape", outp.shape)
        outp = self.dec_linear(outp) # T, bs, ntokens_vqvae
        print("outp.shape", outp.shape)

        return outp, None, outp