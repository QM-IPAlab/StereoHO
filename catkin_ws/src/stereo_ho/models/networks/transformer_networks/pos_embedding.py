
import math
import torch
import torch.nn as nn

from einops import rearrange, repeat

class PEPixelTransformer(nn.Module):
	"""Returns the positional embeddings for tokens."""

	# def __init__(self, d_model: int, dropout: float = 0.1, h):
	def __init__(self, pe_conf=None):
		super().__init__()

		pos_dim = pe_conf.pos_dim
		pos_embed_dim = pe_conf.pos_embed_dim
		assert pos_embed_dim % 2 == 0, 'require even embedding dimension'
		
		# pos_dim = grid_flat.shape[0]
		self.proj_layer = nn.Parameter(torch.randn(pos_dim, pos_embed_dim // 2) * pe_conf.init_factor) # proj_layer is a matrix to look up the proj of the certain coor.
		self.proj_layer.requires_grad = False

	def forward(self, pos):
		"""
		Args:
			pos (coordinate): BS, pos_dim
		Returns:
			pos_proj: BS, pos_embed_dim
		"""
		pos_proj = torch.matmul(2 * math.pi * pos, self.proj_layer)
		pos_proj = torch.cat([torch.sin(pos_proj), torch.cos(pos_proj)], dim=-1)

		return pos_proj
	
class PEFourier(nn.Module):
	"""Returns the positional embeddings for tokens."""

	# def __init__(self, d_model: int, dropout: float = 0.1, h):
	def __init__(self, pe_conf=None):
		super().__init__()

		self.pos_dim = pe_conf.pos_dim
		self.pos_embed_dim = pe_conf.pos_embed_dim
		self.N = int((self.pos_embed_dim / self.pos_dim) // 2)
		assert self.pos_embed_dim % 2 == 0, 'require even embedding dimension'
		self.scale = torch.zeros(self.N)
		for i in range(self.N):
			self.scale[i] = (2 ** i) * math.pi
		self.scale = self.scale.repeat(self.pos_dim)
		self.scale = nn.Parameter(self.scale)
		self.scale.requires_grad = False

	def forward(self, pos):
		"""
		Args:
			pos (coordinate): BS, pos_dim
		Returns:
			pos_proj: BS, pos_embed_dim
		"""
		B = pos.shape[0]
		pos = pos.repeat_interleave(self.N, dim=1)
		scale = self.scale.repeat(B, 1)
		pos = pos * scale
		pos_proj = torch.cat([torch.sin(pos), torch.cos(pos)], dim=-1)

		return pos_proj