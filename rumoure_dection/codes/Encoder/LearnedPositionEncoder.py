import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


class LearnedPositionEncoder(nn.Module):
  """

	This set of codes would encode the structural information

	"""

  def __init__(self, config, n_heads):
    super(LearnedPositionEncoder, self).__init__()

    self.config = config
    self.n_heads = n_heads
    self.d_emb_dim = self.config.emb_dim // self.n_heads
    self.n_pos = self.config.num_structure_index + 1  # +1 for padding

    # <------------- Defining the position embedding ------------->
    self.structure_emb = nn.Embedding(self.n_pos, self.d_emb_dim)
    self.structure_emb.requries_grad = True

  def forward(self, src_seq):
    # <------------- Get the shape ------------->
    batch_size, num_posts, num_posts = src_seq.shape

    # <------------- Duplicate the src_seq based on the number of heads first ------------->
    src_seq = src_seq.repeat(self.n_heads, 1, 1)
    #batsize*n_heads x num_posts x num_posts
    encoded_structure_features = self.structure_emb(src_seq)
    #batsize*n_heads x num_posts x num_posts x emb_dim/n_heads(300/2=150 )
    del src_seq
    torch.cuda.empty_cache()

    # <------------- Break into individual heads ------------->
    encoded_structure_features = encoded_structure_features.view(batch_size, self.n_heads, num_posts, num_posts, -1)
    #batchsize x n_heads x nums_posts x nums_posts x 150
    return encoded_structure_features
