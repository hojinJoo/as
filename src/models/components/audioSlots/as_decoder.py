# input size [64,4,512] [B,N,C]
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, Tuple


class As_Decoder(nn.Module):
    def __init__(self, hidden_dim,pos_embed_size):
        super().__init__()
        self.pos_embed_size = pos_embed_size
        self.dim_after_pos = hidden_dim + 2 * pos_embed_size
        self.pos_embedder = PositionEmbedding(num_fourier_bases=pos_embed_size)
        self.decoder_mlp = nn.Linear(self.dim_after_pos, 1)
    def forward(self, x):
        x = self.pos_embedder(x)
        x = self.decoder_mlp(x).squeeze(-1)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, num_fourier_bases=3,
                 gaussian_sigma=1.0):
        super(PositionEmbedding, self).__init__()
        self.num_fourier_bases = num_fourier_bases
        self.gaussian_sigma = gaussian_sigma
        
        
    def _create_gradient_grid(self,samples_per_dim, value_range=(-1.0, 1.0)):
        s = [np.linspace(value_range[0], value_range[1], n) for n in samples_per_dim]
        pe = np.stack(np.meshgrid(*s, indexing="ij"), axis=-1)
        return torch.tensor(pe, dtype=torch.float32)

    
    def _make_pos_embedding_tensor(self,  input_shape):
        B,F,T,C = input_shape
        pos_embedding = self._create_gradient_grid((F,T), [-1.0, 1.0]) # [F,T,2]
        num_dims = pos_embedding.shape[-1]
        
        projection = torch.normal(0.0, 1.0, size=(num_dims, self.num_fourier_bases)) * self.gaussian_sigma
        pos_embedding = torch.pi * torch.matmul(pos_embedding, projection)
        pos_embedding = torch.cat([torch.sin(pos_embedding), torch.sin(pos_embedding + 0.5 * torch.pi)], dim=-1)

        # Add batch dimension.
        pos_embedding = pos_embedding.unsqueeze(0)
        return pos_embedding

    def forward(self, inputs):
        # input shape : B*N_slots,H,W,C
        # Compute the position embedding only in the initial call using the same rng
        # as is used for initializing learnable parameters.
        pos_embedding = self._make_pos_embedding_tensor(inputs.shape)
        
        # pos_embedding = pos_embedding.detach()

        pos_embedding = pos_embedding.expand(inputs.shape[:-1] + pos_embedding.shape[-1:])
        pos_embedding = pos_embedding.to(inputs.device)
        x = torch.cat((inputs, pos_embedding), dim=-1)
        
        return x
    
    
    
if __name__ == "__main__":
    # a = SoftPositionEmbed(64,(10,20))
    pos_embedder = PositionEmbedding()
    sample = torch.rand(123, 10,20,100)
    print(pos_embedder(sample).shape)
    # pos_embedder = PositionalEncoder1(512, 1)
    # print(pos_embedder(torch.rand(64, 4, 512)).shape)