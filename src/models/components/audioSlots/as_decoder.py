# input size [64,4,512] [B,N,C]
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, Tuple


class As_Decoder(nn.Module):
    def __init__(self, slot_dim,num_fourier_bases,input_ft,dec_mlp_hid_dim,cac=False,channels=2):
        super().__init__()
        self.dim_after_pos = slot_dim + 2 * num_fourier_bases
        self.pos_embedder = PositionEmbedding(input_ft=input_ft,num_fourier_bases=num_fourier_bases)
        
        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.dim_after_pos, dec_mlp_hid_dim),
            nn.ReLU(),
            nn.Linear(dec_mlp_hid_dim,dec_mlp_hid_dim),
            nn.ReLU(),
            nn.Linear(dec_mlp_hid_dim,dec_mlp_hid_dim),
            nn.ReLU(),
            nn.Linear(dec_mlp_hid_dim,dec_mlp_hid_dim),
            nn.ReLU(),
            nn.Linear(dec_mlp_hid_dim,dec_mlp_hid_dim),
            nn.ReLU()
        )
        if cac : 
            self.last = nn.Linear(dec_mlp_hid_dim,channels * 2)
        else :
            self.last = nn.Linear(dec_mlp_hid_dim,channels)
    def forward(self, x):
        x = self.pos_embedder(x)
        x = self.decoder_mlp(x)
        x = self.last(x).squeeze(-1)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, input_ft,num_fourier_bases=3):
        super(PositionEmbedding, self).__init__()
        self.num_fourier_bases = num_fourier_bases
        self.pos_embedding = self._make_pos_embedding_tensor(input_ft)
        
    def _create_gradient_grid(self,samples_per_dim, value_range=(-1.0, 1.0)):
        s = [np.linspace(value_range[0], value_range[1], n) for n in samples_per_dim]
        pe = np.stack(np.meshgrid(*s, indexing="ij"), axis=-1)
        pe = pe.astype(np.float32)
        return torch.from_numpy(pe)

    
    def _make_pos_embedding_tensor(self,  input_shape):
        F,T = input_shape
        pos_embedding = self._create_gradient_grid((F,T), [-1.0, 1.0]) # [F,T,2]
        num_dims = pos_embedding.shape[-1]
        projection = torch.normal(0.0, 1.0, size=(num_dims, self.num_fourier_bases))
        pos_embedding = torch.pi * torch.matmul(pos_embedding, projection)
        pos_embedding = torch.cat([torch.sin(pos_embedding), torch.sin(pos_embedding + 0.5 * torch.pi)], dim=-1)

        # Add batch dimension.
        pos_embedding = pos_embedding.unsqueeze(0)
        return pos_embedding

    def forward(self, inputs):
        # input shape : B*N_slots,H,W,C
        pos_embedding = self.pos_embedding.to(inputs.device)
        pos_embedding = pos_embedding.expand(inputs.shape[:-1] + pos_embedding.shape[-1:])
        # pos_embedding = pos_embedding.to(inputs.device)
        x = torch.cat((inputs, pos_embedding), dim=-1)
        return x
    
    
    
if __name__ == "__main__":
    # a = SoftPositionEmbed(64,(10,20))
    pos_embedder = PositionEmbedding((10,20),num_fourier_bases=3)
    sample = torch.rand(128, 10,20,100)
    print(pos_embedder(sample).shape)
    # pos_embedder = PositionalEncoder1(512, 1)
    # print(pos_embedder(torch.rand(64, 4, 512)).shape)