# input size [64,4,512] [B,N,C]
import torch
import torch.nn as nn


class AS_Decoder(nn.Module):
    def __init__(self, args, num_slots, hidden_dim):
        super().__init__()
        self.dim_after_pos = hidden_dim * (1 + 2 * 1)

        self.pos_embedder = PositionalEncoder(hidden_dim, 1)
        self.decoder_pos1 = nn.Linear(self.dim_after_pos, self.dim_after_pos * 2)
        self.decoder_pos2 = nn.Linear(self.dim_after_pos * 2, self.dim_after_pos * 2)
        self.decoder_pos3 = nn.Linear(self.dim_after_pos * 2, 1)
    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(2)
        x = x.repeat((1, 1, 257, 65, 1))
        x = self.pos_embedder(x)
        x = self.decoder_pos1(x)
        x = self.decoder_pos2(x)
        x = self.decoder_pos3(x).squeeze(-1)
        return x


# class PositionalEncoder(nn.Module):
#     def __init__(
#         self,
#         d_input: int,
#         n_freqs: int,
#         log_space: bool = False
#     ):
#         super().__init__()
#         self.d_input = d_input
#         self.n_freqs = n_freqs
#         self.log_space = log_space
#         self.d_output = d_input * (1 + 2 * self.n_freqs)
#         self.embed_fns = [lambda x: x]

#         # Define frequencies in either linear or log scale
#         if self.log_space:
#             freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
#         else:
#             freq_bands = torch.linspace(
#                 2.**0., 2.**(self.n_freqs - 1), self.n_freqs)
        
#         # Alternate sin and cos
#         for freq in freq_bands:
#             self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
#             self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

#     def forward(
#         self,
#         x
#     ) -> torch.Tensor:
       
#         return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)


class PositionalEncoder(nn.Module):
    def __init__(
        self,
        d_input: int,
        n_freqs: int,
        log_space: bool = False
    ):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)

        # Define frequencies in either linear or log scale
        if self.log_space:
            self.freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            self.freq_bands = torch.linspace(
                2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        # for freq in freq_bands:
        #     self.embed_fns.append(self.get_sin_embedding(freq))
        #     self.embed_fns.append(self.get_cos_embedding(freq))

    def forward(self, x) -> torch.Tensor:
        ret = x
        for freq in self.freq_bands :
            ret = torch.cat([ret,torch.sin(x * freq)], dim=-1)
            ret = torch.cat([ret,torch.cos(x * freq)], dim=-1)
        return ret
    
if __name__ == "__main__":
    pos_embedder = PositionalEncoder(512, 1)
    print(pos_embedder(torch.rand(64, 4, 512)).shape)
    pos_embedder = PositionalEncoder1(512, 1)
    print(pos_embedder(torch.rand(64, 4, 512)).shape)