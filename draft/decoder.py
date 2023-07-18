# input size [64,4,512] [B,N,C]
import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, args, num_slots, hidden_dim):
        super().__init__()
        self.dim_after_pos = hidden_dim * (1 + 2 * 1)

        self.pos_embedder = PositionalEncoder(hidden_dim, 1)
        self.decoder_pos = nn.Linear(self.dim_after_pos, 1,device='cuda')

    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(2)
        x = x.repeat((1, 1, 257, 65, 1))
        x = self.pos_embedder(x)
        x = self.decoder_pos(x).squeeze(-1)
        return x


class PositionalEncoder(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

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
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs, device='cuda')
        else:
            freq_bands = torch.linspace(
                2.**0., 2.**(self.n_freqs - 1), self.n_freqs,device='cuda')

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(
        self,
        x
    ) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)

def build_decoder() :
    return Decoder(1, 1, 512)

if __name__ == "__main__":
    sample = torch.randn((16, 4, 512),device='cuda')
    model = Decoder(1, 1, 512)
    model(sample)
