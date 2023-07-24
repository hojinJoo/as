import torch
import torch.nn as nn
from decoder import build_decoder
from resnet import build_resnet
from transformer import build, PositionEmbeddingLearned

class fullModel(nn.Module) :
    def __init__(self, args, num_slots, hidden_dim) :
        super().__init__()
        self.encoder = build_resnet().to(device='cuda')
        self.decoder = build_decoder().to(device='cuda')
        self.transformer = build().to(device='cuda')
        self.learnable_pos = PositionEmbeddingLearned(256).to(device='cuda')
        
    def forward(self,x) :
        x = self.encoder(x)
        pos = self.learnable_pos(x)
        x = self.transformer(x,pos)
        input(x.size())
        x = self.decoder(x)
        input(x.size())
        return x

if __name__=="__main__" :
    sample_wav = torch.randn(8000,device='cuda')
    after_stft = torch.stft(sample_wav, n_fft=512, win_length=512,
                            hop_length=125, return_complex=True)
    after_stft = torch.abs(after_stft)
    
    sample = after_stft.unsqueeze(0).repeat((16, 1, 1, 1))
    model = fullModel(1, 4, 512)
    print(model(sample).size())
