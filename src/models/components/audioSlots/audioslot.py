import torch
import torch.nn as nn
from src.models.components.audioSlots.as_encoder import AS_Encoder
from src.models.components.audioSlots.as_decoder import AS_Decoder

class AudioSlot(nn.Module) :
    def __init__(self,
                train=True) :
        super().__init__()
        self.encoder = AS_Encoder(1,512)
        self.decoder = AS_Decoder(1,1,512)
        
        
    def forward(self,x) :
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__=="__main__" :
    sample_wav = torch.randn(8000,device='cuda')
    after_stft = torch.stft(sample_wav, n_fft=512, win_length=512,
                            hop_length=125, return_complex=True)
    after_stft = torch.abs(after_stft)
    
    sample = after_stft.unsqueeze(0).repeat((16, 1, 1, 1))
    model = fullModel(1, 4, 512)
    print(model(sample).size())
