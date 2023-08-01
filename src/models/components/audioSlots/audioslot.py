import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck

from src.models.components.audioSlots.as_encoder import Backbone
from src.models.components.audioSlots.as_encoder import SlotAttention
from src.models.components.audioSlots.as_decoder import As_Decoder

# from as_encoder import Backbone, SlotAttention
# from as_decoder import AS_Decoder

class AudioSlot(nn.Module) :
    def __init__(self,) :
        super().__init__()
        self.backbone = Backbone(BasicBlock, [3, 4, 6, 3])    
        self.slot_attention = SlotAttention()
        self.decoder = As_Decoder(64,32)    
        
    def forward(self,x,train=True) :
        x = self.backbone(x)
        B,C,F,T = x.size()
        
        x = x.permute(0,2,3,1)
        x = torch.flatten(x, 1,2)
        x = self.slot_attention(x,train=train)
        
        slots = x['slots'] # [B,N_slots,C]
        attention = x['attn'] # [B,N_slots,N_slots]

        B,N_slots,C = slots.size()
        x = slots.reshape(B*N_slots,C)
        x = x[:,None,None,:] # [B*N_slots,1,1,C]
        x = x.tile((1, 257, 65, 1))
        x = self.decoder(x)
        x = x.reshape(B,N_slots,257,65)
        return x, attention

if __name__=="__main__" :
    sample_wav = torch.randn(8000)
    after_stft = torch.stft(sample_wav, n_fft=512, win_length=512,
                            hop_length=125, return_complex=True)
    after_stft = torch.abs(after_stft)
    
    sample = after_stft.unsqueeze(0).repeat((16, 1, 1))
    fullModel = AudioSlot()
    out = fullModel(sample)
    print(out.size())
