import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck

from src.models.components.audioSlots.as_encoder import Backbone
from src.models.components.audioSlots.as_encoder import SlotAttention
from src.models.components.audioSlots.as_decoder import As_Decoder

# from as_encoder import Backbone, SlotAttention
# from as_decoder import AS_Decoder

class AudioSlot(nn.Module) :
    def __init__(self,
                num_slots : int = 2,
                num_iterations : int = 7,
                num_attn_heads : int = 1,
                slot_dim : int = 1024,
                mlp_hid_dim : int = 2048,
                eps : float = 1e-8,
                num_fourier_bases : int = 1024,
                input_ft : list = [257,65]) :
        super().__init__()
        self.backbone = Backbone(BasicBlock, [3, 4, 6, 3])    
        self.slot_attention = SlotAttention(num_slots=num_slots,num_iterations=num_iterations,num_attn_heads=num_attn_heads,slot_dim=slot_dim,mlp_hid_dim=mlp_hid_dim,eps=eps)
        self.decoder = As_Decoder(slot_dim=slot_dim,num_fourier_bases=num_fourier_bases,input_ft=input_ft)    
        
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
        # x = x.tile((1, 257, 65, 1)) # original
        x = x.tile((1, 257, 65, 1)) # TODO: repeat으롷 바꾸기
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
