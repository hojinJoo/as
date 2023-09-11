import torch
import torch.nn as nn

from src.models.components.audioSlots.resnet import get_backbone
from src.models.components.audioSlots.as_encoder import SlotAttention
from src.models.components.audioSlots.as_decoder import As_Decoder

# from resnet import get_backbone
# from as_encoder import Backbone, SlotAttention
# from as_decoder import As_Decoder

class AudioSlot(nn.Module) :
    def __init__(self,
                num_slots : int = 2,
                num_iterations : int = 7,
                num_attn_heads : int = 1,
                slot_dim : int = 1024,
                mlp_hid_dim : int = 2048,
                eps : float = 1e-8,
                num_fourier_bases : int = 1024,
                input_ft : list = [257,65],
                dec_mlp_hid_dim : int = 256,
                resnet : str = '34',
                cac : bool = False,
                channels : int = 2) :
        super().__init__()
        self.cac = cac
        self.channels = channels
        self.backbone = get_backbone(resnet=resnet,cac=cac,channles=channels)
        self.input_ft = input_ft
        self.slot_attention = SlotAttention(num_slots=num_slots,num_iterations=num_iterations,num_attn_heads=num_attn_heads,slot_dim=slot_dim,mlp_hid_dim=mlp_hid_dim,eps=eps)
        self.decoder = As_Decoder(slot_dim=slot_dim,num_fourier_bases=num_fourier_bases,input_ft=input_ft,dec_mlp_hid_dim=dec_mlp_hid_dim,cac=cac,channels=channels)    
        
        
    def forward(self,x,train=True) :
        if self.cac :
            B,C,Fr,T = x.size()
            x = torch.view_as_real(x).permute(0,1,4,2,3)
            x = x.reshape(B,C * 2, Fr, T).squeeze(1)
            print(f"{self.cac} and {x.size()}")
        else : 
            x = torch.abs(x)
        
        self.backbone = self.backbone.to(x.device)
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
        x = x.repeat((1, self.input_ft[0], self.input_ft[1], 1))
        x = self.decoder(x)
        print(f"audioslot ln 59 size : {x.size()}")
        x = x.reshape(B,N_slots,self.input_ft[0], self.input_ft[1])
        print(f"audioslot ln 61 size : {x.size()}")
        return x, attention

if __name__=="__main__" :
    sample_wav = torch.randn(8000)
    after_stft = torch.stft(sample_wav, n_fft=512, win_length=512,
                            hop_length=125, return_complex=True)
    after_stft = torch.abs(after_stft)
    
    sample = after_stft.unsqueeze(0).repeat((16, 1, 1))
    fullModel = AudioSlot()
    out = fullModel(sample)
    print(out[0].size())
