import torch
import torch.nn as nn

from src.models.components.audioSlots.resnet import get_backbone
from src.models.components.audioSlots.as_encoder import SlotAttention
from src.models.components.audioSlots.as_decoder import As_Decoder
from src.utils.transforms import torch_isfft as istft
import sys
sys.path.append("/workspace/as")   
from nussl import nussl

# from resnet import get_backbone
# from as_encoder import Backbone, SlotAttention
# from as_decoder import As_Decoder

class AudioSlot(nn.Module) :
    def __init__(self,
                num_slots : int = 2,
                num_iterations : int = 7,
                num_attn_heads : int = 1,
                hid_dim : int = 2048,
                slot_dim : int = 1024,
                mlp_hid_dim : int = 2048,
                eps : float = 1e-8,
                num_fourier_bases : int = 1024,
                input_ft : list = [257,65],
                dec_mlp_hid_dim : int = 256,
                resnet : str = '34',
                cac : bool = False,
                channels : int = 2,
                init_method : str = 'clustering'
        ) :
        super().__init__()
        self.cac = cac
        self.channels = channels
        self.backbone = get_backbone(resnet=resnet,cac=cac,channles=channels)
        self.input_ft = input_ft
        self.slot_attention = SlotAttention(num_slots=num_slots,num_iterations=num_iterations,num_attn_heads=num_attn_heads,slot_dim=slot_dim,mlp_hid_dim=mlp_hid_dim,eps=eps,hid_dim=hid_dim,init_method=init_method)
        self.decoder = As_Decoder(slot_dim=slot_dim,num_fourier_bases=num_fourier_bases,input_ft=input_ft,dec_mlp_hid_dim=dec_mlp_hid_dim,cac=cac,channels=channels)    
        self.init_method = init_method
        if self.init_method == 'clustering' :
            self.clusterer = Clusterer()
            
    def forward(self,mix,train=True) :
        if self.cac :
            B,C,Fr,T = x.size()
            x = torch.view_as_real(x).permute(0,1,4,2,3)
            x = x.reshape(B,C * 2, Fr, T).squeeze(1)
        
        else : 
            x = torch.pow(mix,0.3)
            x = torch.abs(x)
            
        
        self.backbone = self.backbone.to(x.device)
        x = self.backbone(x)
        B,C,F,T = x.size()
        if self.init_method == 'clustering':
            cluster = self.clusterer(mix).to(mix.device)
        
        x = x.permute(0,2,3,1)
        x = torch.flatten(x, 1,2)
        if self.init_method == 'clustering':
            x = self.slot_attention(x,cluster=cluster,train=train)
        else :
            x = self.slot_attention(x,train=train)
        slots = x['slots'] # [B,N_slots,C]
        attention = x['attn'] # [B,N_slots,N_slots]

        B,N_slots,C_slot = slots.size()
        x = slots.reshape(B*N_slots,C_slot)
        
        x = x[:,None,None,:] # [B*N_slots,1,1,C_slot]
        x = x.repeat((1, self.input_ft[0], self.input_ft[1], 1))
        
        x = self.decoder(x) # [B*N_slots,Fr,T,C * 2 (cac)]
        
        x = x.permute(0,3,1,2)  # [B*N_slots,C * 2,Fr,T]
        if self.cac :
            x = x.reshape(B,N_slots,2,2,self.input_ft[0], self.input_ft[1]).permute(0,1,2,4,5,3)
        else  :
            x = x.reshape(B,N_slots,2,self.input_ft[0], self.input_ft[1])
        return x, attention

class Clusterer(nn.Module) :
    def __init__(self) :
        super().__init__()
        #TODO: TEST시에 original length로 istft를 해야함    

    def forward(self,x) :
        clusters = []
        for i in range(x.size(0)) :
            sample = x[i]
            audio_signal = nussl.AudioSignal(audio_data_array=istft(sample,nfft=4096,hop_length=1024,original_length=44100).cpu().numpy(),sample_rate=44100)
            audio_signal.stft_data = sample.permute(1,2,0).cpu().numpy()

            separators = [
                nussl.separation.primitive.FT2D(audio_signal),
                nussl.separation.primitive.HPSS(audio_signal),
                nussl.separation.factorization.RPCA(audio_signal),
            ]
            weights = [1, 1,1]
            returns = [[1], [1],[1]]
            ensemble = nussl.separation.composite.EnsembleClustering(
                audio_signal, 2, separators=separators, 
                fit_clusterer=True, weights=weights, returns=returns,extracted_feature='estimates')
            feat = ensemble.extract_features()
            cluster = torch.from_numpy(ensemble.cluster_features(feat,ensemble.clusterer))
            
            clusters.append(cluster)
        clusters = torch.stack(clusters)
        return clusters
    
        
        
def clusterer(x) :
    audio_signal = nussl.AudioSignal(audio_data_array=istft(x,nfft=4096,hop_length=1024,original_length=44100).numpy(),sample_rate=44100)
    audio_signal.stft_data = mix.permute(1,2,0).numpy()

    separators = [
        nussl.separation.primitive.FT2D(audio_signal),
        nussl.separation.primitive.HPSS(audio_signal),
        nussl.separation.factorization.RPCA(audio_signal),
    ]
    weights = [1, 1,1]
    returns = [[1], [1],[1]]
    ensemble = nussl.separation.composite.EnsembleClustering(
        audio_signal, 2, separators=separators, 
        fit_clusterer=True, weights=weights, returns=returns,extracted_feature='estimates')
    feat = ensemble.extract_features()
    cluster = ensemble.cluster_features(feat,ensemble.clusterer)
    return cluster

if __name__=="__main__" :
    sample_wav = torch.randn(8000)
    after_stft = torch.stft(sample_wav, n_fft=512, win_length=512,
                            hop_length=125, return_complex=True)
    after_stft = torch.abs(after_stft)
    
    sample = after_stft.unsqueeze(0).repeat((16, 1, 1))
    fullModel = AudioSlot()
    out = fullModel(sample)
    print(out[0].size())
