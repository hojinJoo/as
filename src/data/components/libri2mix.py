import json
import os
import pandas as pd
from collections import defaultdict

import torch
import torchaudio
from torch.utils.data import Dataset
from scipy import signal
from src.utils.transforms import stft
import numpy as np






class Libri2Mix(Dataset):
    def __init__(
        self,
        metadata_path: str = "/workspace/data/Libri2Mix/wav16k/max/metadata/mixture_train-360_mix_clean.csv",
        crop_size : int = 8000,
        test : bool = False,
        n_fft : int = 512,
        win_length : int = 512,
        hop_length : int = 125
    ):
        super().__init__()
        self.meta_data = pd.read_csv(metadata_path,encoding="ms932",sep=",")
        self.crop_size = crop_size
        self.test = test
        self.num_files = len(self.meta_data)
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length


    def __getitem__(self, index):
        if self.test :
            mixtureID,mixture_path,source_1_path,source_2_path,length = self.meta_data.values[index]
            

            mixture_wav,sample_rate = torchaudio.load(mixture_path)
            source_1,sample_rate = torchaudio.load(source_1_path)
            source_2,sample_rate = torchaudio.load(source_2_path)
            
            original_length = mixture_wav.size(1)
            
            
            mixture = stft(mixture_wav,fs=sample_rate,window_length=self.win_length,nfft=self.n_fft,hop_length=self.hop_length)
            source_1 = stft(source_1,fs=sample_rate,window_length=self.win_length,nfft=self.n_fft,hop_length=self.hop_length)
            source_2 = stft(source_2,fs=sample_rate,window_length=self.win_length,nfft=self.n_fft,hop_length=self.hop_length)
            
            
            sample = {"mixture": mixture, "source_1": source_1, "source_2": source_2,"original_length":original_length, "mix_id": mixtureID}
            
            return sample
            
            
        mixtureID,mixture_path,source_1_path,source_2_path,length = self.meta_data.values[index]

        mixture_wav,sample_rate = torchaudio.load(mixture_path)
        source_1,sample_rate = torchaudio.load(source_1_path)
        source_2,sample_rate = torchaudio.load(source_2_path)
        
        mixture_wav = mixture_wav.squeeze(0)[:sample_rate*10]
        source_1 = source_1.squeeze(0)[:sample_rate*10]
        source_2 = source_2.squeeze(0)[:sample_rate*10]

        crop_idx = torch.randint(0, mixture_wav.size(0) - self.crop_size, (1,))

        mixture_wav = mixture_wav[crop_idx:crop_idx + self.crop_size]
        source_1 = source_1[crop_idx:crop_idx + self.crop_size]
        source_2 = source_2[crop_idx:crop_idx + self.crop_size]
                    
        mixture = torch.abs(stft(mixture_wav,fs=sample_rate,window_length=self.win_length,nfft=self.n_fft,hop_length=self.hop_length))
        source_1 = torch.abs(stft(source_1,fs=sample_rate,window_length=self.win_length,nfft=self.n_fft,hop_length=self.hop_length))
        source_2 = torch.abs(stft(source_2,fs=sample_rate,window_length=self.win_length,nfft=self.n_fft,hop_length=self.hop_length))

        mixture = torch.pow(mixture,0.3)
        source_1 = torch.pow(source_1,0.3)
        source_2 = torch.pow(source_2,0.3)
        
        sample = {"mixture": mixture, "source_1": source_1, "source_2": source_2}
        
        return sample

    def __len__(self):
        return self.num_files

if __name__ =="__main__" :  
    
    a = Libri2Mix("/media/ssd1/users/hj/Libri2Mix/mixture_test_mix_clean_metadata_in_ssd.csv",test=True)
    mixture = a[0]['mixture']
    s1 = a[0]['source_1'].squeeze(0)
    s2 = a[0]['source_2'].squeeze(0)
    original_length = a[0]['original_length']
    print(f"original length : {a[0]['original_length']}")
    print(f"mixture : {mixture.shape}")
    to_tensor = mixture
    print(f"to tensor dtype : {to_tensor.dtype}")
    print(f"to tensor size : {to_tensor.size()}")
    print(f"abs dtype : {torch.abs(to_tensor).dtype}")
    print(f"abs size : {torch.abs(to_tensor).size()}")
    # print(f"to tensor view as real : {torch.view_as_real(to_tensor).size()}")
    
    
    # torchaudio.save("/workspace/as/test.wav",torch.from_numpy(to_np),16000)
    
    stacked = torch.stack([s1,s2],dim=0)
    stacked_wav = torch.from_numpy(istft(stacked,16000,512,512,125,original_length=83120))
    print(f"staked wav  : {stacked_wav.shape}")
    
    to_tensor_1 = s1.unsqueeze(0)
    to_np_1 = torch.from_numpy(istft(to_tensor_1.numpy(),16000,512,512,125,original_length=83120))
    
    # torchaudio.save("/workspace/as/test_1.wav",torch.from_numpy(to_np_1),16000)
    
    to_tensor_2 = s2.unsqueeze(0)
    to_np_2 = istft(to_tensor_2.numpy(),16000,512,512,125,original_length=83120)
    # torchaudio.save("/workspace/as/test_2.wav",torch.from_numpy(to_np_2),16000)
    print(f"same : {torch.sum(stacked_wav[0,:] == to_np_1)}")
    
    
    
    # print(mixture[masks[0]].size())
    # print(ibm.size())
    # print(gt.size())
    # segment_step = 65
    
    # prediction = torch.zeros((1, 2,mixture.size(1),mixture.size(2)))
    
    # for segment in range(0, a[0]['mixture'].size(2), segment_step):
    #     semi = a[0]['mixture'][:, :,segment:segment + segment_step]
    #     out = torch.ones(( 2, mixture.size(1),segment_step))
    #     semi_original_size = semi.size()
    #     if semi.size(2) < segment_step:
    #         print("HERERERERE")
    #         print(semi.size(2))
    #         # zero pad
    #         semi = torch.nn.functional.pad(semi, (0, segment_step - semi.size(2)))
    #         out = out[:, :, : semi_original_size[2]]
    #         print('ioutotuotuto',out.size())
    #     # print(prediction[:,:,:,segment:segment + segment_step].size())
    #     print(prediction[:,:,:,segment:segment + segment_step].size())
    #     print(out.size())
    #     prediction[:,:,:,segment:segment + segment_step] = out
        
    # print( (prediction==1).sum())
    
# 1에폭 :50800, 1 step 64,
# 