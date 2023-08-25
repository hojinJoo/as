import json
import os
import pandas as pd
from collections import defaultdict

import torch
import torchaudio
from torch.utils.data import Dataset


class Libri2Mix(Dataset):
    def __init__(
        self,
        metadata_path: str = "/workspace/data/Libri2Mix/wav16k/max/metadata/mixture_train-360_mix_clean.csv",
        crop_size : int = 8000,
        test : bool = False
    ):
        super().__init__()
        self.meta_data = pd.read_csv(metadata_path,encoding="ms932",sep=",")
        self.crop_size = crop_size
        self.test = test
        self.num_files = len(self.meta_data)



    def __getitem__(self, index):
        if self.test :
            mixtureID,mixture_path,source_1_path,source_2_path,length = self.meta_data.values[index]
            
            # mixture_path = mixture_path.replace("/media","/media/NAS3/CIPLAB/users/hj")
            # source_1_path = source_1_path.replace("/media","/media/NAS3/CIPLAB/users/hj")
            # source_2_path = source_2_path.replace("/media","/media/NAS3/CIPLAB/users/hj")
            
            mixture_wav,sample_rate = torchaudio.load(mixture_path)
            source_1,sample_rate = torchaudio.load(source_1_path)
            source_2,sample_rate = torchaudio.load(source_2_path)
            
            mixture = torch.abs(torch.stft(mixture_wav, n_fft=512, win_length=512,
                                hop_length=125, return_complex=True))
            source_1 = torch.abs(torch.stft(source_1, n_fft=512, win_length=512,
                                    hop_length=125, return_complex=True))
            source_2 = torch.abs(torch.stft(source_2, n_fft=512, win_length=512,
                                    hop_length=125, return_complex=True))             
            mixture = torch.pow(mixture,0.3)
            source_1 = torch.pow(source_1,0.3)
            source_2 = torch.pow(source_2,0.3)
            
            
            sample = {"mixture": mixture, "source_1": source_1, "source_2": source_2}
            
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
        
        mixture = torch.abs(torch.stft(mixture_wav, n_fft=512, win_length=512,
                                hop_length=125, return_complex=True))
        source_1 = torch.abs(torch.stft(source_1, n_fft=512, win_length=512,
                                hop_length=125, return_complex=True))
        source_2 = torch.abs(torch.stft(source_2, n_fft=512, win_length=512,
                                hop_length=125, return_complex=True))             
        
        mixture = torch.pow(mixture,0.3)
        source_1 = torch.pow(source_1,0.3)
        source_2 = torch.pow(source_2,0.3)
        
        sample = {"mixture": mixture, "source_1": source_1, "source_2": source_2}
        
        return sample

    def __len__(self):
        return self.num_files

if __name__ =="__main__" :  
    a = Libri2Mix("/media/NAS3/CIPLAB/users/hj/Libri2Mix/mixture_test_mix_clean_metadata.csv",test=True)
    mixture = a[0]['mixture']
    s1 = a[0]['source_1']
    s2 = a[0]['source_2']
    
    gt = torch.cat((s1.unsqueeze(0),s2.unsqueeze(0)),dim=1)
    
    ibm = (gt == torch.max(gt, dim=1, keepdim=True).values).float()
    print(ibm.size())
    ibm = ibm / torch.sum(ibm, dim=1, keepdim=True)
    print(ibm.size())
    ibm[ ibm <= 0.5] = 0
    masks = []
    for i in range(2):
        mask = ibm[:, i, :, :]
        masks.append(mask)
    
    print((ibm==0).sum())
    print((ibm==1).sum())
    
    aa = ibm * mixture
    print(aa.size())
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