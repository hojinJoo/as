import json
import os
import pandas as pd
from collections import defaultdict

import torch
import torchaudio
from torch.utils.data import Dataset


class MusDB(Dataset):
    def __init__(
        self,
        metadata_path: str = "/media/musdb18hq/musdb_wav.json",
        tracks : str,
        crop_size : int = 8000,
    ):
        super().__init__()
        
        self.data_path = "/".join(metadata_path.split("/")[:-1]) + "/train" if mode == 'train' or 'val' else "/".join(metadata_path.split("/")[:-1]) + "/test"
        with open(metadata_path, "r") as f:
            self.meta_data = json.load(f)
        self.crop_size = crop_size
        # 이거는 datamodule에서 불러오기
        # train_tracks = get_musdb_tracks(args.musdb, is_wav=True, subsets=["train"], split="train")

        
        self.num_files = len(self.tracks)

    def _get_musdb_track(self,root,*args,**kwargs):
        mus = musdb.DB(root=root,*args,**kwargs)
        return {track.name: track.path for track in mus}



    def __getitem__(self, index):
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
        mixture_normarlized = mixture / torch.max(mixture)
        source_1 = torch.pow(source_1,0.3)
        source_2 = torch.pow(source_2,0.3)
        
        sample = {"mixture": mixture, "source_1": source_1, "source_2": source_2}
        
        return sample

    def __len__(self):
        return self.num_files

if __name__ =="__main__" :
    a = Libri2Mix("/workspace/data/Libri2Mix/wav16k/max/metadata/mixture_train-360_mix_clean.csv")
    print(a[0]['mixture'].size())
    print(a[0]['source_1'].size())
    print(a[0]['source_2'].size())
    
    