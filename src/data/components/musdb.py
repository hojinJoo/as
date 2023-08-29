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
        metadata_path: str = "/media/ssd1/users/hj/musdb18hq/musdb_wav.json",
        tracks : str,
        mode : str = 'train',
        duration : int = 5,
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
        
        return sample

    def __len__(self):
        return self.num_files

if __name__ =="__main__" :
    a = Libri2Mix("/workspace/data/Libri2Mix/wav16k/max/metadata/mixture_train-360_mix_clean.csv")
    print(a[0]['mixture'].size())
    print(a[0]['source_1'].size())
    print(a[0]['source_2'].size())
    
    