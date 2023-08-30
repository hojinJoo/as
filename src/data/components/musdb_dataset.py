import json
import os
import pandas as pd
from collections import defaultdict

from typing import Any, Dict, Optional, Tuple
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset
import musdb
from torch.nn import functional as F
import math
import os

class STFT(nn.Module) :
    def __init__(self,
                n_fft : int = 4096,
                n_hop : int = 1024,
                center : bool = False, 
                window : Optional[nn.Parameter] = None) :
        super(STFT, self).__init__()
        if window is None :
            self.window = nn.Parameter(torch.hann_window(n_fft),requires_grad=False)
        else :
            self.window = window
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center
        
    def forward(self, x) :
        _,length = x.size()
        stft = torch.stft(x, n_fft=self.n_fft, hop_length=self.n_hop, window=self.window,normalized=False,onesided=True,center=self.center,return_complex=True,pad_mode='reflect')
        stft = torch.view_as_real(stft)
        stft = torch.abs(torch.view_as_complex(stft))
        return stft
    
    
        

class MusDB(Dataset):
    def __init__(
        self,
        metadata: dict,
        data_path : str =  "/media/ssd1/users/hj/musdb18hq",
        sources : [str] = ["vocals", "drums", "bass", "other"],
        segment : int = 5,
        sample_rate : int = 44100,
        normalize : bool = True,
        mode : str = "train"
    ):
        super().__init__()
        
        self.data_path = data_path
        self.metadata = metadata
        self.segment = segment
        self.sources = sources
        self.shift = segment
        self.normalize = normalize
        self.num_examples = []
        self.samplerate = sample_rate
        self.mode = 'train' if mode == 'train' or 'val' else 'test'
        for name, meta in self.metadata.items():
            track_duration = meta['length'] / meta['samplerate']
            if segment is None or track_duration < segment:
                examples = 1
            else:
                examples = int(math.ceil((track_duration - self.segment)/self.shift) + 1)
            self.num_examples.append(examples)
        
        self.num_files = len(self.num_examples)
        self.stft = STFT()
    def get_file(self,name,source):
        return os.path.join(self.data_path, self.mode ,name, source + ".wav")
    
    def __getitem__(self, index):
        
        ret = {}
        for name, examples in zip(self.metadata, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            meta = self.metadata[name]
            num_frames = -1
            offset = 0
            if self.segment is not None:
                offset = int(meta['samplerate'] * self.shift * index)
                num_frames = int(math.ceil(meta['samplerate'] * self.segment))
            wavs = {}
            for source in self.sources:
                file = self.get_file(name, source)
                wav, _ = torchaudio.load(str(file), frame_offset=offset, num_frames=num_frames)
                wav = wav.mean(dim=-2, keepdim=True)
                if self.normalize:
                    wav = (wav - meta['mean']) / meta['std']
                if self.segment:
                    length = int(self.segment * self.samplerate)
                    wav = wav[..., :length]
                    wav = F.pad(wav, (0, length - wav.shape[-1]))
                    
                wav = self.stft(wav)
                wav = wav.squeeze(0).squeeze(0)
                # print(wav.size())
                wavs[source] = wav
                # wavs[]
            
            # example = torch.stack(wavs)
            # example = example.squeeze(1) # 4, F, T, 2
            
            return wavs
        
        

    def __len__(self):
        return self.num_files

if __name__ =="__main__" :
    import musdb
    musdb_metadata = "/media/ssd1/users/hj/musdb18hq/musdb_wav.json"
    with open(musdb_metadata, 'r') as f :
        metadata = json.load(f)
    metadata_train = {name : meta for name, meta in metadata.items()}
    a = MusDB(metadata_train)
    # print(a[0].keys())
    sources = ['vocals', 'drums', 'bass', 'other']
    mix = torch.sum(torch.stack([a[0][source] for source in sources], dim=0), dim=0)
    v = a[0]['vocals']
    d = a[0]['drums']
    b = a[0]['bass']
    o = a[0]['other']
    print(torch.sum(mix == v + d + b + o))
    print(2049 * 212)
    print(torch.stack([a[0][source] for source in sources], dim=0).size())