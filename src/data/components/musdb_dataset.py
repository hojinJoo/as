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
# from src.utils.transforms import stft
import musdb

def stft(wav,fs,window_length,nfft,hop_length) :
    from scipy import signal
    window = signal.get_window('hann',window_length)
    _,_,ret = signal.stft(wav,fs=fs,nperseg=window_length,noverlap=window_length-hop_length,window=window,nfft=nfft)
    return torch.from_numpy(ret)
  

class MusDB(Dataset):
    def __init__(
        self,
        metadata: dict,
        data_path : str =  "/media/ssd1/users/hj/musdb18hq",
        sources : [str] = ["vocals", "drums", "bass", "other"],
        segment : int = 3,
        sample_rate : int = 44100,
        normalize : bool = False,
        mode : str = "train",
        n_fft : int = 4096,
        win_length : int = 1024,
        hop_length : int = 1024,
        cac : bool = False
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
        self.mode = 'train' if mode == 'train' or mode =="val"  else 'test'
        if self.mode == "train" :
            for name, meta in self.metadata.items():
                track_duration = meta['length'] / meta['samplerate']
                if segment is None or track_duration < segment:
                    examples = 1
                else:
                    examples = int(math.ceil((track_duration - self.segment)/self.shift) + 1)
                self.num_examples.append(examples)
        else :
            self.test_tracks = musdb.DB(root=self.data_path, subsets='test',is_wav=True)

        self.num_files = sum(self.num_examples)
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.cac = cac
    def get_file(self,name,source):        
        return os.path.join(self.data_path, self.mode ,name, source + ".wav")
    
    def __getitem__(self, index):
        
        if self.mode == "test" :
            ret = {}
            name = list(self.test_tracks)[index]
            ret['name'] = str(name)
            for source in self.sources:
                file = self.get_file(str(name), source)
                wav, _ = torchaudio.load(str(file))
                original_length = wav.size(-1)
                ret['original_length'] = original_length
                wav = stft(wav,fs=self.samplerate,window_length=self.win_length,nfft=self.n_fft,hop_length=self.hop_length)
                wav = wav.squeeze(0)
                ret[source] = wav
            return ret
        
        else :
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
                    
                    if self.segment:
                        length = int(self.segment * self.samplerate)
                        wav = wav[..., :length]
                        wav = F.pad(wav, (0, length - wav.shape[-1]))
                        
                    wav = stft(wav,fs=self.samplerate,window_length=self.win_length,nfft=self.n_fft,hop_length=self.hop_length)
                    
                    wav = wav.squeeze(0)
                
                    # print(wav.size())
                    wavs[source] = wav
                    # wavs[]
                
                # example = torch.stack(wavs)
                # example = example.squeeze(1) # 4, F, T, 2
                
                return wavs
        
        

    def __len__(self):
        if self.mode == "test" :
            return len(self.test_tracks)
        return self.num_files

if __name__ =="__main__" :
    import musdb
    musdb_metadata = "/media/ssd1/users/hj/musdb18hq/musdb_wav.json"
    
    with open(musdb_metadata, 'r') as f :
        metadata = json.load(f)
    metadata_train = {name : meta for name, meta in metadata.items()}
    a = MusDB(metadata_train,cac=True,mode="train",segment=1)
    sample = a[0]
    print(sample['vocals'].shape)
    # print(a[0].keys())
    # sources = ['vocals', 'drums', 'bass', 'other']
    # tmp = torch.stack([sample[source] for source in sources if source != 'vocals'], dim=1)
    # print(f'size : {tmp.size()}')
    # print(torch.sum(mix == v + d + b + o))
    # print(2049 * 212)
    # print(torch.stack([a[0][source] for source in sources], dim=0).size())
    