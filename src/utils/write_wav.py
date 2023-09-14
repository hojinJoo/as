import torchaudio
import torch
import os 


def write_wav(wav,path,name,sample_rate=16000) :
    os.makedirs(os.path.join(path,name), exist_ok=True)
    s1 = wav[0].cpu()
    s2 = wav[1].cpu()

    torchaudio.save(os.path.join(path,name,"vocal.wav"),s1,sample_rate=sample_rate)
    torchaudio.save(os.path.join(path,name,"accompaniment.wav"),s2,sample_rate=sample_rate)

