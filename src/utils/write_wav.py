import torchaudio
import torch
import os 


def write_wav(wav,path,name,sample_rate=16000) :
    os.makedirs(os.path.join(path,name), exist_ok=True)
    s1 = wav[:,0].squeeze(0)
    s2 = wav[:,1].squeeze(0)

    torchaudio.save(os.path.join(path,name,"s1.wav"),s1,sample_rate=sample_rate)
    torchaudio.save(os.path.join(path,name,"s2.wav"),s2,sample_rate=sample_rate)

