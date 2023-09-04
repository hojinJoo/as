import torchaudio
import torch
import os 


def write_wav(wav,path,name):
    os.makedirs(os.path.join(path,name), exist_ok=True)
    s1 = wav[:,0]
    s2 = wav[:,1]
    
    torchaudio.save(os.path.join(path,name,"s1.wav"),s1,sample_rate=16000)
    torchaudio.save(os.path.join(path,name,"s2.wav"),s2,sample_rate=16000)
    