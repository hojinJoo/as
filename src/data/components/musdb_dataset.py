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
# from src.utils.transforms import torch_stft as stft
import musdb

def stft(wav,nfft,hop_length) :
    window = torch.hann_window(nfft)
    return torch.stft(wav,n_fft=nfft,hop_length=hop_length,return_complex=True,window=window,onesided=True,normalized=True)  

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
        cac : bool = False,
        test : bool = False
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
                wav = stft(wav,nfft=self.n_fft,hop_length=self.hop_length)
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
                        
                    wav = stft(wav,nfft=self.n_fft,hop_length=self.hop_length)
                    wav = torch.pow(wav,0.3)
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
       
    def istft(wav,fs,window_length,nfft,hop_length,original_length) :
        from scipy import signal
        window = signal.get_window('hann',window_length)
        _,ret = signal.istft(wav,fs=fs,nperseg=window_length,noverlap=window_length-hop_length,window=window,nfft=nfft)
        return torch.from_numpy(ret[:,:original_length])

    def torch_stft(wav,window_length,nfft,hop_length) :
        window = torch.hann_window(nfft)
        return torch.stft(wav,n_fft=nfft,hop_length=hop_length,return_complex=True,window=window,onesided=True,normalized=True)

    def torch_isfft(wav,window_length,nfft,hop_length,original_length) :
        window = torch.hann_window(nfft)
        return torch.istft(wav,n_fft=nfft,hop_length=hop_length,window=window,length=original_length,onesided=True,return_complex=False,normalized=True)
   
    def ibm(sources) :
        sources = torch.abs(sources)
        ibm = (sources == torch.max(sources, dim=0, keepdim=True).values).float()
        # ibm = ibm / torch.sum(ibm, dim=0, keepdim=True)
        # ibm[ ibm <= 0.5] = 0
        return ibm
    
    
    def _wiener(model_out ,mix_spec, n_iters) :
        from openunmix.filtering import wiener
        
        model_out = model_out.unsqueeze(0)
        mix_spec = mix_spec.unsqueeze(0)
        init = mix_spec.dtype
        win_len = 300
        wiener_residual = True
        B,S,C,Fr,T = model_out.size()
        model_out = model_out.permute(0,4,3,2,1) # B,T,Fr,C,S
        mix_spec = torch.view_as_real(mix_spec.permute(0,3,2,1)) # B,T,Fr,2 (complex)
        
        outs = []
        for sample in range(B) :
            pos = 0
            out = []
            for pos in range(0,T,win_len) :
                frame = slice(pos,pos+win_len)
                z_out = wiener(model_out[sample,frame],mix_spec[sample,frame],n_iters,residual=wiener_residual)
                
                out.append(z_out.transpose(-1,-2))
            outs.append(torch.cat(out,dim=0))

        out = torch.view_as_complex(torch.stack(outs,dim=0))
        out = out.permute(0,4,3,2,1).contiguous()
        if wiener_residual :
            out = out[:,:-1]
        return out.to(init)
        # pass
    def new_sdr(references, estimates):
        """ 
        Compute the SDR according to the MDX challenge definition.
        Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
        """
        references = references[None]
        estimates = estimates[None]
        assert references.dim() == 4
        assert estimates.dim() == 4
        delta = 1e-7  # avoid numerical errors
        num = torch.sum(torch.square(references), dim=(2, 3))
        den = torch.sum(torch.square(references - estimates), dim=(2, 3))
        num += delta
        den += delta
        scores = 10 * torch.log10(num / den)
        return scores
    
    
    
    
    import musdb
    musdb_metadata = "/media/ssd1/users/hj/musdb18hq/musdb_wav.json"
    
    with open(musdb_metadata, 'r') as f :
        metadata = json.load(f)
    metadata_train = {name : meta for name, meta in metadata.items()}
    a = MusDB(metadata_train,cac=True,mode="test",segment=1)
    sample = a[40]
    print(f"vocal size : {sample['vocals'].size()}")
    print(f"name : {sample['name']}")

    vocal_spec = sample['vocals']
    accom_spec = torch.sum(torch.stack([sample[source] for source in ['drums', 'bass', 'other']], dim=0), dim=0)
    print(f"vocal spec negative value : {torch.sum(vocal_spec < 0)}")

    cat = torch.stack([vocal_spec,accom_spec],dim=0)
    mix = vocal_spec + accom_spec
    # print(f"cat0 vocal {torch.sum(cat[0] == vocal_spec)}")
    # print(f"cat1 vocal {torch.sum(cat[1] == accom_spec)}")
    mask = ibm(cat)
    # mask = ibm(torch.view_as_real(cat))
    m = _wiener(torch.abs(cat),mix,n_iters=0)
    
    pred = mix * mask
    
    
    # print(f"vocal pred size : {vocal_pred.size()}")
    # print(f"vocal_pred precision : {torch.sum(vocal_pred == vocal_spec)}")
    # print(f"accompaniment precision : {torch.sum(accom_pred == accom_spec)}")
    
    print(f"cat size : {cat.size()}")
    print(f"cat dim : {cat.dim()}")
    original_vocal = torch_isfft(cat[0],window_length=1024,nfft=4096,hop_length=1024,original_length=sample['original_length'])
    original_accom = torch_isfft(cat[1],window_length=1024,nfft=4096,hop_length=1024,original_length=sample['original_length'])
    pred_vocal = torch_isfft(pred[0],window_length=1024,nfft=4096,hop_length=1024,original_length=sample['original_length'])
    pred_accom = torch_isfft(pred[1],window_length=1024,nfft=4096,hop_length=1024,original_length=sample['original_length'])
    
    m_vocal = torch_isfft(m[0,0],window_length=1024,nfft=4096,hop_length=1024,original_length=sample['original_length'])
    m_accom = torch_isfft(m[0,1],window_length=1024,nfft=4096,hop_length=1024,original_length=sample['original_length'])
    
    ms = torch.stack([m_vocal,m_accom],dim=0)
    ms = ms.transpose(1,2).double()
    
    originals = torch.stack([original_vocal,original_accom],dim=0)
    preds = torch.stack([pred_vocal,pred_accom],dim=0)
    
    # real_wav_vocal = sample['vocals_wav']
    # real_wav_accom = torch.sum(torch.stack([sample[source + '_wav'] for source in ['drums', 'bass', 'other']], dim=0),dim=0)
    # real_wavs = torch.stack([real_wav_vocal,real_wav_accom],dim=0)
    # real_wavs = real_wavs.transpose(1,2).double()
    
        
    originals = originals.transpose(1,2).double()
    preds = preds.transpose(1,2).double()
    
    score = new_sdr(originals,preds)
    score2 = new_sdr(originals,ms)
    print(f"score : {score[0][0]}")
    print(f"score2 : {score2}")
    print(f"with real wav : {new_sdr(real_wavs,originals)}")
    print(f"with real wav : {new_sdr(real_wavs,preds)}")
    print(f"with real wav : {new_sdr(real_wavs,ms)}")
    
    from torchmetrics.audio import SignalDistortionRatio
    torch_sdr = SignalDistortionRatio()
    print(f"torch sdr1 : {torch_sdr(originals[0],preds[0])}")
    print(f"torch sdr2 : {torch_sdr(originals[0],ms[0])}")
    
    
    import nussl
    
    nussl_mix = nussl.AudioSignal(audio_data_array=(real_wav_vocal.numpy() + real_wav_accom.numpy()),sample_rate=44100)
    nussl_sources = [nussl.AudioSignal(audio_data_array=real_wav_vocal.numpy(),sample_rate=44100),nussl.AudioSignal(audio_data_array=real_wav_accom.numpy(),sample_rate=44100)]
    nussl_wiener = nussl.separation.benchmark.IdealBinaryMask(nussl_mix,nussl_sources)
    estimates = nussl_wiener()
    nussl_vocal = torch.from_numpy(estimates[0].audio_data)
    nussl_accom = torch.from_numpy(estimates[1].audio_data)
    
    nussl_all = torch.stack([nussl_vocal,nussl_accom],dim=0)
    print(f"nussl all size : {nussl_all.size()}")
    nussl_all = nussl_all.transpose(1,2).double()
    print(f"sdr : {new_sdr(real_wavs,nussl_all)}")
    # print(f"nussl sdr : {nussl.evaluation.bss_eval.bss_eval_sources(nussl_sources,estimates)}")
    
    
    # torchaudio.save("vocal_original.wav",original_vocal,44100)
    # torchaudio.save("accompaniment_original.wav",original_accom,44100)
    # torchaudio.save("vocal_ibm.wav",pred_vocal,44100)
    # torchaudio.save("accompaniment_ibm.wav",pred_accom,44100)
    # torchaudio.save("vocal_wiener.wav",m_vocal,44100)
    # torchaudio.save("accompaniment_wiener.wav",m_accom,44100)
    
    
    # print(f"vocal spec size : {vocal_spec.size()}")
    # print(f"vocal dtype : {vocal_spec.dtype}")
    # print(f"vocal size : {torch.view_as_real(vocal_spec).size()}")
    # vocal_s2w = torch_isfft(vocal_spec,window_length=1024,nfft=4096,hop_length=1024,original_length=original_length)
    # accom_s2w = torch_isfft(accom_spec,window_length=1024,nfft=4096,hop_length=1024,original_length=original_length)
    # # print(f"sample name : {sample['name']}")
    # print(f"vocal s2w siz : {vocal_s2w.size()}")
    # print(torch.sum(vocal_wav == vocal_s2w))
    # print(f"accom : {torch.sum(accompaniment_wav == accom_s2w)}")
    
    # acoompaniment_wav = istft(accompaniment,fs=44100,window_length=1024,nfft=4096,hop_length=1024,original_length=sample['original_length'])
    
    # torchaudio.save("vocal_th.wav",vocal_s2w,44100)
    # torchaudio.save("accompaniment_th.wav",accom_s2w,44100)
    
    
    # print(a[0].keys())
    # sources = ['vocals', 'drums', 'bass', 'other']
    # tmp = torch.stack([sample[source] for source in sources if source != 'vocals'], dim=1)
    # print(f'size : {tmp.size()}')
    # print(torch.sum(mix == v + d + b + o))
    # print(2049 * 212)
    # print(torch.stack([a[0][source] for source in sources], dim=0).size())
    