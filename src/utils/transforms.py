from scipy import signal
import torch

def stft(wav,fs,window_length,nfft,hop_length) :
    window = signal.get_window('hann',window_length)
    _,_,ret = signal.stft(wav,fs=fs,nperseg=window_length,noverlap=window_length-hop_length,window=window,nfft=nfft)
    return torch.from_numpy(ret)
   
   
def istft(wav,fs,window_length,nfft,hop_length,original_length) :
    window = signal.get_window('hann',window_length)
    _,ret = signal.istft(wav,fs=fs,nperseg=window_length,noverlap=window_length-hop_length,window=window,nfft=nfft)
    return torch.from_numpy(ret[:,:original_length])

