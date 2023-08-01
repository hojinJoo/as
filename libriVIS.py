import json
import os
import pandas as pd
from collections import defaultdict

import torch
import torchaudio
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np


out_path = "/workspace/vis"
os.makedirs(out_path,exist_ok=True)
metadata_path = "/media/Libri2Mix//mixture_train-360_mix_clean_metadata.csv"
meta_data = pd.read_csv(metadata_path,encoding="ms932",sep=",")

for index in range(len(meta_data)) :
    mixtureID,mixture_path,source_1_path,source_2_path,length = meta_data.values[index]
    mixture_wav,sample_rate = torchaudio.load(mixture_path)
    source_1,sample_rate = torchaudio.load(source_1_path)
    source_2,sample_rate = torchaudio.load(source_2_path)
    mixture = torch.abs(torch.stft(mixture_wav, n_fft=512, win_length=512,
                                hop_length=125, return_complex=True)).squeeze(0)
    source_1 = torch.abs(torch.stft(source_1, n_fft=512, win_length=512,
                            hop_length=125, return_complex=True)).squeeze(0)
    source_2 = torch.abs(torch.stft(source_2, n_fft=512, win_length=512,
                            hop_length=125, return_complex=True)).squeeze(0)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    
    # axes[0].imshow((20 * np.log10(mixture.numpy() + 1e-8)), cmap="viridis", vmin=-160, vmax=150, origin="lower", aspect="auto")
    im = axes[0].imshow((20 * np.log10(mixture.numpy() + 1e-8)), origin="lower", aspect="auto")
    
    axes[0].set_title(f'mix')
    axes[0].axis('off')
    
    axes[1].imshow((20 * np.log10(source_1.numpy() + 1e-8)), origin="lower", aspect="auto")
    axes[1].set_title(f'source1')
    axes[1].axis('off')
    
    axes[2].imshow((20 * np.log10(source_2.numpy() + 1e-8)), origin="lower", aspect="auto")
    axes[2].set_title(f'source2')
    axes[2].axis('off')            
            
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.5)
    fig.tight_layout()

    plt.savefig(os.path.join(out_path,f'{index}.png'), dpi=300)