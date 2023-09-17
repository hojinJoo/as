import sys
sys.path.append("/workspace/as")
from nussl import nussl
import matplotlib.pyplot as plt
import time
import warnings
import numpy as np
print(nussl)

warnings.filterwarnings("ignore")
start_time = time.time()


audio_path = nussl.efz_utils.download_audio_file(
    'schoolboy_fascination_excerpt.wav')
audio_signal = nussl.AudioSignal(audio_path)
print(audio_signal.audio_data.shape)
separators = [
    nussl.separation.primitive.FT2D(audio_signal),
    nussl.separation.primitive.HPSS(audio_signal),
]

weights = [2, 1]
returns = [[1], [1]]



ensemble = nussl.separation.composite.EnsembleClustering(
    audio_signal, 2, separators=separators, 
    fit_clusterer=True, weights=weights, returns=returns,extracted_feature='estimates')

estimates,res = ensemble.run()

# estimates = {
#     f'Cluster {i}': e for i, e in enumerate(estimates)
# }
print(res[:,:,:,0])
# print(ensemble.clu)
# print(f"Cluster 0: {estimates['Cluster 0']}")