"""Real PCM decoding for the Windows LTX port, using SoundFile/FFmpeg.

TorchCodec 0.8/0.9 has no Windows Python wheel compatible with the official
AMD torch 2.9 runtime. This decoder replaces audio I/O only; models and losses
remain the official implementations.
"""
import subprocess
import numpy as np
import soundfile as sf
import torch

def load(path):
    try:
        data,rate = sf.read(path,dtype='float32',always_2d=True)
    except (sf.LibsndfileError,RuntimeError):
        # Lossless float PCM from FFmpeg; stereo 48kHz is subsequently resampled
        # by the official audio preprocessor to the model's required rate.
        result = subprocess.run(['ffmpeg','-v','error','-i',str(path),'-vn','-f','f32le','-ac','2','-ar','48000','pipe:1'],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        data = np.frombuffer(result.stdout,dtype='<f4').reshape(-1,2).copy()
        rate = 48000
    if not len(data):
        raise ValueError('Decoded audio is empty.')
    return torch.from_numpy(data.T.copy()),rate
