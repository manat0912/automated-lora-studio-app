"""CPU speech inference isolated from PyTorch's AMD/NVIDIA OpenMP libraries."""
import argparse
import json
from pathlib import Path

def transcribe(model_path,audio_path,output):
    # Do not import torch, cv2, the UI, or the vision model in this process.
    from faster_whisper import WhisperModel
    import soundfile as sf
    import numpy as np
    model = WhisperModel(model_path,device='cpu',compute_type='int8')
    audio,rate = sf.read(audio_path,dtype='float32')
    if rate!=16000 or audio.ndim!=1:
        raise ValueError('Speech worker requires mono 16kHz audio from the FFmpeg preparation stage.')
    segments,info = model.transcribe(np.asarray(audio),vad_filter=True,word_timestamps=True,condition_on_previous_text=False)
    rows = []
    for s in segments:
        if s.no_speech_prob<.6 and s.avg_logprob>-1:
            rows.append({'start':round(s.start,3),'end':round(s.end,3),'text':s.text.strip(),'words':[{'word':w.word,'start':w.start,'end':w.end,'probability':w.probability} for w in s.words or []]})
    Path(output).write_text(json.dumps({'segments':rows,'language':info.language},indent=2),encoding='utf-8')

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--audio',required=True)
    p.add_argument('--output',required=True)
    a = p.parse_args()
    transcribe(a.model,a.audio,a.output)
