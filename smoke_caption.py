"""Real local vision/speech inference on clearly labeled synthetic test footage."""
import json
import argparse
import os
import shutil
import subprocess
import threading
from pathlib import Path
from studio.engine import ROOT,hardware
from studio import media
from studio.selector import select
from studio.caption import caption_clips

parser = argparse.ArgumentParser()
parser.add_argument('--audio',action='store_true')
parser.add_argument('--audio-budget',type=float)
args = parser.parse_args()
if args.audio_budget is not None:
    if args.audio_budget<=0:
        raise ValueError('Audio memory budget must be positive.')
    os.environ['STUDIO_AUDIO_MAX_VRAM_GIB'] = str(args.audio_budget)

directory = ROOT/'jobs'/'validation-media'
directory.mkdir(parents=True,exist_ok=True)
path = directory/'synthetic-test.mp4'
subprocess.run([shutil.which('ffmpeg'),'-v','error','-y','-f','lavfi','-i','testsrc2=size=256x160:rate=16','-f','lavfi','-i','sine=frequency=440:sample_rate=48000','-t','5','-c:v','libx264','-c:a','aac',str(path)],check=True)
h = hardware()
device = f'cuda:{h.get("device_index",0)}'
cancel = threading.Event()
decisions = select(path,[0],4.5,'moving test pattern; do not invent people',device,cancel,print)
records = [{'index':0,'path':str(path),'source_start':0,'media':media.probe(path,True)}]
records = caption_clips(records,'testpattern','motion','Describe the visible test pattern; do not invent performers.',args.audio,device,cancel,print)
assert records[0]['caption'] and records[0]['analysis']
if args.audio:
    assert records[0].get('audio_description','').strip()
(directory/('caption-audio-offload-result.json' if args.audio_budget else 'caption-audio-result.json' if args.audio else 'caption-result.json')).write_text(json.dumps({'synthetic':True,'semantic_audio_tested':args.audio,'audio_budget_gib':args.audio_budget,'selection':decisions,'clips':records},indent=2))
print('REAL_VISION_SPEECH_AND_AUDIO_SMOKE_PASSED' if args.audio else 'REAL_VISION_AND_SPEECH_SMOKE_PASSED',flush=True)
