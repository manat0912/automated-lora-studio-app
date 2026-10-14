"""Validate actual downloaded Wan 5B weights with tiny synthetic training and I2V.

This establishes loader/gradient/checkpoint/inference compatibility, not motion quality.
"""
import gc
import os
import json
import shutil
import subprocess
import time
from pathlib import Path
from studio.engine import ROOT,hardware,plan
from studio.catalog import MODELS
from studio import media

directory = ROOT/'validation'/'wan5b'
directory.mkdir(parents=True,exist_ok=True)
report_path = ROOT/'validation'/'full-model-report.json'
report_path.parent.mkdir(exist_ok=True)
report_path.write_text(json.dumps({'status':'waiting_for_download','synthetic':True,'model':'wan22-ti2v-5b'},indent=2))
manifest = ROOT/'models/downloads.json'
repo = MODELS['wan22-ti2v-5b'].repo
start = time.time()
while True:
    try:
        downloads = json.loads(manifest.read_text())
        if downloads.get(repo,{}).get('complete'):
            break
    except (OSError,json.JSONDecodeError):
        pass
    if time.time()-start>8*3600:
        raise TimeoutError('Wan download has not completed within 8 hours. Resume the Pinokio install; no training was attempted.')
    print('Waiting for verified Wan 5B model files; full-model smoke has not started.',flush=True)
    time.sleep(30)

try:
    import torch
    from PIL import Image
    from studio.train_native import train
    from studio.infer import infer
    device_index = hardware().get('device_index',0)
    os.environ['STUDIO_DEVICE_INDEX'] = str(device_index)
    torch.cuda.set_device(device_index)
    source = directory/'synthetic-source.mp4'
    subprocess.run([shutil.which('ffmpeg'),'-v','error','-y','-f','lavfi','-i','testsrc2=size=64x64:rate=24','-t','1.2','-c:v','libx264',str(source)],check=True)
    dataset = directory/'dataset'
    dataset.mkdir(exist_ok=True)
    configuration = plan('wan22-ti2v-5b',source,'synthetic_motion_test','motion','moving colored test pattern; no human characters')
    configuration.update({'width':64,'height':64,'frames':9,'fps':24,'clip_seconds':9/24,'rank':2,'steps':2,'checkpoint_every':1,'preview_steps':2,'semantic_audio':False})
    reference = directory/'synthetic-reference.png'
    Image.new('RGB',(64,64),(70,90,110)).save(reference)
    configuration['reference'] = str(reference)
    clips = []
    for i in range(2):
        target = dataset/f'synthetic_{i}.mp4'
        meta = media.split(source,target,i*9/24,9,24,64,64)
        caption = 'synthetic_motion_test, moving colored test pattern, stationary camera'
        target.with_suffix('.txt').write_text(caption)
        clips.append({'path':str(target),'media':meta,'caption':caption})
    project_path = directory/'project.json'
    project = {'id':'synthetic_validation','synthetic':True,'plan':configuration,'clips':clips,'checkpoints':[]}
    project_path.write_text(json.dumps(project,indent=2))
    report_path.write_text(json.dumps({'status':'training','synthetic':True,'model':repo,'test_geometry':'64x64, 9 frames, batch 1, rank 2, 2 steps'},indent=2))
    train(project_path)
    project['checkpoints'] = [str(p) for p in sorted(directory.joinpath('checkpoints').glob('*.safetensors'))]
    if not project['checkpoints']:
        raise RuntimeError('No full-model test LoRA saved.')
    project_path.write_text(json.dumps(project,indent=2))
    gc.collect()
    torch.cuda.empty_cache()
    report_path.write_text(json.dumps({'status':'inference','synthetic':True,'model':repo,'training_steps_passed':2},indent=2))
    infer(project_path)
    video = media.probe(directory/'preview.mp4',True)
    if video['frames']!=9:
        raise RuntimeError('Full-model I2V preview frame count is wrong.')
    report = {'status':'passed','synthetic':True,'model':repo,'test_geometry':'64x64, 9 frames, rank 2','training_steps':2,'inference_steps':2,'device':hardware().get('runtime_devices'),'preview':str(directory/'preview.mp4'),'quality_validated':False,'limitations':'Real footage, intended training resolution and motion/character transfer quality have not been tested.'}
    report_path.write_text(json.dumps(report,indent=2))
    print('FULL_WAN5B_TRAIN_AND_I2V_SMOKE_PASSED',flush=True)
except Exception as error:
    report_path.write_text(json.dumps({'status':'failed','synthetic':True,'model':repo,'error':str(error)},indent=2))
    raise
