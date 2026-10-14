"""Actual media preprocessing and pretrained LTX training through studio engine."""
import json
import shutil
import subprocess
import threading
from pathlib import Path
from PIL import Image
from studio import engine,media
from studio.ltx_assets import default_asset

ROOT = Path(__file__).resolve().parent
report_path = ROOT/'validation/ltx-pipeline-report.json'
report_path.parent.mkdir(exist_ok=True)
try:
    selected = default_asset('ltx23')
    if not selected or not selected.get('local'):
        raise ValueError('The Windows validation run requires the discovered local LTX-2.3 v1.1 Quanto bundle.')
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    project = None
    if previous.get('project_id') and previous.get('status')=='i2v_generation':
        candidate = engine.load(previous['project_id'])
        if candidate.get('synthetic') and candidate['plan'].get('variant',{}).get('id')==selected['id']:
            project = candidate
    directory = ROOT/'validation/ltx-media'
    directory.mkdir(parents=True,exist_ok=True)
    source = directory/'synthetic-source.mp4'
    subprocess.run([shutil.which('ffmpeg'),'-v','error','-y','-f','lavfi','-i','testsrc2=size=64x64:rate=24','-f','lavfi','-i','sine=frequency=440:sample_rate=48000','-t','1.2','-c:v','libx264','-c:a','aac',str(source)],check=True)
    reference = directory/'reference.png'
    Image.new('RGB',(64,64),(80,100,110)).save(reference)
    if project is None:
        project = engine.create(source,'ltx23','synthetic_ltx_validation','motion','moving colored test pattern with an audible tone',reference,False)
        project['synthetic'] = True
        project['plan'].update({'variant':selected,'width':64,'height':64,'frames':9,'fps':24,'clip_seconds':9/24,'rank':2,'steps':2,'checkpoint_every':1,'preview_steps':2})
    job = engine.jobdir(project['id'])
    dataset = job/'dataset'
    dataset.mkdir(exist_ok=True)
    if not project['clips']:
        for i in range(2):
            target = dataset/f'synthetic_{i}.mp4'
            info = media.split(source,target,i*9/24,9,24,64,64)
            caption = 'synthetic_ltx_validation, moving colored test pattern, static camera. Audio: a continuous tone.'
            target.with_suffix('.txt').write_text(caption)
            project['clips'].append({'index':i,'path':str(target),'media':info,'caption':caption})
    engine.save(project)
    report_path.write_text(json.dumps({'status':'preprocessing_and_training','project_id':project['id'],'synthetic':True},indent=2))
    engine.dispatch(project['id'],'train')
    engine.ACTIVE[project['id']]['thread'].join()
    result = engine.load(project['id'])
    if result['status']!='trained' or not result['checkpoints'] or not result.get('audio_training_completed'):
        raise RuntimeError(result.get('error') or 'Expected completed real audio/video training and nonempty checkpoints.')
    report_path.write_text(json.dumps({'status':'i2v_generation','project_id':project['id'],'synthetic':True,'training_steps_passed':2},indent=2))
    engine.dispatch(project['id'],'generate')
    engine.ACTIVE[project['id']]['thread'].join()
    result = engine.load(project['id'])
    if result.get('error') or not result.get('preview'):
        raise RuntimeError(result.get('error') or 'LTX I2V preview is missing.')
    info = media.probe(result['preview'],True)
    if info['frames']!=9:
        raise RuntimeError('LTX I2V preview has an incorrect frame count.')
    report_path.write_text(json.dumps({'status':'passed','project_id':project['id'],'synthetic':True,'pretrained_model':'LTX-2.3 v1.1 distilled INT8 Quanto','media_preprocessing_tested':True,'training_steps':2,'geometry':'64x64, 9 frames at24fps, rank2, batch1','audio_and_video_training':True,'checkpoints':result['checkpoints'],'i2v_generation_tested':True,'preview':result['preview'],'quality_validated':False},indent=2))
    print('LTX_REAL_MEDIA_TRAIN_AND_I2V_PASSED',flush=True)
except Exception as e:
    report_path.write_text(json.dumps({'status':'failed','error':str(e),'synthetic':True,'project_id':project['id'] if 'project' in globals() else None},indent=2))
    raise
