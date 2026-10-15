import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from .catalog import MODELS, model_dict
from . import media
from .assets import ensure_base

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT/'jobs'
GPU_LOCK = threading.Lock()
ACTIVE = {}

def hardware():
    path = ROOT/'hardware.json'
    return json.loads(path.read_text()) if path.exists() else {'gpu_verified':False,'selected_gpu':'cpu','warnings':['Run Pinokio Scan PC & install first.']}

def jobdir(job_id):
    if not re.fullmatch(r'[a-f0-9]{12}',str(job_id)):
        raise ValueError('Invalid project ID.')
    return JOBS/job_id

def load(job_id):
    return json.loads(jobdir(job_id).joinpath('project.json').read_text())

def save(project):
    directory = jobdir(project['id'])
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/'project.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(project,indent=2),encoding='utf-8')
    temp.replace(path)

def plan(key, source, trigger, purpose='motion', instructions=''):
    if key not in MODELS:
        raise ValueError('Unknown model preset.')
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{1,47}',trigger):
        raise ValueError('Trigger must start with a letter and contain 2–48 letters, digits, underscores or hyphens.')
    m = MODELS[key]
    info = media.probe(source)
    h = hardware()
    vram = h.get('vram_gb',0)
    # Explicit heuristics, not promises about exact VRAM consumption.
    longest = 512 if vram >= 31 else 384
    low_memory_ltx = m.arch in ('ltx2','ltx23','ltx25') and vram<48
    if low_memory_ltx:
        longest = 320
    width, height = media.dimensions(info['width'],info['height'],longest,m.spatial)
    requested_seconds = 5.5 if vram >= 31 else 3.5
    if low_memory_ltx:
        requested_seconds = 2.1
    frames = media.frame_count(min(requested_seconds,info['duration']),m.fps,m.stride)
    seconds = frames/m.fps
    if info['duration'] < seconds:
        raise ValueError(f'Source is too short: need at least {seconds:.2f}s for the selected frame rule.')
    return {'model':model_dict(key),'source':str(Path(source).resolve()),'source_info':info,'trigger':trigger,'purpose':purpose,'instructions':instructions,'width':width,'height':height,'frames':frames,'fps':m.fps,'clip_seconds':seconds,'rank':8 if low_memory_ltx else 16,'steps':2000,'learning_rate':5e-5,'batch_size':1,'checkpoint_every':100 if low_memory_ltx else 500,'seed':42,'hardware':h,'memory_profile':'LTX 32GiB: 320px longest side, 49 frames, rank8' if low_memory_ltx else 'Standard','warnings':[
        'Hyperparameters are conservative heuristics; a real dataset run determines memory and convergence.',
        'A motion LoRA can retain source appearance. Diverse performers and I2V reference tests reduce that risk.',
        'LoRA weights require the same base model family and variant; no universal cross-model adapter is possible.',
    ] + ([] if m.audio else ['This model learns video motion. Audio is recorded for tags and timing but is not trained into the visual LoRA.'])}

def create(source,key,trigger,purpose,instructions,reference=None,semantic_audio=True):
    p = plan(key,source,trigger,purpose,instructions)
    id = uuid.uuid4().hex[:12]
    directory = jobdir(id)
    directory.joinpath('input').mkdir(parents=True)
    src = directory/'input'/('source'+Path(source).suffix.lower())
    shutil.copy2(source,src)
    p['source'] = str(src.resolve())
    p['semantic_audio'] = semantic_audio
    p['reference'] = None
    if reference:
        from PIL import Image
        target = directory/'input'/'reference.png'
        Image.open(reference).convert('RGB').save(target)
        p['reference'] = str(target.resolve())
    project = {'id':id,'name':trigger,'created':time.time(),'status':'new','plan':p,'clips':[],'checkpoints':[],'error':None}
    save(project)
    return project

def append_log(id,line):
    directory = jobdir(id)
    with directory.joinpath('events.log').open('a',encoding='utf-8') as f:
        f.write(f'{time.strftime("%H:%M:%S")}  {line}\n')

def retry_smaller(id):
    if id in ACTIVE and ACTIVE[id]['thread'].is_alive():
        raise ValueError('Cancel the original run and wait for it to stop before retrying.')
    original = load(id)
    old = original['plan']
    if old['model']['arch'] not in ('ltx2','ltx23','ltx25'):
        raise ValueError('The smaller-memory retry currently applies to LTX training.')
    p = create(old['source'],old['model']['key'],old['trigger'],old['purpose'],old['instructions'],old.get('reference'),old.get('semantic_audio',True))
    longest,seconds = (256,1.1) if max(old['width'],old['height'])<=320 and old['frames']<=49 else (320,2.1)
    width,height = media.dimensions(old['source_info']['width'],old['source_info']['height'],longest,32)
    frames = media.frame_count(min(seconds,old['source_info']['duration']),24,8)
    p['plan'].update(width=width,height=height,frames=frames,clip_seconds=frames/24,memory_profile=f'Smaller LTX retry: {width}x{height}, {frames} frames')
    p['plan'].update(steps=old['steps'],rank=min(8,old['rank']),checkpoint_every=100)
    if old.get('variant'):
        p['plan']['variant'] = old['variant']
    p['retry_of'] = id
    p['name'] = old['trigger']+' · smaller memory retry'
    p['candidate_starts'] = [c['source_start'] for c in original['clips'] if 'source_start' in c]
    save(p)
    dispatch(p['id'],'prepare',True)
    return {'project_id':p['id'],'retry_of':id,'plan':p['plan']}

def status(id):
    if not id:
        return {},'Create a project to begin.'
    p = load(id)
    log = jobdir(id)/'events.log'
    if p['status']=='training':
        p['checkpoints'] = [str(f.resolve()) for f in sorted(jobdir(id).joinpath('checkpoints').glob('*.safetensors'),key=lambda f:f.stat().st_mtime)]
    return p, log.read_text(encoding='utf-8')[-16000:] if log.exists() else ''

def list_projects(include_synthetic=False):
    JOBS.mkdir(exist_ok=True)
    projects = []
    for path in JOBS.glob('*/project.json'):
        try:
            p = json.loads(path.read_text())
            if p.get('synthetic') and not include_synthetic:
                continue
            projects.append(p)
        except (ValueError,OSError):
            continue
    return sorted(projects,key=lambda p:p['created'],reverse=True)

def check_training(p):
    m = MODELS[p['plan']['model']['key']]
    h = hardware()
    if not h.get('gpu_verified'):
        raise ValueError('GPU forward/backward verification has not passed. Run the Pinokio installer.')
    if m.arch.startswith('ltx'):
        from .ltx_adapter import check
        check(p,h)
        return
    if h.get('vram_gb',0) < m.min_vram:
        raise ValueError(f'{m.label} requires at least {m.min_vram}GiB for this full-precision trainer. Choose a smaller model; this runtime does not fake CUDA-only 4-bit quantization on AMD.')
    if not m.native:
        backend = ROOT/'backends'/'ai-toolkit'/'backend-ready.json'
        if not backend.exists():
            raise ValueError('Install and validate the optional AI Toolkit backend in the Pinokio menu before training this preset.')

def dispatch(id,stage='prepare',auto_train=False):
    if stage not in ('prepare','train','generate'):
        raise ValueError('Unknown operation.')
    if id in ACTIVE and ACTIVE[id]['thread'].is_alive():
        raise ValueError('This project is already running.')
    p = load(id)
    if stage == 'train' or auto_train:
        check_training(p)
    if stage == 'train' and not p['clips']:
        raise ValueError('Prepare a dataset before training.')
    if stage == 'generate' and (not p['checkpoints'] or not p['plan']['reference']):
        raise ValueError('Generation needs a trained checkpoint and a reference image.')
    cancel = threading.Event()
    p['status'] = 'queued'
    p['error'] = None
    save(p)
    def work():
        try:
            with GPU_LOCK:
                if cancel.is_set():
                    raise InterruptedError('Cancelled before start.')
                if stage == 'prepare':
                    prepare(id,cancel)
                    if auto_train:
                        train(id,cancel)
                elif stage == 'train':
                    train(id,cancel)
                else:
                    generate(id,cancel)
        except Exception as e:
            project = load(id)
            project['status'] = 'cancelled' if isinstance(e,InterruptedError) else 'failed'
            project['error'] = str(e)
            save(project)
            append_log(id,traceback.format_exc())
    thread = threading.Thread(target=work,daemon=True)
    ACTIVE[id] = {'thread':thread,'cancel':cancel}
    thread.start()
    return f'Project {id}: {stage} queued. Progress appears below.'

def cancel_job(id):
    if id in ACTIVE:
        ACTIVE[id]['cancel'].set()
        append_log(id,'Cancellation requested; the worker will stop and preserve completed checkpoints.')
    return 'Cancellation requested.'

def prepare(id,cancel):
    subprocess_job([sys.executable,'-m','studio.prepare_worker','--id',id],id,cancel)

def _prepare(id,cancel):
    p = load(id)
    plan = p['plan']
    p['status'] = 'splitting'
    save(p)
    log = lambda message:append_log(id,message)
    directory = jobdir(id)/'dataset'
    # Resume only verified media; no stale captions or wrong model geometry reused.
    directory.mkdir(exist_ok=True)
    cuts = media.boundaries(plan['source'],plan['source_info']['duration'],cancel)
    starts = media.windows(cuts,plan['clip_seconds'])
    if p.get('candidate_starts'):
        starts = [s for s in p['candidate_starts'] if any(a<=s and s+plan['clip_seconds']<=b+.01 for a,b in zip(cuts,cuts[1:]))]
    if not starts:
        m = MODELS[plan['model']['key']]
        longest_scene = max((b-a for a,b in zip(cuts,cuts[1:])),default=0)
        frames = media.frame_count(longest_scene,m.fps,m.stride)
        if frames<m.stride+1:
            raise ValueError('No scene contains enough frames for a motion clip in this architecture.')
        plan['frames'] = frames
        plan['clip_seconds'] = frames/m.fps
        plan['warnings'].append('Clip length was shortened automatically to fit the detected scene cuts.')
        starts = media.windows(cuts,plan['clip_seconds'])
        save(p)
        log(f'Adjusted scene-aware clip duration to {plan["clip_seconds"]:.2f}s, {frames} frames.')
    if len(starts)>250:
        raise ValueError('Source would create more than 250 clips. Divide it into smaller projects.')
    h = hardware()
    device = f'cuda:{h.get("device_index",0)}' if h.get('gpu_verified') else 'cpu'
    from .selector import select
    p['status'] = 'selecting_motion'
    save(p)
    decisions = select(plan['source'],starts,plan['clip_seconds'],plan['instructions'],device,cancel,log)
    p['selection'] = decisions
    starts = [d['start'] for d in decisions if d['keep']]
    if not starts:
        save(p)
        raise ValueError('AI selection found no usable motion clips. Check your footage and instructions. Decisions are saved in the project.')
    p['status'] = 'splitting'
    save(p)
    clips = []
    for index,start in enumerate(starts):
        if cancel.is_set():
            raise InterruptedError('Cancelled during clipping.')
        target = directory/f'clip_{index:04d}.mp4'
        meta = media.split(plan['source'],target,start,plan['frames'],plan['fps'],plan['width'],plan['height'],cancel)
        clips.append({'index':index,'path':str(target.resolve()),'source_start':start,'media':meta})
        log(f'Clip {index+1}/{len(starts)} verified: {meta["frames"]} frames · {meta["width"]}×{meta["height"]} · {meta["fps"]}fps')
    p['clips'] = clips
    p['status'] = 'captioning'
    save(p)
    from .caption import caption_clips
    clips = caption_clips(clips,plan['trigger'],plan['purpose'],plan['instructions'],plan['semantic_audio'],device,cancel,log)
    # Audio-video backends receive natural language audio and speech in captions.
    if plan['model']['audio']:
        from .caption import compose_caption
        for clip in clips:
            clip['caption'] = compose_caption(clip,plan['trigger'],True)
            Path(clip['path']).with_suffix('.txt').write_text(clip['caption'],encoding='utf-8')
    p['clips'] = clips
    p['status'] = 'dataset_ready'
    save(p)
    write_handoff(p)
    log(f'Dataset ready: {len(clips)} verified .mp4/.txt pairs. AI captions may need corrections.')

def write_handoff(p):
    plan = p['plan']
    instructions = {'base_model':plan['model']['repo'],'model_key':plan['model']['key'],'trigger':plan['trigger'],'reference_image':plan['reference'],'width':plan['width'],'height':plan['height'],'fps':plan['fps'],'frames':plan['frames'],'recommended_strength':0.7,'prompt':f'{plan["trigger"]}, {plan["instructions"] or "perform the learned movement"}, maintain the appearance and number of the characters in the reference image','audio_trained':plan['model']['audio'],'limitations':['Reference-image identity preservation requires visual evaluation.','Use a compatible model variant. LoRAs cannot be merged across unrelated architectures.','A motion LoRA does not contain explicit multi-character identity assignments.']}
    instructions['audio_trained'] = bool(p.get('audio_training_completed',False) and p.get('checkpoints'))
    instructions['audio_in_dataset'] = plan['model']['audio']
    if plan.get('variant'):
        instructions['checkpoint_variant'] = plan['variant']
        instructions['generation_defaults'] = {'steps':plan['variant'].get('generation_steps'),'guidance_scale':plan['variant'].get('guidance_scale')}
    jobdir(p['id']).joinpath('inference.json').write_text(json.dumps(instructions,indent=2))

def subprocess_job(args,id,cancel):
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    h = hardware()
    env['STUDIO_DEVICE_INDEX'] = str(h.get('device_index',0))
    backend_python = Path(args[0]).resolve()
    if backend_python.is_relative_to((ROOT/'backends/ltx-env').resolve()):
        from .runtime_env import ltx_environment
        env, removed = ltx_environment(env,h.get('selected_gpu'),sys.platform)
        if removed:
            append_log(id,'Native Windows ROCm defaults: cleared legacy process overrides '+', '.join(removed))
        # Official Accelerate training uses logical cuda:0 in a single-GPU run.
        # Expose only the scanned GPU so an integrated/smaller adapter cannot
        # be selected instead on multi-GPU hosts.
        env['CUDA_VISIBLE_DEVICES'] = str(h.get('device_index',0))
        if h.get('selected_gpu')=='amd':
            env['HIP_VISIBLE_DEVICES'] = str(h.get('device_index',0))
        env['STUDIO_DEVICE_INDEX'] = '0'
    with subprocess.Popen(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env) as process:
        # Drain output independently so cancellation remains responsive during kernels.
        def drain():
            for line in process.stdout:
                append_log(id,line.rstrip())
        reader = threading.Thread(target=drain,daemon=True)
        reader.start()
        while process.poll() is None:
            if cancel.wait(.5):
                import psutil
                try:
                    descendants = psutil.Process(process.pid).children(recursive=True)
                    for child in descendants:
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    _,live = psutil.wait_procs(descendants,timeout=3)
                    for child in live:
                        child.kill()
                except psutil.NoSuchProcess:
                    pass
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                reader.join(timeout=5)
                raise InterruptedError('Cancelled. Last completed checkpoints are preserved.')
        reader.join(timeout=5)
        if process.returncode:
            raise RuntimeError(f'Trainer exited with status {process.returncode}. Read the log for the actual dependency, memory, or dataset error.')

def train(id,cancel):
    p = load(id)
    check_training(p)
    if len(p['clips']) < 2:
        raise ValueError('Use at least two clips to train. A larger, varied motion dataset is strongly preferred.')
    p['status'] = 'training'
    save(p)
    audio_trained = False
    if p['plan']['model']['arch'].startswith('ltx'):
        from .ltx_adapter import launch
        audio_trained = launch(p,lambda args:subprocess_job(args,id,cancel),lambda message:append_log(id,message))
    elif p['plan']['model']['native']:
        ensure_base(p['plan']['model']['key'],lambda message:append_log(id,message))
        subprocess_job([sys.executable,'-m','studio.train_native','--project',str(jobdir(id)/'project.json')],id,cancel)
    else:
        from .toolkit_adapter import launch
        launch(p,lambda args:subprocess_job(args,id,cancel))
    p = load(id)
    checkpoints = sorted(jobdir(id).joinpath('checkpoints').glob('*.safetensors'),key=lambda path:path.stat().st_mtime)
    if not checkpoints:
        raise RuntimeError('Trainer produced no safetensors checkpoint. Training is not marked complete.')
    from safetensors import safe_open
    for path in checkpoints:
        with safe_open(path,framework='pt',device='cpu') as f:
            if not list(f.keys()):
                raise RuntimeError(f'Empty checkpoint: {path}')
    p['checkpoints'] = [str(path.resolve()) for path in checkpoints]
    p['audio_training_completed'] = audio_trained
    p['status'] = 'trained'
    save(p)
    write_handoff(p)
    append_log(id,'Training finished and checkpoint tensors verified. Generate reference-image samples to assess motion and identity preservation.')

def generate(id,cancel):
    p = load(id)
    is_ltx = p['plan']['model']['arch'] in ('ltx2','ltx23','ltx25')
    if not p['plan']['model']['native'] and not is_ltx:
        raise ValueError('Use the exported inference instructions in a compatible LTX/Wan runner for this backend.')
    p['status'] = 'generating'
    save(p)
    if is_ltx:
        from .ltx_adapter import check
        backend = check(p,hardware())
        subprocess_job([backend['python'],str(ROOT/'ltx_infer.py'),'--project',str(jobdir(id)/'project.json')],id,cancel)
    else:
        subprocess_job([sys.executable,'-m','studio.infer','--project',str(jobdir(id)/'project.json')],id,cancel)
    p = load(id)
    target = jobdir(id)/'preview.mp4'
    if not target.is_file():
        raise RuntimeError('Inference did not produce a video.')
    p['preview'] = str(target)
    p['status'] = 'trained'
    save(p)

def edit_caption(id,index,text):
    if id in ACTIVE and ACTIVE[id]['thread'].is_alive():
        raise ValueError('Wait for this project to finish before editing captions.')
    p = load(id)
    if not text or not text.strip():
        raise ValueError('Caption cannot be empty.')
    c = p['clips'][int(index)]
    c['caption'] = text.strip()
    c['caption_status'] = 'Reviewed by user'
    Path(c['path']).with_suffix('.txt').write_text(c['caption'],encoding='utf-8')
    Path(c['path']).with_suffix('.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
    save(p)
    # Caption changes invalidate cached text embeddings.
    return 'Caption saved. Text cache keys include the caption content.'

def export(id):
    p = load(id)
    if id in ACTIVE and ACTIVE[id]['thread'].is_alive():
        raise ValueError('Wait for the project to finish before exporting.')
    import zipfile
    directory = jobdir(id)
    target = directory/'export.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_STORED) as z:
        for name in ['dataset','checkpoints']:
            for path in directory.joinpath(name).glob('*'):
                if path.is_file():
                    z.write(path,str(path.relative_to(directory)))
        for name in ['project.json','inference.json','preview.mp4']:
            path = directory/name
            if path.exists():
                z.write(path,name)
    return str(target)
