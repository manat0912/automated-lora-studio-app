"""Pinned official LTX preprocessing/training adapter with tested Quanto loading."""
import hashlib
import json
import re
import sys
from pathlib import Path
import yaml
from .assets import ROOT

def check(project,h):
    if sys.platform not in ('win32','linux') or h.get('selected_gpu') not in ('amd','nvidia'):
        raise ValueError('LTX training needs a verified Windows/Linux HIP or CUDA runtime.')
    if h.get('vram_gb',0)<31:
        raise ValueError('Current LTX 22B low-memory training requires a nominal 32GiB GPU.')
    variant = project['plan'].get('variant',{})
    from .local_ltx import FILENAME
    quanto = variant.get('local') or (variant.get('repo')=='DeepBeepMeep/LTX-2' and variant.get('file')==FILENAME)
    eligible = (variant.get('official') and variant.get('quantization')=='BF16/FP16') or (quanto and variant.get('quantization')=='INT8 Quanto')
    if variant.get('family') not in ('ltx2','ltx23','ltx25') or not eligible or variant.get('role')!='checkpoint':
        raise ValueError('This training path accepts official BF16/FP16 or the mapped LTX-2.3 v1.1 INT8 Quanto bundle. GGUF, FP8, NVFP4, ConvRot and legacy files require their own validated training loader.')
    if not quanto and h.get('ram_gb',0)<63:
        raise ValueError('Loading the full LTX BF16 checkpoint requires a nominal 64GiB of system RAM in this trainer. Select the LTX-2.3 v1.1 INT8 Quanto bundle for this PC.')
    marker = ROOT/'backends/ltx-official/backend-ready.json'
    if not marker.exists():
        raise ValueError('Install and validate the isolated LTX backend from the Pinokio menu first.')
    return json.loads(marker.read_text())

def configuration(project,files,precomputed,output,h):
    p = project['plan']
    low_memory = h.get('vram_gb',0)<80
    nvidia = h.get('selected_gpu')=='nvidia'
    audio = all(c.get('media',{}).get('audio') for c in project['clips'])
    model = {'model_path':files['local_file'],'text_encoder_path':files['text_encoder_path'],'training_mode':'lora','load_checkpoint':None}
    for field in ('video_vae_path','audio_vae_path'):
        if files.get(field):
            model[field] = files[field]
    strategy = {'name':'flexible','video':{'is_generated':True,'latents_dir':'latents','conditions':[{'type':'first_frame','probability':0.5}]}}
    if audio:
        strategy['audio'] = {'is_generated':True,'latents_dir':'audio_latents'}
    return {'model':model,'lora':{'rank':p['rank'],'alpha':p['rank'],'dropout':0.0,'target_modules':['to_k','to_q','to_v','to_out.0']},'training_strategy':strategy,
        'optimization':{'learning_rate':p['learning_rate'],'steps':p['steps'],'batch_size':1,'gradient_accumulation_steps':1,'max_grad_norm':1.0,'optimizer_type':'adamw8bit' if low_memory and nvidia else 'adamw','scheduler_type':'linear','scheduler_params':{},'enable_gradient_checkpointing':True},
        'acceleration':{'mixed_precision_mode':'bf16','quantization':'int8-quanto' if low_memory else None,'load_text_encoder_in_8bit':low_memory and nvidia},
        'data':{'preprocessed_data_root':str(precomputed),'num_dataloader_workers':0 if sys.platform=='win32' else 2},
        'validation':{'samples':[],'video_dims':[p['width'],p['height'],p['frames']],'frame_rate':p['fps'],'seed':p['seed'],'inference_steps':30,'interval':None,'skip_initial_validation':True},
        'checkpoints':{'interval':min(p['checkpoint_every'],p['steps']),'keep_last_n':-1,'precision':'bfloat16'},
        'flow_matching':{'timestep_sampling_mode':'shifted_logit_normal','timestep_sampling_params':{}},'hub':{'push_to_hub':False},'wandb':{'enabled':False},'seed':p['seed'],'output_dir':str(output)}

def launch(project,run,log=print):
    from .engine import hardware,jobdir
    from .ltx_assets import download_asset
    h = hardware()
    backend = check(project,h)
    p = project['plan']
    files = download_asset(p['variant']['id'],True,log)
    if not files.get('text_encoder_path'):
        raise ValueError('The selected checkpoint has no mapped official text encoder. Do not substitute a different Gemma generation.')
    directory = jobdir(project['id'])
    # Fingerprint prevents old embeddings/latents being reused after edits.
    digest = hashlib.sha256()
    digest.update(json.dumps({'plan':p,'files':files},sort_keys=True).encode())
    for c in project['clips']:
        digest.update(c['caption'].encode())
        with Path(c['path']).open('rb') as source:
            for block in iter(lambda:source.read(8*1024**2),b''):
                digest.update(block)
    identity = digest.hexdigest()
    lock = directory/'ltx-training-fingerprint.json'
    checkpoints = sorted(directory.joinpath('checkpoints').glob('*.safetensors'),key=lambda f:f.stat().st_mtime)
    if checkpoints and (not lock.exists() or json.loads(lock.read_text()).get('fingerprint')!=identity):
        raise ValueError('Dataset/model/settings changed since these LTX checkpoints. Create a new project for the changed training run; old checkpoints are preserved.')
    lock.write_text(json.dumps({'fingerprint':identity},indent=2))
    precomputed = directory/'ltx-cache'/identity
    precomputed.mkdir(parents=True,exist_ok=True)
    metadata = directory/'ltx-dataset.json'
    metadata.write_text(json.dumps([{'caption':c['caption'],'video':c['path']} for c in project['clips']],indent=2))
    config = configuration(project,files,precomputed,directory,h)
    if files.get('prequantized'):
        config['acceleration']['quantization'] = None
        config['acceleration']['load_text_encoder_in_8bit'] = False
        config['optimization']['optimizer_type'] = 'adamw'
    if checkpoints:
        config['model']['load_checkpoint'] = str(checkpoints[-1])
    config_path = directory/'ltx-train.yaml'
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    if checkpoints:
        last = re.fullmatch(r'lora_weights_step_(\d+)\.safetensors',checkpoints[-1].name)
        if last and int(last.group(1))>=p['steps']:
            from safetensors import safe_open
            with safe_open(checkpoints[-1],framework='pt',device='cpu') as weights:
                keys = list(weights.keys())
                if not keys or not all('lora_' in key for key in keys):
                    raise ValueError('The completed LTX checkpoint contains invalid adapter tensors.')
            log('Requested training steps are already saved in a verified checkpoint. Reusing it for preview/export.')
            return bool(config['training_strategy'].get('audio'))
    trainer = Path(backend['trainer_scripts'])
    preprocess = [backend['python'],str(trainer/'process_dataset.py'),str(metadata),'--resolution-buckets',f'{p["width"]}x{p["height"]}x{p["frames"]}','--model-path',files['local_file'],'--text-encoder-path',files['text_encoder_path'],'--output-dir',str(precomputed),'--vae-tiling','--device','cuda:0']
    for field in ('video_vae_path','audio_vae_path'):
        if files.get(field):
            preprocess += ['--'+field.replace('_','-'),files[field]]
    if not config['training_strategy'].get('audio'):
        preprocess += ['--skip-audio']
        log('At least one clip has no audio. This LTX run trains video motion only; audio is not marked trained.')
    if config['acceleration']['load_text_encoder_in_8bit']:
        preprocess += ['--load-text-encoder-in-8bit']
    run(preprocess)
    run([backend['python'],str(trainer/'train.py'),str(config_path),'--disable-progress-bars'])
    return bool(config['training_strategy'].get('audio'))
