import json
from pathlib import Path
from .engine import ROOT, jobdir
from .assets import model_path, download

def configuration(p):
    plan = p['plan']
    model = plan['model']
    process = {
        'type':'sd_trainer','training_folder':str(jobdir(p['id'])/'checkpoints'), 'device':'cuda:0',
        'network':{'type':'lora','linear':plan['rank'],'linear_alpha':plan['rank']},
        'save':{'dtype':'float16','save_every':plan['checkpoint_every'],'max_step_saves_to_keep':4},
        'datasets':[{'folder_path':str(jobdir(p['id'])/'dataset'),'caption_ext':'txt','num_frames':plan['frames'],'fps':plan['fps'],'shrink_video_to_frames':False,'resolution':[int((plan['width']*plan['height'])**.5)],'do_i2v':True,'do_audio':model['audio'],'cache_latents_to_disk':True,'cache_text_embeddings':True}],
        'train':{'batch_size':1,'steps':plan['steps'],'gradient_accumulation':1,'train_unet':True,'train_text_encoder':False,'gradient_checkpointing':True,'noise_scheduler':'flowmatch','optimizer':'adamw','lr':plan['learning_rate'],'dtype':'bf16','cache_text_embeddings':True,'disable_sampling':True},
        'model':{'name_or_path':model['repo'],'arch':model['arch'],'low_vram':True,'quantize':False,'quantize_te':False},
        'sample':{'sampler':'flowmatch','sample_every':plan['checkpoint_every'],'width':plan['width'],'height':plan['height'],'num_frames':plan['frames'],'fps':plan['fps'],'prompts':[plan['trigger']+', '+plan['instructions']],'seed':plan['seed'],'sample_steps':30}
    }
    if model['arch']=='wan22_14b_i2v':
        process['model']['model_kwargs']={'train_high_noise':True,'train_low_noise':True}
        process['train']['switch_boundary_every']=10
    if model['arch']=='ltx25':
        raise ValueError('LTX-2.5 needs the architecture-specific Gemma 4 text encoder and component map. This adapter is not yet validated for that layout; native Wan presets are executable.')
    return {'job':'extension','config':{'name':plan['trigger'],'process':[process]},'meta':{'name':plan['trigger'],'version':'1.0'}}

def launch(p,runner):
    import yaml
    repo = ROOT/'backends'/'ai-toolkit'
    ready = json.loads(repo.joinpath('backend-ready.json').read_text())
    cfg = configuration(p)
    # Only Diffusers-layout Wan is automatically resolved here. Official LTX unified
    # files need a matching text encoder map; never present a guessed path as ready.
    if p['plan']['model']['arch'].startswith('ltx'):
        raise ValueError('LTX architecture export is available, but automatic checkpoint/text-encoder mapping is not validated in this build. Use a validated LTX ComfyUI workflow via the bridge.')
    base = download(p['plan']['model']['repo'],['*.json','*.safetensors','*.txt','*.model'])
    cfg['config']['process'][0]['model']['name_or_path']=str(base)
    target = jobdir(p['id'])/'toolkit.yaml'
    target.write_text(yaml.safe_dump(cfg,sort_keys=False))
    # run.py expects its working directory on sys.path for extension discovery.
    bridge = ROOT/'toolkit_runner.py'
    runner([ready['python'],str(bridge),str(repo),str(target)])
    for weight in jobdir(p['id']).joinpath('checkpoints').glob('*/*.safetensors'):
        import shutil
        shutil.copy2(weight,jobdir(p['id'])/'checkpoints'/weight.name)
