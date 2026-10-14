"""Versioned LTX catalog from official Hub metadata and known GGUF publishers.

Cataloging/downloading a checkpoint is distinct from validating its training loader.
"""
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path,PurePosixPath
from .assets import ROOT,download,model_path

OFFICIAL = ['Lightricks/LTX-2','Lightricks/LTX-2.3','Lightricks/LTX-2.3-fp8','Lightricks/LTX-2.5','Lightricks/LTX-Video']
MIRRORS = ['DeepBeepMeep/LTX-2']
PUBLISHERS = ['unsloth','QuantStack']
CATALOG = ROOT/'ltx-catalog.json'
SEED = ROOT/'ltx-catalog.seed.json'

def get_json(url):
    with urllib.request.urlopen(url,timeout=45) as response:
        return json.load(response)

def family_for(repo,file):
    text = (repo+'/'+file).lower()
    if 'ltx-2.5' in text or 'ltx2.5' in text:
        return 'ltx25'
    if 'ltx-2.3' in text or 'ltx2.3' in text:
        return 'ltx23'
    if re.search(r'ltx[-_]?2(?:\b|[-_])',text):
        return 'ltx2'
    return 'ltxv'

def classify(repo,file,size,revision):
    name = Path(file).name
    low = file.lower()
    if 'upscal' in low:
        role = 'upscaler'
    elif 'lora' in low:
        role = 'distillation LoRA' if 'distill' in low else 'LoRA'
    elif 'text_encoder' in low or 'gemma' in low:
        role = 'text encoder'
    elif 'vae' in low:
        role = 'VAE'
    elif 'connector' in low or 'vocoder' in low or 'duration-head' in low or 'embedding_projection' in low:
        role = 'component'
    else:
        role = 'checkpoint'
    match = re.search(r'(Q\d(?:_K(?:_[SML])?|_[01]|_[SML])?|IQ\d(?:_[A-Z0-9]+)?)',name,re.I)
    quant = ('NVFP4' if 'nvfp4' in low else 'INT8 Quanto' if 'quanto' in low and 'int8' in low else 'INT8 ConvRot' if 'convrot' in low and 'int8' in low else 'INT8' if 'int8' in low else 'FP4' if 'fp4' in low else 'FP8' if 'fp8' in low else match.group(0).upper() if match else 'BF16/FP16')
    family = family_for(repo,file)
    return {'id':hashlib.sha256((repo+'/'+file).encode()).hexdigest()[:16],'family':family,'repo':repo,'file':file,'name':name,'revision':revision,'size_bytes':size,'size_gib':round(size/2**30,2),'role':role,'quantization':quant,'format':'GGUF' if low.endswith('.gguf') else 'safetensors','distilled':'distill' in low,'official':repo.startswith('Lightricks/'),'generation_steps':8 if 'distill' in low and family!='ltxv' else None,'guidance_scale':1.0 if 'distill' in low and family!='ltxv' else None,'training_status':'Requires a matching LTX trainer; not validated by this native Wan trainer','backend_requirement':'NVIDIA NVFP4 loader or validated conversion' if 'nvfp4' in low else 'LTX-aware GGUF custom nodes' if low.endswith('.gguf') else 'Matching official LTX / ComfyUI loader'}

def refresh(log=print):
    repos = set(OFFICIAL+MIRRORS)
    for author in PUBLISHERS:
        listing = get_json('https://huggingface.co/api/models?'+urllib.parse.urlencode({'author':author,'search':'LTX','limit':100}))
        repos.update(row['id'] for row in listing if 'gguf' in row['id'].lower())
    assets = []
    errors = []
    for repo in sorted(repos):
        try:
            # expand=siblings lacks sizes; the tree endpoint supplies LFS lengths.
            info = get_json('https://huggingface.co/api/models/'+repo)
            revision = info['sha']
            tree_url = f'https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true&limit=1000'
            # Models selected here have fewer than 1000 files. Never call a partial
            # page "all files" if a repository grows beyond that limit.
            with urllib.request.urlopen(tree_url,timeout=45) as response:
                tree = json.load(response)
                if response.headers.get('Link') and 'rel="next"' in response.headers['Link']:
                    errors.append(repo+': file list exceeded one page; add pagination before claiming completeness')
            for row in tree:
                file = row['path']
                if row['type']!='file' or not file.lower().endswith(('.safetensors','.gguf')):
                    continue
                if repo in MIRRORS and not any(word in file.lower() for word in ('ltx','gemma')):
                    continue
                if '..' in PurePosixPath(file).parts or file.startswith('/'):
                    continue
                # Unified checkpoints / GGUF and split components are included;
                # sharded Diffusers weight fragments are not standalone variants.
                if re.search(r'(model|diffusion_pytorch_model)-\d+-of-\d+',file):
                    continue
                if repo.startswith('Lightricks/') and '/' in file and family_for(repo,file)!='ltx25':
                    continue
                size = row.get('lfs',{}).get('size') or row.get('size') or 0
                assets.append(classify(repo,file,size,revision))
            log(f'LTX catalog inspected: {repo}')
        except Exception as e:
            errors.append(f'{repo}: {e}')
    assets.sort(key=lambda a:(a['family'],not a['official'],a['role'],a['name']))
    result = {'assets':assets,'repositories':sorted(repos),'errors':errors,'scope':'All discovered published standalone checkpoint files in the listed official and GGUF repositories, not every community model on the internet'}
    CATALOG.write_text(json.dumps(result,indent=2))
    return result

def catalog():
    path = CATALOG if CATALOG.exists() else SEED
    result = json.loads(path.read_text()) if path.exists() else {'assets':[],'errors':['Refresh the LTX catalog to discover real files.']}
    from .local_ltx import discover
    result['assets'] += discover()
    return result

def asset(id):
    match = next((a for a in catalog()['assets'] if a['id']==id),None)
    if not match:
        raise ValueError('Unknown LTX asset; refresh or select an existing catalog entry.')
    return match

def default_asset(family):
    options = [a for a in catalog()['assets'] if a['family']==family and a['role']=='checkpoint']
    h_path = ROOT/'hardware.json'
    h = json.loads(h_path.read_text()) if h_path.exists() else {}
    if h.get('selected_gpu')=='amd' and h.get('platform')=='win32':
        local = next((a for a in options if a.get('local')),None)
        if local:
            return local
    if h.get('vram_gb',0)<80 and family=='ltx23':
        from .local_ltx import FILENAME
        portable = next((a for a in options if a['repo']=='DeepBeepMeep/LTX-2' and a['file']==FILENAME),None)
        if portable:
            return portable
    preferred = next((a for a in options if a['official'] and 'distilled-1.1' in a['name'] and a['quantization']=='BF16/FP16'),None)
    return preferred or next((a for a in options if a['official'] and a['quantization']=='BF16/FP16'),options[0] if options else None)

def download_requests(selected,include_components=True):
    """Single download plan shared by the interactive installer and asset loader."""
    if selected.get('local'):
        return []
    patterns = [selected['file']]
    from .local_ltx import FILENAME
    if include_components and selected['repo']=='DeepBeepMeep/LTX-2' and selected['file']==FILENAME:
        gemma = 'gemma-3-12b-it-qat-q4_0-unquantized/'
        patterns += ['ltx-2.3-22b_'+name+'.safetensors' for name in ['vae','audio_vae','vocoder','embeddings_connector','text_embedding_projection']]
        patterns += [gemma+'*.json',gemma+'*.model',gemma+'*.jinja',gemma+'gemma-3-12b-it-qat-q4_0-unquantized_quanto_bf16_int8.safetensors']
    if include_components and selected['family']=='ltx25' and selected['role']=='checkpoint':
        patterns += ['vae/ltx-2.5-audio-vae-bf16.safetensors','vae/ltx-2.5-video-vae-bf16.safetensors']
        te = 'text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors' if selected['quantization']=='INT8 ConvRot' else 'text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors'
        patterns.append(te)
    requests = [{'repo':selected['repo'],'patterns':patterns,'revision':selected['revision']}]
    portable = selected['repo']=='DeepBeepMeep/LTX-2' and selected['file']==FILENAME
    if include_components and selected['family'] in ('ltx2','ltx23') and not portable and selected['role']=='checkpoint':
        requests.append({'repo':'Lightricks/gemma-3-12b-it-qat-q4_0-unquantized','patterns':['*.json','*.safetensors','*.model','*.txt','*.jinja'],'revision':None})
    return requests

def download_asset(id,include_components=True,log=print):
    selected = asset(id)
    if selected.get('local'):
        from .local_ltx import prepare
        result = prepare(selected)
        log('Reusing local LTX/Gemma weights and components; original application files are unchanged.')
        return result
    from .engine import hardware
    h = hardware()
    if selected['quantization']=='NVFP4' and h.get('selected_gpu')!='nvidia':
        raise ValueError('No native NVFP4 loader is validated on this AMD runtime. Select FP8, GGUF or BF16, or use an explicitly validated conversion in your runner.')
    requests = download_requests(selected,include_components)
    patterns = requests[0]['patterns']
    location = download(selected['repo'],patterns,log,revision=selected['revision'])
    from .local_ltx import FILENAME
    if include_components and selected['repo']=='DeepBeepMeep/LTX-2' and selected['file']==FILENAME:
        from .local_ltx import prepare
        result = prepare({**selected,'local_file':str(location/selected['file'])})
        result['status'] = 'Matching Quanto weights and components downloaded, integrity checked and prepared for the isolated LTX trainer.'
        return result
    requirements = []
    backend_paths = {}
    if include_components and selected['family']=='ltx25' and selected['role']=='checkpoint':
        te = patterns[-1]
        backend_paths = {'text_encoder_path':str(location/te),'video_vae_path':str(location/'vae/ltx-2.5-video-vae-bf16.safetensors'),'audio_vae_path':str(location/'vae/ltx-2.5-audio-vae-bf16.safetensors')}
    if include_components and selected['family'] in ('ltx2','ltx23') and selected['role']=='checkpoint':
        encoder = requests[1]
        te_repo = encoder['repo']
        te_path = download(te_repo,encoder['patterns'],log)
        requirements.append({'role':'matching Gemma 3 text encoder','repo':te_repo,'path':str(te_path)})
        backend_paths['text_encoder_path'] = str(te_path)
    if selected['family']=='ltxv':
        requirements.append({'role':'matching T5 text encoder','status':'Select the encoder supplied with your validated legacy LTX workflow; it is not silently substituted with Gemma.'})
    handoff = {**selected,**backend_paths,'local_file':str(location/selected['file']),'downloaded_files':[str(location/p) for p in patterns],'components':requirements,'status':'Downloaded and file integrity checked; loader/training validation is separate'}
    path = ROOT/'models'/'ltx-installed.json'
    records = json.loads(path.read_text()) if path.exists() else {}
    records[id] = handoff
    path.write_text(json.dumps(records,indent=2))
    return handoff

def choices(family=None,checkpoints_only=False):
    return [(f'{a["name"]} · {a["quantization"]} · {a["size_gib"]} GiB',a['id']) for a in catalog()['assets'] if (not family or a['family']==family) and (not checkpoints_only or a['role']=='checkpoint')]

if __name__=='__main__':
    result = refresh()
    print(f'LTX assets discovered: {len(result["assets"])}; metadata errors: {len(result["errors"])}',flush=True)
