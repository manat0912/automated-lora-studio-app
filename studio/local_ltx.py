"""Discover/reuse local LTX model bundles without changing another app."""
import base64
import hashlib
import json
import os
import shutil
from pathlib import Path
from .assets import ROOT
from ltx_quanto_loader import metadata

FILENAME = 'ltx-2.3-22b-distilled-1.1_diffusion_model_quanto_bf16_int8.safetensors'

def discover():
    roots = []  # External bundles are opt-in via local-model-roots.json.
    settings = ROOT/'local-model-roots.json'
    if settings.exists():
        roots += [Path(p) for p in json.loads(settings.read_text())]
    result = []
    for root in roots:
        path = root/FILENAME
        if not path.is_file():
            continue
        try:
            if metadata(path).get('quantization_format')!='quanto':
                continue
        except (OSError,ValueError):
            continue
        size = path.stat().st_size
        result.append({'id':'local-'+hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16],'family':'ltx23','repo':'Local/Quanto-LTX-2.3-v1.1','file':path.name,'name':path.name,'local_file':str(path.resolve()),'size_bytes':size,'size_gib':round(size/2**30,2),'role':'checkpoint','format':'safetensors','quantization':'INT8 Quanto','distilled':True,'official':False,'local':True,'generation_steps':8,'guidance_scale':1.0,'backend_requirement':'Studio LTX Quanto source port; exact component bundle required','training_status':'Local checkpoint; see actual validation report before claiming full training support','revision':'local file; content fingerprint recorded per training run'})
    return result

def prepare(selected):
    source = Path(selected['local_file'])
    base = source.parent
    gemma = base/'gemma-3-12b-it-qat-q4_0-unquantized'
    weights = gemma/(gemma.name+'_quanto_bf16_int8.safetensors')
    components = {name:base/('ltx-2.3-22b_'+suffix+'.safetensors') for name,suffix in [('video_vae_path','vae'),('audio_vae_path','audio_vae'),('vocoder','vocoder'),('connector','embeddings_connector'),('projection','text_embedding_projection')]}
    required = [source,weights,*components.values(),gemma/'tokenizer.json',gemma/'tokenizer_config.json',gemma/'processor_config.json']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ValueError('Local LTX bundle is incomplete: '+', '.join(missing))
    target = ROOT/'models/local-ltx'/selected['id']
    target.mkdir(parents=True,exist_ok=True)
    for path in gemma.iterdir():
        if path.is_file() and path.suffix in ('.json','.model','.jinja'):
            shutil.copy2(path,target/path.name)
    config = json.loads(base64.b64decode(metadata(weights)['config_base64']))
    (target/'config.json').write_text(json.dumps(config,indent=2))
    # The asset loader expects a weight file in its root. A hard link avoids
    # duplicating 12GiB; the source and link are opened read-only by this backend.
    link = target/'model.safetensors'
    if not link.exists():
        try:
            os.link(weights,link)
        except OSError:
            shutil.copy2(weights,link)
    bundle = {'gemma_weights':str(link),'embedding_weights':[str(components['connector']),str(components['projection'])],'original_files':[str(p) for p in required],'source_modified':False}
    (target/'studio-local-ltx.json').write_text(json.dumps(bundle,indent=2))
    audio_pack = target/'audio-vae-and-vocoder.safetensors'
    if not audio_pack.exists():
        from safetensors.torch import load_file,save_file
        state = load_file(components['audio_vae_path'])
        vocoder = load_file(components['vocoder'])
        if state.keys() & vocoder.keys():
            raise ValueError('Audio VAE and vocoder have overlapping tensor keys; inspect this local bundle.')
        state.update(vocoder)
        temp = audio_pack.with_suffix('.tmp')
        save_file(state,temp,metadata=metadata(components['audio_vae_path']))
        temp.replace(audio_pack)
    return {**selected,'text_encoder_path':str(target),'video_vae_path':str(components['video_vae_path']),'audio_vae_path':str(audio_pack),'vocoder_path':str(components['vocoder']),'embedding_weights':bundle['embedding_weights'],'downloaded_files':[],'components':bundle,'prequantized':True,'status':'Reusing existing local weights; full pipeline validation is recorded separately'}
