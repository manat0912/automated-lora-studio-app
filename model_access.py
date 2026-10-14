"""Plan local downloads and poll real Hub authorization; never accept terms."""
import argparse
import json
import time
from pathlib import Path
from studio.assets import download
from studio.catalog import MODELS, VISION_REPO, SPEECH_REPO, AUDIO_REPO

ROOT = Path(__file__).resolve().parent
PLAN = ROOT/'model-download-plan.json'
STATE = ROOT/'model-access.json'

def base_requests(key, variant=None, hardware=None):
    if key not in MODELS:
        raise ValueError('Unknown model preset.')
    if key.startswith('ltx'):
        from studio.ltx_assets import asset,default_asset,download_requests
        chosen = asset(variant) if variant else default_asset(key)
        if not chosen or chosen['family'] != key or chosen['role'] != 'checkpoint':
            raise ValueError('Select a checkpoint from the requested LTX family.')
        if chosen['quantization'] == 'NVFP4' and (hardware or {}).get('selected_gpu') != 'nvidia':
            raise ValueError('NVFP4 requires a validated NVIDIA loader. No model files were downloaded.')
        return download_requests(chosen), chosen['id']
    model = MODELS[key]
    if not model.native:
        raise ValueError('This model needs the optional AI Toolkit backend; it is not a native installer preset.')
    return [{'repo':model.repo,'patterns':['*.json','*.safetensors','*.txt','*.model','*.tiktoken'],'revision':None}], None

def plan(options, hardware):
    component = options.get('component','model')
    if component not in ('install','model','sources'):
        raise ValueError('Unknown download component.')
    requests = []
    variant = None
    if component != 'sources' and (component != 'install' or hardware.get('download_models')):
        if component == 'install':
            requests = [
                {'repo':VISION_REPO,'patterns':['*.json','*.safetensors','*.txt','*.model','*.jinja'],'revision':None},
                {'repo':SPEECH_REPO,'patterns':['*.json','*.bin','*.txt'],'revision':None}]
            if hardware.get('audio_semantics'):
                requests.append({'repo':AUDIO_REPO,'patterns':['*.json','*.safetensors','*.txt','*.model','*.jinja'],'revision':None})
        key = hardware['model'] if component == 'install' else options['model']
        variant_id = (hardware.get('ltx_asset') if hardware.get('ltx_asset_requested') else None) if component == 'install' else options.get('ltx_asset')
        base, variant = base_requests(key,variant_id,hardware)
        requests += base
    return {'component':component,'requests':requests,'ltx_asset':variant}

def probe(request):
    from huggingface_hub import HfApi, get_token
    from huggingface_hub.errors import HfHubHTTPError
    repo = request['repo']
    token = get_token() or False  # Re-read the managed token after browser login.
    api = HfApi(token=token)
    try:
        info = api.model_info(repo,revision=request.get('revision'))
        if info.gated:
            api.auth_check(repo_id=repo,repo_type='model')
        blocked = False
    except HfHubHTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status not in (401,403):
            raise
        blocked = True
    return {'repo':repo,'url':'https://huggingface.co/'+repo,'needs_access':blocked}

def wait_download(request, timeout=1800, interval=5):
    deadline = time.monotonic()+timeout
    announced = False
    while probe(request)['needs_access']:
        if time.monotonic() >= deadline:
            raise RuntimeError('Hugging Face access is still pending for '+request['repo']+'. Obtain approval/read permission, then rerun the installer; partial downloads are preserved.')
        if not announced:
            print('Waiting for Hugging Face approval: '+request['repo']+'. Downloads start automatically when access is granted.',flush=True)
            announced = True
        time.sleep(interval)
    # Starts in the same process immediately after the authorization check passes.
    return download(request['repo'],request['patterns'],revision=request.get('revision'))

def complete(result):
    if result.get('ltx_asset'):
        from studio.ltx_assets import download_asset
        download_asset(result['ltx_asset'],include_components=True)
    if result['component'] == 'sources':
        from studio.assets import download_sources
        download_sources(force=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['plan','probe','wait-download','complete'])
    parser.add_argument('--index',type=int,default=0)
    args = parser.parse_args()
    if args.action == 'plan':
        options = json.loads((ROOT/'model-request.json').read_text())
        hardware = json.loads((ROOT/'hardware.json').read_text())
        result = plan(options,hardware)
        PLAN.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('MODEL_DOWNLOAD_PLAN_READY',flush=True)
        return
    result = json.loads(PLAN.read_text(encoding='utf-8'))
    if args.action == 'complete':
        complete(result)
        print('MODEL_DOWNLOADS_COMPLETE',flush=True)
        return
    if not 0 <= args.index < len(result['requests']):
        raise ValueError('Download index is outside the plan.')
    request = result['requests'][args.index]
    if args.action == 'probe':
        STATE.write_text(json.dumps(probe(request)),encoding='utf-8')
    else:
        wait_download(request)

if __name__ == '__main__':
    main()
