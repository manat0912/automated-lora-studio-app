import json
import os
import shutil
import subprocess
from pathlib import Path
from .catalog import MODELS, VISION_REPO, SPEECH_REPO, AUDIO_REPO, INTEGRATIONS

ROOT = Path(__file__).resolve().parents[1]
MODELDIR = ROOT / 'models'

def model_path(repo):
    return MODELDIR / repo.replace('/','--')

def download(repo, patterns=None, log=print,revision=None):
    from filelock import FileLock
    MODELDIR.mkdir(exist_ok=True)
    with FileLock(str(MODELDIR/(repo.replace('/','--')+'.lock')),timeout=7200):
        return _download(repo,patterns,log,revision)

def _download(repo, patterns=None, log=print,revision=None):
    from huggingface_hub import HfApi, snapshot_download
    MODELDIR.mkdir(exist_ok=True)
    api = HfApi()
    info = api.model_info(repo, revision=revision,files_metadata=True)
    import fnmatch
    selected = [f for f in info.siblings if (not patterns or any(fnmatch.fnmatch(f.rfilename,p) for p in patterns))]
    if not selected:
        raise ValueError(f'No model files match the requested asset set in {repo}.')
    if patterns:
        for pattern in patterns:
            if not any(c in pattern for c in '*?[') and pattern not in {f.rfilename for f in selected}:
                raise ValueError(f'Required model component does not exist at the pinned revision: {repo}/{pattern}')
    size = sum(f.size or 0 for f in selected)
    dest = model_path(repo)
    lock_path = MODELDIR/'downloads.json'
    locks = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    # A pinned complete download can be reused without re-fetching a newer revision.
    old = locks.get(repo)
    required_names = {f.rfilename for f in selected}
    old_names = {f['name'] for f in old.get('files',[])} if old else set()
    if old and old.get('complete') and required_names.issubset(old_names) and (revision is None or old['revision']==revision) and all((dest/f['name']).is_file() and (dest/f['name']).stat().st_size == f['size'] for f in old['files']):
        log(f'Already verified: {repo} ({old["revision"][:12]})')
        return dest
    existing = sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()) if dest.exists() else 0
    if shutil.disk_usage(MODELDIR).free < max(0,size-existing) + 5*2**30:
        raise RuntimeError(f'Insufficient disk space for {repo} ({size/2**30:.1f}GiB plus 5GiB reserve).')
    log(f'Downloading {repo} at {info.sha}: {size/2**30:.1f}GiB')
    snapshot_download(repo, revision=info.sha, local_dir=dest, allow_patterns=patterns, max_workers=3)
    files = []
    for f in selected:
        path = dest/f.rfilename
        if not path.is_file() or (f.size is not None and path.stat().st_size != f.size):
            raise RuntimeError(f'Missing or truncated model file: {path}')
        checksum = getattr(getattr(f,'lfs',None),'sha256',None)
        if checksum:
            import hashlib
            actual = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):
                    actual.update(chunk)
            if actual.hexdigest() != checksum:
                raise RuntimeError(f'Model SHA256 mismatch: {path}')
        files.append({'name':f.rfilename,'size':path.stat().st_size,'sha256':checksum})
    from filelock import FileLock
    with FileLock(str(MODELDIR/'.manifest.lock'),timeout=60):
        locks = json.loads(lock_path.read_text()) if lock_path.exists() else {}
        previous = locks.get(repo,{})
        all_files = {f['name']:f for f in previous.get('files',[])} if previous.get('revision')==info.sha else {}
        all_files.update({f['name']:f for f in files})
        locks[repo] = {'revision':info.sha,'complete':True,'files':list(all_files.values())}
        temp = lock_path.with_suffix('.tmp')
        temp.write_text(json.dumps(locks,indent=2))
        temp.replace(lock_path)
    return dest

def ensure_base(key, log=print):
    m = MODELS[key]
    if not m.native:
        raise RuntimeError('This preset requires the separate AI Toolkit runtime; use its Pinokio installer.')
    return download(m.repo, ['*.json','*.safetensors','*.txt','*.model','*.tiktoken'],log)

def install_models():
    hardware = json.loads(ROOT.joinpath('hardware.json').read_text())
    if not hardware['download_models']:
        print('Model downloads deferred by explicit installer argument.',flush=True)
        return
    download(VISION_REPO, ['*.json','*.safetensors','*.txt','*.model','*.jinja'])
    download(SPEECH_REPO, ['*.json','*.bin','*.txt'])
    if hardware['audio_semantics']:
        download(AUDIO_REPO, ['*.json','*.safetensors','*.txt','*.model','*.jinja'])
    if hardware['model'].startswith('ltx'):
        from .ltx_assets import default_asset,download_asset
        chosen = hardware.get('ltx_asset') if hardware.get('ltx_asset_requested') else default_asset(hardware['model'])['id']
        chosen = chosen or default_asset(hardware['model'])['id']
        download_asset(chosen,include_components=True)
    else:
        ensure_base(hardware['model'])

def download_sources(force=False):
    hardware = json.loads(ROOT.joinpath('hardware.json').read_text())
    if not hardware.get('source_library') and not force:
        print('Optional source library not selected. Core dependencies are fully installed.',flush=True)
        return
    directory = ROOT/'sources'
    directory.mkdir(exist_ok=True)
    results = []
    for name,repo,role in INTEGRATIONS:
        dest = directory/repo.replace('/','--')
        if not dest.exists():
            env = os.environ.copy()
            env['GIT_LFS_SKIP_SMUDGE'] = '1'
            subprocess.run(['git','clone','--depth','1',f'https://github.com/{repo}.git',str(dest)], check=True,env=env)
        revision = subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
        results.append({'name':name,'repo':repo,'revision':revision,'role':role,'status':'source downloaded; runtime not installed'})
    directory.joinpath('manifest.json').write_text(json.dumps(results,indent=2))
