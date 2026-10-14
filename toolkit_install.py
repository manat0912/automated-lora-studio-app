"""Optional upstream runtime. CUDA on Windows, CUDA/HIP on Linux only.

Never patches or substitutes unsupported distributed/quantization modules on AMD Windows.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = 'f7a1fb9a77e5418269b1e9eda4ace6d7c1b7bb47'

def install():
    h = json.loads((ROOT/'hardware.json').read_text())
    if not h.get('gpu_verified'):
        raise RuntimeError('Install and verify the native GPU runtime first.')
    if sys.platform == 'win32' and h['selected_gpu']=='amd':
        raise RuntimeError('The upstream AI Toolkit dependency stack is not validated for AMD Windows. Native Wan training is available here; LTX/AI Toolkit training requires a compatible Linux/NVIDIA runtime.')
    base = ROOT/'backends'
    base.mkdir(exist_ok=True)
    repo = base/'ai-toolkit'
    if not repo.exists():
        subprocess.run(['git','clone','https://github.com/ostris/ai-toolkit.git',str(repo)],check=True)
        subprocess.run(['git','-C',str(repo),'checkout',REVISION],check=True)
    actual = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if actual != REVISION:
        raise RuntimeError('AI Toolkit revision differs from the validated configuration adapter. Keep local changes; do not reset automatically.')
    runtime = base/'toolkit-env'
    python = runtime/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.exists():
        subprocess.run(['uv','venv','--python','3.12',str(runtime)],check=True)
    subprocess.run([str(python), str(ROOT/'runtime_install.py')],cwd=ROOT,check=True)
    constraints = base/'toolkit-runtime.txt'
    pins = subprocess.check_output([str(python),'-c','from importlib.metadata import version; print("\\n".join(n+"=="+version(n) for n in ["torch","torchvision","torchaudio"]))'],text=True)
    constraints.write_text(pins)
    subprocess.run(['uv','pip','install','--python',str(python),'-r',str(repo/'requirements.txt'),'-c',str(constraints)],check=True)
    subprocess.run(['uv','pip','check','--python',str(python)],check=True)
    # Exercise the real CLI import path; installation is not marked ready by a venv alone.
    subprocess.run([str(python),'run.py','--help'],cwd=repo,check=True)
    check = 'import torch; from toolkit.config_modules import ModelConfig, DatasetConfig; from toolkit.models.base_model import BaseModel; assert torch.cuda.is_available(); x=torch.randn(64,64,device="cuda",requires_grad=True); (x@x).mean().backward(); torch.cuda.synchronize()'
    subprocess.run([str(python),'-c',check],cwd=repo,check=True)
    repo.joinpath('backend-ready.json').write_text(json.dumps({'revision':REVISION,'python':str(python),'gpu':h['selected_gpu'],'validated':'dependency resolution, CLI imports and GPU backward; no full training run'},indent=2))
    print('AI Toolkit backend dependency checks passed. Actual dataset training still needs validation.',flush=True)

if __name__=='__main__':
    install()
