"""Install the pinned official LTX trainer with vendor-specific wheels."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = '9ec55f9f22798a3198d9c923856824821bc3317e'

def install():
    h = json.loads((ROOT/'hardware.json').read_text())
    if sys.platform not in ('win32','linux') or h.get('selected_gpu') not in ('nvidia','amd') or not h.get('gpu_verified'):
        raise RuntimeError('LTX requires a verified Windows/Linux AMD HIP or NVIDIA CUDA runtime. No LTX packages were installed.')
    if h.get('vram_gb',0) < 31:
        raise RuntimeError('The current LTX 22B low-memory training preset requires a nominal 32GiB GPU. No LTX packages were installed.')
    base = ROOT/'backends'
    base.mkdir(exist_ok=True)
    repo = base/'ltx-official'
    if not repo.exists():
        env = {**os.environ,'GIT_LFS_SKIP_SMUDGE':'1'}
        subprocess.run(['git','clone','https://github.com/Lightricks/LTX-2.git',str(repo)],check=True,env=env)
        subprocess.run(['git','-C',str(repo),'checkout',REVISION],check=True)
    actual = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if actual != REVISION:
        raise RuntimeError('LTX revision differs from this adapter. Local changes are preserved; no automatic reset is performed.')
    (repo/'backend-ready.json').unlink(missing_ok=True)
    runtime = base/'ltx-env'
    python = runtime/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.exists():
        subprocess.run(['uv','venv','--python','3.12',str(runtime)],check=True)
    # Reuse the audited vendor installer, which preserves HIP on AMD Windows.
    subprocess.run([str(python),str(ROOT/'runtime_install.py')],cwd=ROOT,check=True)
    constraints = base/'ltx-runtime.txt'
    pins = subprocess.check_output([str(python),'-c','from importlib.metadata import version; print("\\n".join(n+"=="+version(n) for n in ["torch","torchvision","torchaudio"]))'],text=True)
    # 5.8 meets the official Gemma4 floor and retains distributed-capability
    # guards that newer releases dropped. AMD Windows has no FSDP extension.
    constraints.write_text(pins+'\ntransformers==5.8.0\npeft==0.18.1\naccelerate==1.10.1\nav==16.0.1\nnumpy==2.2.6\n'+('' if os.name=='nt' else 'torchcodec==0.8.1\n'))
    from ltx_port import patch
    package_root = patch(repo,REVISION)
    packages = [str(package_root/name) for name in ['ltx-core','ltx-pipelines','ltx-trainer']]
    subprocess.run(['uv','pip','install','--python',str(python),*packages,'mmgp==3.7.12','-c',str(constraints)],check=True)
    subprocess.run(['uv','pip','check','--python',str(python)],check=True)
    trainer = package_root/'ltx-trainer'
    for script in ['train.py','process_dataset.py']:
        subprocess.run([str(python),str(trainer/'scripts'/script),'--help'],cwd=trainer,check=True)
    # Real video/audio native extension imports, not merely package metadata.
    check = 'import torch, transformers; from ltx_trainer.config import LtxTrainerConfig; from ltx_trainer.trainer import LtxvTrainer; assert torch.cuda.is_available(); x=torch.randn(64,64,device="cuda",requires_grad=True); (x@x).mean().backward(); torch.cuda.synchronize()'
    subprocess.run([str(python),'-c',check],cwd=trainer,check=True)
    subprocess.run([str(python),str(ROOT/'ltx_runtime_smoke.py')],cwd=ROOT,check=True)
    (repo/'backend-ready.json').write_text(json.dumps({'revision':REVISION,'python':str(python),'repo':str(repo),'trainer_scripts':str(trainer/'scripts'),'platform':sys.platform,'gpu':h['selected_gpu'],'windows_audio_port':os.name=='nt','validated':'dependency resolution, CLI imports, real PCM decoding, quantized LoRA GPU backward; full pretrained LTX model test is recorded separately'},indent=2))
    print('OFFICIAL_LTX_BACKEND_DEPENDENCY_CHECKS_PASSED',flush=True)

if __name__ == '__main__':
    install()
