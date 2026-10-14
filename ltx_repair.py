"""Refresh the pinned copied trainer port without replacing vendor GPU packages."""
import json
import subprocess
import sys
from pathlib import Path
from ltx_install import REVISION
from ltx_port import patch

ROOT = Path(__file__).resolve().parent
repo = ROOT/'backends/ltx-official'
marker = repo/'backend-ready.json'
backend = json.loads(marker.read_text())
actual = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
if actual!=REVISION or backend['revision']!=REVISION:
    raise RuntimeError('Pinned LTX revision changed; refusing an unreviewed port update.')
marker.unlink()
port = patch(repo,REVISION)
subprocess.run(['uv','pip','install','--python',backend['python'],'--reinstall-package','ltx-trainer',str(port/'ltx-trainer'),'-c',str(ROOT/'backends/ltx-runtime.txt')],check=True)
subprocess.run(['uv','pip','check','--python',backend['python']],check=True)
subprocess.run([backend['python'],str(ROOT/'ltx_runtime_smoke.py')],check=True)
backend['trainer_scripts'] = str(port/'ltx-trainer/scripts')
marker.write_text(json.dumps(backend,indent=2),encoding='utf-8')
print('LTX_PROGRESS_AND_RECOVERY_PORT_UPDATED',flush=True)
