"""Pinokio verification dispatch with validated args and no shell interpolation."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--request',default='ltx-verify-request.json')
options = json.loads(Path(parser.parse_args().request).read_text())
python = ROOT/'backends/ltx-env'/('Scripts/python.exe' if __import__('os').name=='nt' else 'bin/python')
if options.get('stage')=='runtime':
    command = [str(python),str(ROOT/'ltx_runtime_smoke.py')]
elif options.get('stage')=='local':
    path = options.get('checkpoint')
    if not path:
        from studio.local_ltx import discover
        found = discover()
        path = found[0]['local_file'] if found else ''
    if not Path(path).is_file():
        raise ValueError('A local LTX Quanto checkpoint was not found. Supply its exact path with --checkpoint.')
    command = [str(python),str(ROOT/'ltx_local_smoke.py'),'--checkpoint',str(path)]
elif options.get('stage')=='pipeline':
    command = [__import__('sys').executable,str(ROOT/'ltx_pipeline_smoke.py')]
elif options.get('stage')=='catalog':
    command = [__import__('sys').executable,'-m','studio.ltx_assets']
elif options.get('stage')=='audit':
    command = [__import__('sys').executable,str(ROOT/'installation_audit.py')]
elif options.get('stage')=='repair':
    command = [__import__('sys').executable,str(ROOT/'ltx_repair.py')]
else:
    raise ValueError('Unknown LTX verification stage.')
subprocess.run(command,cwd=ROOT,check=True)
