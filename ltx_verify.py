"""Pinokio verification dispatch with validated args and no shell interpolation."""
import argparse
import json
import subprocess
import re
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
elif options.get('stage')=='benchmark':
    project = options.get('project','')
    profile = options.get('profile','inherited')
    steps = int(options.get('benchmark_steps',3))
    if not re.fullmatch(r'[a-f0-9]{12}',project) or profile not in ('inherited','native') or not 3 <= steps <= 32:
        raise ValueError('Benchmark requires an existing project ID and inherited/native profile.')
    command = [str(python),str(ROOT/'ltx_benchmark.py'),'--project',project,'--profile',profile,'--steps',str(steps)]
else:
    raise ValueError('Unknown LTX verification stage.')
subprocess.run(command,cwd=ROOT,check=True,timeout=900 if options.get('stage')=='benchmark' else None)
