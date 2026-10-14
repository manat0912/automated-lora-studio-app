"""Fail-fast dependency, runtime, download, and readiness stages."""
import subprocess
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def finish(phase):
    if not ROOT.joinpath('hardware.json').is_file():
        raise RuntimeError('Hardware scan must complete before installation.')
    if not ROOT.joinpath('runtime-constraints.txt').is_file():
        raise RuntimeError('GPU runtime installation must complete before dependencies.')
    ROOT.joinpath('.installed').unlink(missing_ok=True)
    if phase in ('all', 'dependencies'):
        subprocess.run(['uv','pip','install','--python',sys.executable,'-r','requirements.txt','-c','runtime-constraints.txt'],check=True)
        subprocess.run(['uv','pip','check','--python',sys.executable],check=True)
        subprocess.run([sys.executable,'setup.py','verify'],check=True)
        hardware = json.loads(ROOT.joinpath('hardware.json').read_text())
        if hardware['model'] in ('ltx2','ltx23','ltx25') and hardware.get('gpu_verified') and hardware.get('training_capacity_ok'):
            subprocess.run([sys.executable,'ltx_install.py'],check=True)
        if hardware['model'].startswith('ltx') and hardware.get('download_models'):
            subprocess.run([sys.executable,'-m','studio.ltx_assets'],check=True)
    if phase in ('all', 'finalize'):
        for stage in (['models','sources','ready'] if phase == 'all' else ['sources','ready']):
            subprocess.run([sys.executable,'setup.py',stage],check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['all','dependencies','finalize'], default='all')
    finish(parser.parse_args().phase)
