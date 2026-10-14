"""Capture installed dependency/model provenance without modifying runtimes."""
import importlib.metadata
import json
import subprocess
from pathlib import Path
from studio.local_ltx import discover

ROOT = Path(__file__).resolve().parent
report_dir = ROOT/'validation'
report_dir.mkdir(exist_ok=True)
hardware = json.loads((ROOT/'hardware.json').read_text())
core = {d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
ltx = {}
marker = ROOT/'backends/ltx-official/backend-ready.json'
if marker.exists():
    backend = json.loads(marker.read_text())
    code = 'import importlib.metadata,json; print(json.dumps({d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}))'
    ltx = json.loads(subprocess.check_output([backend['python'],'-c',code],text=True,encoding='utf-8'))
models = json.loads((ROOT/'models/downloads.json').read_text())
sources = json.loads((ROOT/'sources/manifest.json').read_text()) if (ROOT/'sources/manifest.json').exists() else []
audit = {'hardware':hardware,'core_dependencies':core,'ltx_dependencies':ltx,'model_downloads':models,'local_ltx_bundles':discover(),'integration_sources':sources,'source_library_is_runtime_validation':False}
(report_dir/'installation-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
for name,pins in [('core',core),('ltx',ltx)]:
    (report_dir/(name+'-environment.txt')).write_text('\n'.join(f'{k}=={v}' for k,v in sorted(pins.items()))+'\n',encoding='utf-8')
print(f'INSTALLATION_AUDIT_SAVED: {len(core)} core packages, {len(ltx)} LTX packages, {len(sources)} source repositories.',flush=True)
