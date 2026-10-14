"""Standard-library hardware scan runs BEFORE any app package/model installation."""
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def option_flag(value):
    if isinstance(value,str):
        if value.lower() in ('false','0','no'):
            return False
        if value.lower() in ('true','1','yes'):
            return True
        raise ValueError('Boolean installer options must be true or false.')
    return bool(value)

def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=30).strip()
    except (OSError, subprocess.SubprocessError):
        return ''

def scan(gpu='auto', model='auto', audio=False, sources=False, download=True,ltx_asset=None):
    from studio.catalog import MODELS
    if gpu not in ('auto','amd','nvidia','cpu') or model not in ['auto',*MODELS]:
        raise ValueError('Unknown GPU or model argument. No dependencies installed.')
    from runtime_install import wheel_commands
    wheel_commands('cpu')  # Reject unsupported OS/architecture before downloads.
    controllers = []
    ram = None
    cpu_name = platform.processor()
    if sys.platform == 'win32':
        # WMI AdapterRAM truncates at 4GiB. Registry qword is used where available.
        script = '''$v = @(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,PNPDeviceID);
        $r = @(Get-ChildItem 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}' -ErrorAction SilentlyContinue | ForEach-Object { Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue } | Select-Object DriverDesc,@{n='vram_bytes';e={$_.'HardwareInformation.qwMemorySize'}});
        @{gpus=$v;registry=$r;ram_bytes=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory;cpu=(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)} | ConvertTo-Json -Depth 5 -Compress'''
        raw = command(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script])
        if raw:
            data = json.loads(raw)
            cpu_name = data.get('cpu') or cpu_name
            ram = data.get('ram_bytes')
            for g in data.get('gpus', []):
                memory = next((r.get('vram_bytes') for r in data.get('registry', []) if r.get('DriverDesc') == g['Name'] and r.get('vram_bytes')), None)
                controllers.append({'name':g['Name'], 'driver':g.get('DriverVersion'), 'vram_gb': round(memory / 2**30, 1) if memory else None, 'memory_source':'registry' if memory else 'unavailable'})
    elif sys.platform == 'darwin':
        ram_raw = command(['sysctl', '-n', 'hw.memsize'])
        ram = int(ram_raw) if ram_raw.isdigit() else None
        cpu_name = command(['sysctl', '-n', 'machdep.cpu.brand_string']) or cpu_name
        raw = command(['system_profiler', 'SPDisplaysDataType', '-json'])
        if raw:
            controllers = [{'name':g.get('sppci_model',g.get('_name','Apple GPU')), 'vram_gb':None, 'memory_source':'unified memory'} for g in json.loads(raw).get('SPDisplaysDataType',[])]
    else:
        pci = command(['lspci'])
        controllers = [{'name':s, 'vram_gb':None, 'memory_source':'unavailable'} for s in pci.splitlines() if 'VGA' in s or 'Display' in s or '3D controller' in s]
        if not controllers:
            for vendor_file in Path('/sys/class/drm').glob('card[0-9]*/device/vendor'):
                vendor = vendor_file.read_text().strip().lower()
                label = {'0x1002':'AMD Radeon','0x10de':'NVIDIA'}.get(vendor)
                if label:
                    controllers.append({'name':label+' ('+vendor_file.parent.parent.name+')','vram_gb':None,'memory_source':'unavailable'})
        if Path('/proc/meminfo').exists():
            ram = int(Path('/proc/meminfo').read_text().split('MemTotal:')[1].split()[0]) * 1024
    nv = command(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'])
    if nv:
        for line in nv.splitlines():
            n, m, d = [x.strip() for x in line.split(',')]
            controllers = [g for g in controllers if n not in g['name']]
            controllers.append({'name':n, 'driver':d, 'vram_gb':round(float(m)/1024,1), 'memory_source':'nvidia-smi'})
    found = {('nvidia' if 'nvidia' in g['name'].lower() else 'amd' if any(x in g['name'].lower() for x in ['amd','radeon']) else 'other') for g in controllers}
    selected = gpu if gpu != 'auto' else 'nvidia' if 'nvidia' in found else 'amd' if 'amd' in found else 'cpu'
    if gpu not in ('auto', 'cpu') and gpu not in found:
        raise RuntimeError(f'Requested {gpu}, but the scan found {found}. No dependencies installed.')
    free = shutil.disk_usage(ROOT).free / 2**30
    report = {'os':platform.platform(), 'platform':sys.platform, 'architecture':platform.machine(), 'cpu':cpu_name, 'logical_cores':os.cpu_count(), 'ram_gb':round(ram/2**30,1) if ram else None, 'disk_free_gb':round(free,1), 'gpus':controllers, 'selected_gpu':selected, 'requested_model':model, 'audio_semantics':audio, 'source_library':sources, 'download_models':download, 'python':sys.version, 'warnings':[]}
    report['ltx_asset'] = ltx_asset
    report['ltx_asset_requested'] = bool(ltx_asset)
    if model.startswith('ltx'):
        from studio.ltx_assets import asset,default_asset
        chosen = asset(ltx_asset) if ltx_asset else default_asset(model)
        if not chosen or chosen['family']!=model or chosen['role']!='checkpoint':
            raise ValueError('LTX installer needs a checkpoint from the selected family. Use its catalog asset ID.')
        if chosen['quantization']=='NVFP4' and selected!='nvidia':
            raise ValueError('No NVFP4 loader is validated on the selected AMD/CPU runtime. Select another variant. No dependencies installed.')
        report['ltx_asset'] = chosen['id']
        report['warnings'].append('LTX-2/2.3/2.5 uses an isolated trainer with real runtime checks; legacy and unsupported quantized formats use compatible-runner handoff. Exact variants require separate validation.')
    if not controllers:
        report['warnings'].append('GPU scan did not return a controller. CPU dataset preparation remains available.')
    if selected == 'amd' and sys.platform == 'win32':
        supported = r'(?:AMD\s+)?Radeon\s+(?:AI PRO R9700|RX 9070(?: XT)?|RX 9060 XT|RX 7900 XTX|PRO W7900(?: Dual Slot)?|RX 7700)'
        if not any(re.fullmatch(supported, g['name'].strip(), re.I) for g in controllers):
            raise RuntimeError('GPU is outside this installer\'s official Windows ROCm support list. Use Linux ROCm or CPU mode. No app packages installed.')
        report['warnings'].append('AMD Windows uses official ROCm 7.2.1 / Python 3.12 wheels. Driver compatibility is verified by an actual GPU forward/backward check after installation.')
    if selected == 'amd' and sys.platform == 'linux':
        kfd = Path('/dev/kfd')
        render = list(Path('/dev/dri').glob('renderD*'))
        if not kfd.exists() or not os.access(kfd,os.R_OK|os.W_OK) or not any(os.access(p,os.R_OK|os.W_OK) for p in render):
            raise RuntimeError('AMD Linux needs a compatible system ROCm/Radeon driver and read/write access to /dev/kfd and /dev/dri/renderD*. Configure render/video group access, restart your login session, then retry; or use CPU mode. No dependencies installed.')
        report['warnings'].append('AMD Linux uses AMD-tested ROCm 7.2.1 Radeon wheels including Triton. The system driver/GPU/distro must match AMD\'s support matrix; an actual GPU backward test is mandatory.')
    if sys.platform not in ('win32','linux'):
        report['selected_gpu'] = 'cpu'
        report['warnings'].append('Training requires Windows/Linux AMD ROCm or NVIDIA CUDA; this platform prepares datasets on CPU.')
    report['support_scope'] = 'Windows/Linux x64: verified CUDA/HIP required for training. Apple Silicon macOS: CPU dataset preparation; MPS training is not implemented. Intel macOS: unavailable with the pinned official PyTorch wheels.'
    if free < 50 and download:
        raise RuntimeError('At least 50GiB free is required for the default model stack. Use --download=false to install the UI alone.')
    ROOT.joinpath('hardware.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print('SCAN_PREFLIGHT_PASSED', flush=True)
    return report

def verify():
    import torch
    from studio.catalog import MODELS
    import importlib
    report = json.loads(ROOT.joinpath('hardware.json').read_text())
    expected = report['selected_gpu']
    modules = ['gradio', 'diffusers', 'transformers', 'accelerate', 'peft', 'safetensors', 'cv2', 'numpy', 'PIL', 'yaml', 'sentencepiece', 'ftfy', 'faster_whisper', 'soundfile', 'psutil']
    for module in modules:
        importlib.import_module(module)
    from diffusers import WanTransformer3DModel, AutoencoderKLWan, WanImageToVideoPipeline
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2AudioForConditionalGeneration
    report['torch'] = torch.__version__
    report['hip'] = torch.version.hip
    report['cuda'] = torch.version.cuda
    report['gpu_verified'] = False
    if expected != 'cpu':
        if not torch.cuda.is_available():
            raise RuntimeError(f'{expected} runtime installed but GPU is unavailable. Check the driver. UI installation is not marked ready.')
        if expected == 'amd' and not torch.version.hip:
            raise RuntimeError('AMD GPU detected but installed torch is not a HIP build.')
        if expected == 'nvidia' and (torch.version.hip or not torch.version.cuda):
            raise RuntimeError('NVIDIA GPU detected but installed torch is not a CUDA build.')
        # Select the largest GPU of the requested vendor on mixed-GPU machines.
        devices = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            devices.append({'index':i, 'name':p.name, 'vram_gb':round(p.total_memory/2**30,1)})
        selected = max(devices, key=lambda d:d['vram_gb'])
        torch.cuda.set_device(selected['index'])
        report['device_index'] = selected['index']
        report['runtime_devices'] = devices
        report['vram_gb'] = selected['vram_gb']
        x = torch.randn(128,128, device='cuda', dtype=torch.float32, requires_grad=True)
        (x @ x).square().mean().backward()
        torch.cuda.synchronize()
        if not torch.isfinite(x.grad).all():
            raise RuntimeError('GPU backward smoke check failed.')
        report['gpu_verified'] = True
        report['bf16_supported'] = torch.cuda.is_bf16_supported()
    if report['requested_model'] == 'auto':
        report['model'] = 'ltx23' if report.get('vram_gb',0) >= 31 else 'wan22-ti2v-5b' if report.get('vram_gb',0) >= 24 else 'wan21-t2v-1.3b'
    else:
        report['model'] = report['requested_model']
    if report['model'] not in MODELS:
        raise RuntimeError('Unknown model preset.')
    m = MODELS[report['model']]
    report['training_capacity_ok'] = report.get('vram_gb',0) >= (31 if m.arch in ('ltx2','ltx23','ltx25') else m.min_vram)
    report['ffmpeg'] = shutil.which('ffmpeg')
    report['ffprobe'] = shutil.which('ffprobe')
    if not report['ffmpeg'] or not report['ffprobe']:
        raise RuntimeError('Pinokio FFmpeg and FFprobe must be present on PATH.')
    ROOT.joinpath('hardware.json').write_text(json.dumps(report,indent=2))
    # A reusable constraints file prevents pip/uv replacing the GPU runtime.
    pinned = []
    for name in ('torch','torchvision','torchaudio'):
        from importlib.metadata import version
        pinned.append(f'{name}=={version(name)}')
    ROOT.joinpath('runtime-constraints.txt').write_text('\n'.join(pinned)+'\n')
    print(json.dumps(report,indent=2), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['scan','verify','models','ready','sources'])
    p.add_argument('--gpu', choices=['auto','amd','nvidia','cpu'], default='auto')
    p.add_argument('--model', default='auto')
    p.add_argument('--audio', action='store_true')
    p.add_argument('--sources', action='store_true')
    p.add_argument('--no-download', action='store_true')
    p.add_argument('--options-file')
    p.add_argument('--ltx-asset')
    a = p.parse_args()
    if a.action == 'scan':
        if a.options_file:
            options = json.loads(Path(a.options_file).read_text())
            scan(options.get('gpu','auto'),options.get('model','auto'),option_flag(options.get('audio',False)),option_flag(options.get('sources',False)),option_flag(options.get('download',True)),options.get('ltx_asset') or None)
        else:
            scan(a.gpu,a.model,a.audio,a.sources,not a.no_download,a.ltx_asset)
    elif a.action == 'verify':
        verify()
    elif a.action == 'models':
        from studio.assets import install_models
        install_models()
    elif a.action == 'sources':
        from studio.assets import download_sources
        download_sources()
    else:
        report = json.loads(ROOT.joinpath('hardware.json').read_text())
        if not report.get('ffprobe'):
            raise RuntimeError('Verification has not passed.')
        ROOT.joinpath('.installed').write_text('Installation checks passed. Training still needs a real dataset smoke run.\n')
        print('AUTOMATED_LORA_STUDIO_INSTALL_READY', flush=True)
