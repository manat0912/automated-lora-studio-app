"""Called only by Pinokio torch.js, in its Python 3.12 virtual environment."""
import json
import subprocess
import sys
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def wheel_commands(gpu, system=None, machine=None):
    """Official vendor release sets; never resolve CUDA on a HIP/macOS host."""
    system = system or sys.platform
    machine = (machine or platform.machine()).lower()
    packages = ['torch==2.9.1', 'torchvision==0.24.1', 'torchaudio==2.9.1']
    if system == 'darwin':
        if machine not in ('arm64', 'aarch64'):
            raise RuntimeError('PyTorch 2.9.1 has no official Intel macOS wheel. Use an Apple Silicon Mac or Windows/Linux x64. No dependencies installed.')
        return [packages]  # macOS arm64 wheels are on PyPI, including CPU/MPS.
    if system not in ('win32', 'linux') or machine not in ('amd64', 'x86_64', 'x64'):
        raise RuntimeError('This vendor runtime requires Windows/Linux x64. No dependencies installed.')
    if gpu == 'amd' and system == 'win32':
        base = 'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/'
        return [[base+x for x in ['rocm_sdk_core-7.2.1-py3-none-win_amd64.whl', 'rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl', 'rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl', 'rocm-7.2.1.tar.gz']],
                [base+x for x in ['torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl', 'torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl', 'torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl']]]
    if gpu == 'amd':
        # AMD's tested Radeon release includes matching Triton and companion builds.
        base = 'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/'
        return [[base+x for x in ['torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl', 'torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl', 'torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl', 'triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl']]]
    return [packages + ['--index-url', 'https://download.pytorch.org/whl/'+('cu128' if gpu == 'nvidia' else 'cpu')]]

def install():
    hardware = json.loads((ROOT/'hardware.json').read_text())
    gpu = hardware['selected_gpu']
    uv = ['uv','pip','install','--python',sys.executable]
    if sys.version_info[:2] != (3,12):
        raise RuntimeError('This installer requires Python 3.12 for the pinned vendor wheels.')
    for command in wheel_commands(gpu):
        subprocess.run(uv + command,check=True)
    # Prevent a downstream dependency resolving CUDA wheels over HIP wheels.
    from importlib.metadata import version
    constraints = ROOT/'runtime-constraints.txt' if Path(sys.prefix).resolve() == (ROOT/'env').resolve() else Path(sys.prefix)/'runtime-constraints.txt'
    constraints.write_text('\n'.join(f'{n}=={version(n)}' for n in ['torch','torchvision','torchaudio'])+'\n')

if __name__ == '__main__':
    install()
