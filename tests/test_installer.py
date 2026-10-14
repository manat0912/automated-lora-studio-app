"""Installer regressions without downloading weights or replacing GPU wheels."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]

def load(name):
    spec = importlib.util.spec_from_file_location(name,ROOT/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize('system,machine,gpu,expected',[
    ('win32','AMD64','nvidia','cu128'),
    ('linux','x86_64','nvidia','cu128'),
    ('win32','AMD64','amd','windows/rocm-rel-7.2.1'),
    ('linux','x86_64','amd','manylinux/rocm-rel-7.2.1'),
    ('win32','AMD64','cpu','/cpu'),
    ('linux','x86_64','cpu','/cpu'),
])
def test_vendor_release_selection(system,machine,gpu,expected):
    commands = load('runtime_install').wheel_commands(gpu,system,machine)
    assert expected in str(commands)
    if gpu == 'amd':
        assert 'cu128' not in str(commands)
    if gpu == 'amd' and system == 'linux':
        assert 'triton-3.5.1' in str(commands)
        assert 'rocm7.1' not in str(commands)

def test_mac_uses_pypi_and_rejects_unavailable_intel_wheels():
    runtime = load('runtime_install')
    assert runtime.wheel_commands('cpu','darwin','arm64') == [['torch==2.9.1','torchvision==0.24.1','torchaudio==2.9.1']]
    with pytest.raises(RuntimeError,match='Intel macOS'):
        runtime.wheel_commands('cpu','darwin','x86_64')
    with pytest.raises(RuntimeError,match='x64'):
        runtime.wheel_commands('nvidia','win32','ARM64')

def test_download_false_and_sources_do_not_download_models():
    access = load('model_access')
    assert access.plan({'component':'install'},{'download_models':False})['requests'] == []
    assert access.plan({'component':'sources'},{})['requests'] == []

def test_install_plan_covers_base_caption_speech_and_audio():
    access = load('model_access')
    result = access.plan({'component':'install'},{'download_models':True,'model':'wan21-t2v-1.3b','audio_semantics':True})
    assert {r['repo'] for r in result['requests']} == {
        'Qwen/Qwen2.5-VL-3B-Instruct','Systran/faster-whisper-small',
        'Qwen/Qwen2-Audio-7B-Instruct','Wan-AI/Wan2.1-T2V-1.3B-Diffusers'}

def test_gated_model_download_starts_as_soon_as_access_is_granted(monkeypatch):
    access = load('model_access')
    states = iter([True,True,False])
    events = []
    monkeypatch.setattr(access,'probe',lambda request:{'needs_access':next(states)})
    monkeypatch.setattr(access.time,'sleep',lambda seconds:events.append('wait'))
    monkeypatch.setattr(access,'download',lambda *a,**kw:events.append('download'))
    access.wait_download({'repo':'test/gated','patterns':['*.safetensors'],'revision':'pin'})
    assert events == ['wait','wait','download']

def test_pending_access_never_downloads_or_marks_ready(monkeypatch):
    access = load('model_access')
    monkeypatch.setattr(access,'probe',lambda request:{'needs_access':True})
    monkeypatch.setattr(access,'download',lambda *a,**kw:pytest.fail('Unauthorized download'))
    with pytest.raises(RuntimeError,match='still pending'):
        access.wait_download({'repo':'test/gated'},timeout=0)

def test_probe_checks_gated_access_without_recording_tokens(monkeypatch):
    import huggingface_hub
    access = load('model_access')
    checks = []
    fake = SimpleNamespace(model_info=lambda *a,**kw:SimpleNamespace(gated='auto'),auth_check=lambda **kw:checks.append(kw))
    monkeypatch.setattr(huggingface_hub,'HfApi',lambda **kw:fake)
    monkeypatch.setattr(huggingface_hub,'get_token',lambda:'private-test-token')
    result = access.probe({'repo':'test/gated'})
    assert checks == [{'repo_id':'test/gated','repo_type':'model'}]
    assert result == {'repo':'test/gated','url':'https://huggingface.co/test/gated','needs_access':False}
    assert 'token' not in str(result)

def test_ltx_plan_includes_matching_gated_encoder():
    from studio.ltx_assets import download_requests
    result = download_requests({'repo':'Lightricks/LTX-2.3','file':'checkpoint.safetensors','revision':'pin','family':'ltx23','role':'checkpoint','quantization':'BF16/FP16'})
    assert result[0]['revision'] == 'pin'
    assert result[1]['repo'] == 'Lightricks/gemma-3-12b-it-qat-q4_0-unquantized'
