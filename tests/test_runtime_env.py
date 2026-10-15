import pytest
import threading
from studio.runtime_env import ltx_environment

def test_windows_amd_clears_legacy_overrides_only_in_child_copy():
    source = {'ROCM_USE_WSL':'1','HSA_ENABLE_SDMA':'0','PYTORCH_CUDA_ALLOC_CONF':'max_split_size_mb:512',
        'HIP_VISIBLE_DEVICES':'0','CUDA_VISIBLE_DEVICES':'0','HF_TOKEN_PATH':'managed-auth-path','PATH':'managed-tools'}
    child, removed = ltx_environment(source,'amd','win32')
    assert set(removed) == {'ROCM_USE_WSL','HSA_ENABLE_SDMA','PYTORCH_CUDA_ALLOC_CONF'}
    assert 'ROCM_USE_WSL' in source
    assert child == {'HIP_VISIBLE_DEVICES':'0','CUDA_VISIBLE_DEVICES':'0','HF_TOKEN_PATH':'managed-auth-path','PATH':'managed-tools'}

@pytest.mark.parametrize('gpu,system',[('nvidia','win32'),('amd','linux'),('nvidia','linux'),('cpu','darwin')])
def test_other_runtimes_keep_their_environment(gpu,system):
    source = {'PYTORCH_CUDA_ALLOC_CONF':'max_split_size_mb:512','LD_LIBRARY_PATH':'system-rocm'}
    child, removed = ltx_environment(source,gpu,system)
    assert child == source and child is not source and removed == []

def test_timing_patch_waits_for_loss_and_starts_before_batch():
    from ltx_port import patch_step_timing
    source = '''            for step in range(remaining_steps * cfg.optimization.gradient_accumulation_steps):
                batch = next(data_iter)
                step_start_time = time.time()
                    step_time = (time.time() - step_start_time) * cfg.optimization.gradient_accumulation_steps
                    step_loss = output.loss.detach().mean().item()
'''
    patched = patch_step_timing(source)
    assert patched.index('step_start_time =') < patched.index('batch = next')
    assert patched.index('step_loss =') < patched.index('step_time =')
    with pytest.raises(RuntimeError,match='timing changed'):
        patch_step_timing('unrecognized upstream code')

def test_ltx_worker_receives_native_settings_and_keeps_device_selection(tmp_path,monkeypatch):
    from studio import engine
    captured = {}
    class Process:
        stdout = []
        returncode = 0
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def poll(self): return 0
    monkeypatch.setattr(engine,'ROOT',tmp_path)
    monkeypatch.setattr(engine.sys,'platform','win32')
    monkeypatch.setattr(engine,'hardware',lambda:{'selected_gpu':'amd','device_index':1})
    monkeypatch.setattr(engine,'append_log',lambda *args:None)
    monkeypatch.setenv('ROCM_USE_WSL','1')
    monkeypatch.setenv('HSA_ENABLE_SDMA','0')
    def launch(*args,**kwargs):
        captured.update(kwargs['env'])
        return Process()
    monkeypatch.setattr(engine.subprocess,'Popen',launch)
    engine.subprocess_job([str(tmp_path/'backends/ltx-env/Scripts/python.exe')],'a'*12,threading.Event())
    assert 'ROCM_USE_WSL' not in captured and 'HSA_ENABLE_SDMA' not in captured
    assert captured['HIP_VISIBLE_DEVICES'] == captured['CUDA_VISIBLE_DEVICES'] == '1'
    assert captured['STUDIO_DEVICE_INDEX'] == '0'
