"""Reviewable compatibility changes to a copied, pinned official source tree."""
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def patch_step_timing(code):
    """Include batch loading and GPU completion in the displayed step duration."""
    replacements = [
        ('            for step in range(remaining_steps * cfg.optimization.gradient_accumulation_steps):\n',
         '            for step in range(remaining_steps * cfg.optimization.gradient_accumulation_steps):\n                step_start_time = time.perf_counter()  # STUDIO wall-clock timing\n'),
        ('                step_start_time = time.time()\n', ''),
        ('                    step_time = (time.time() - step_start_time) * cfg.optimization.gradient_accumulation_steps\n                    step_loss = output.loss.detach().mean().item()',
         '                    step_loss = output.loss.detach().mean().item()\n                    step_time = (time.perf_counter() - step_start_time) * cfg.optimization.gradient_accumulation_steps'),
    ]
    for needle,replacement in replacements:
        if code.count(needle) != 1:
            raise RuntimeError('Pinned LTX step timing changed; review complete-step measurement.')
        code = code.replace(needle,replacement,1)
    return code

def patch(repo,revision):
    port = ROOT/'backends/ltx-studio-port'
    shutil.copytree(Path(repo)/'packages',port,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['LICENSE','LICENSE.md','NOTICE']:
        original = Path(repo)/name
        if original.is_file():
            shutil.copy2(original,port/name)
    source = port/'ltx-trainer/src/ltx_trainer'
    shutil.copy2(ROOT/'ltx_quanto_loader.py',source/'studio_quanto_loader.py')
    changed = ['ltx-trainer/src/ltx_trainer/studio_quanto_loader.py']
    if os.name=='nt':
        pyproject = port/'ltx-trainer/pyproject.toml'
        content = pyproject.read_text(encoding='utf-8')
        needle = '"torchcodec>=0.8.1,<1.0.0",'
        if needle not in content:
            raise RuntimeError('Pinned LTX TorchCodec requirement changed; review this Windows port.')
        pyproject.write_text(content.replace(needle,'"torchcodec>=0.8.1,<1.0.0; sys_platform != \'win32\'",'),encoding='utf-8')
        shutil.copy2(ROOT/'ltx_portable_audio.py',source/'studio_portable_audio.py')
        changed += ['ltx-trainer/pyproject.toml','ltx-trainer/src/ltx_trainer/studio_portable_audio.py']
        for name in ['process_videos.py','validation_runner.py']:
            target = source/name
            text = target.read_text(encoding='utf-8')
            if 'torchaudio.load(' not in text:
                raise RuntimeError('Pinned LTX audio API changed; review portable audio decoding.')
            target.write_text(text.replace('import torchaudio\n','import torchaudio\nfrom ltx_trainer.studio_portable_audio import load as studio_audio_load\n').replace('torchaudio.load(','studio_audio_load(') if 'import torchaudio\n' in text else 'from ltx_trainer.studio_portable_audio import load as studio_audio_load\n'+text.replace('torchaudio.load(','studio_audio_load('),encoding='utf-8')
            changed.append('ltx-trainer/src/ltx_trainer/'+name)
    target = source/'model_loader.py'
    code = target.read_text(encoding='utf-8')
    def insert(function,needle,addition):
        nonlocal code
        start = code.index('def '+function+'(')
        end = code.find('\ndef ',start+4)
        end = len(code) if end<0 else end
        section = code[start:end]
        if needle not in section:
            raise RuntimeError('Pinned LTX loader changed: '+function)
        code = code[:start]+section.replace(needle,addition+'\n'+needle,1)+code[end:]
    insert('load_transformer','    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder','    from .studio_quanto_loader import is_quanto, load_transformer as studio_load_transformer\n    if is_quanto(checkpoint_path):\n        return studio_load_transformer(checkpoint_path,device,dtype)')
    insert('load_text_encoder','    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder','    from .studio_quanto_loader import load_gemma\n    if (Path(gemma_model_path)/"studio-local-ltx.json").is_file():\n        return load_gemma(gemma_model_path,device,dtype)')
    insert('embedding_weight_paths','    from ltx_core.text_encoders.gemma.gemma_assets import is_safetensors_file','    from .studio_quanto_loader import embedding_paths\n    if text_encoder_path is not None and Path(text_encoder_path).is_dir():\n        local_paths = embedding_paths(text_encoder_path)\n        if local_paths:\n            return local_paths')
    target.write_text(code,encoding='utf-8')
    changed.append('ltx-trainer/src/ltx_trainer/model_loader.py')
    target = source/'trainer.py'
    code = target.read_text(encoding='utf-8')
    code = patch_step_timing(code)
    if 'tmp_path.rename(state_path)' not in code:
        raise RuntimeError('Pinned LTX state-save implementation changed; review Windows atomic replacement.')
    # A checkpoint is saved both at the interval and on finalization. Windows
    # rename rejects an existing target; replace keeps the atomic-save contract.
    code = code.replace('tmp_path.rename(state_path)','tmp_path.replace(state_path)')
    needle = 'self._global_step % cfg.checkpoints.interval == 0'
    if needle not in code:
        raise RuntimeError('Pinned LTX checkpoint interval changed; review first-step recovery saving.')
    code = code.replace(needle,'(self._global_step % cfg.checkpoints.interval == 0 or self._global_step == initial_step + 1)',1)
    needle = 'if disable_progress_bars and IS_MAIN_PROCESS and self._global_step % 20 == 0:'
    if needle not in code:
        raise RuntimeError('Pinned LTX progress logging changed; review per-step reporting.')
    code = code.replace(needle,'if disable_progress_bars and IS_MAIN_PROCESS:',1)
    needle = '        self._accelerator.end_training()'
    if needle not in code:
        raise RuntimeError('Pinned LTX training cleanup changed; review the single-GPU HIP path.')
    code = code.replace(needle,'        if torch.distributed.is_available():\n            self._accelerator.end_training()\n        else:\n            for tracker in self._accelerator.trackers:\n                tracker.finish()',1)
    target.write_text(code,encoding='utf-8')
    changed.append('ltx-trainer/src/ltx_trainer/trainer.py')
    target = source/'gpu_utils.py'
    code = target.read_text(encoding='utf-8')
    needle = '    try:\n        device_id = device.index if device.index is not None else 0'
    if needle not in code:
        raise RuntimeError('Pinned LTX GPU monitor changed; review the HIP memory query.')
    code = code.replace(needle,'    if torch.version.hip:\n        free,total = torch.cuda.mem_get_info(device)\n        return (total-free)/1024**3\n'+needle,1)
    target.write_text(code,encoding='utf-8')
    changed.append('ltx-trainer/src/ltx_trainer/gpu_utils.py')
    target = port/'ltx-core/src/ltx_core/loader/sft_loader.py'
    code = target.read_text(encoding='utf-8')
    needle = 'parsed[key] = json.loads(value)'
    if needle not in code:
        raise RuntimeError('Pinned LTX metadata parser changed; review Quanto metadata normalization.')
    code = code.replace(needle,'parsed[key] = json.loads(value) if isinstance(value,str) else value')
    code = code.replace('            return parsed','            if "config" not in parsed and "config_base64" in parsed:\n                import base64\n                parsed["config"] = json.loads(base64.b64decode(parsed["config_base64"]))\n            return parsed')
    target.write_text(code,encoding='utf-8')
    changed.append('ltx-core/src/ltx_core/loader/sft_loader.py')
    (port/'studio-port.json').write_text(json.dumps({'upstream_revision':revision,'changes':changed,'reason':'Windows real PCM audio decoding; actual prequantized LTX/Gemma loading with strict weight checks; metadata normalization. Official models, conditioning and losses remain intact. No stub GPU/distributed/codec modules.'},indent=2))
    return port
