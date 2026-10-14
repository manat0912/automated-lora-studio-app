"""Reference-image sampling with the official LTX validation pipeline and LoRA."""
import argparse
import gc
import json
import os
import shutil
from pathlib import Path
import torch
import yaml
from peft import LoraConfig,get_peft_model,set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file
from ltx_trainer.config import ValidationConfig
from ltx_trainer.model_loader import load_transformer
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.quantization import quantize_model
from ltx_trainer.validation_runner import ValidationRunner

def infer(path):
    path = Path(path)
    project = json.loads(path.read_text())
    plan = project['plan']
    if not plan.get('reference') or not project.get('checkpoints'):
        raise ValueError('LTX preview requires a saved LoRA and reference image.')
    cfg = yaml.safe_load(path.parent.joinpath('ltx-train.yaml').read_text())
    device = torch.device(f'cuda:{os.environ.get("STUDIO_DEVICE_INDEX","0")}')
    torch.cuda.set_device(device)
    variant = plan.get('variant',{})
    distilled = variant.get('distilled',False)
    validation = ValidationConfig(samples=[{'prompt':f'{plan["trigger"]}, {plan["instructions"] or "perform the learned movement"}, retain the number and appearance of the characters in the reference image','conditions':[{'type':'first_frame','image_or_video':plan['reference']}]}],video_dims=(plan['width'],plan['height'],plan['frames']),frame_rate=plan['fps'],seed=plan['seed'],inference_steps=plan.get('preview_steps') or variant.get('generation_steps') or 30,video_cfg_scale=1.0 if distilled else 3.0,audio_cfg_scale=1.0 if distilled else 7.0,video_stg_scale=0.0 if distilled else 1.0,audio_stg_scale=0.0 if distilled else 1.0,interval=None,skip_initial_validation=True)
    # Text encoder, conditioning VAE and decoders cache sequentially before the
    # large quantized transformer is moved to GPU.
    runner = ValidationRunner(validation,cfg['model']['model_path'],cfg['model']['text_encoder_path'],video_vae_path=cfg['model'].get('video_vae_path'),audio_vae_path=cfg['model'].get('audio_vae_path'),load_text_encoder_in_8bit=cfg['acceleration']['load_text_encoder_in_8bit'])
    gc.collect()
    torch.cuda.empty_cache()
    model = load_transformer(cfg['model']['model_path'],device='cpu',dtype=torch.bfloat16)
    if cfg['acceleration'].get('quantization'):
        model = quantize_model(model,cfg['acceleration']['quantization'],device=device)
    model.requires_grad_(False)
    model = get_peft_model(model,LoraConfig(r=plan['rank'],lora_alpha=plan['rank'],target_modules=['to_q','to_k','to_v','to_out.0']))
    state = load_file(project['checkpoints'][-1])
    state = {('base_model.model.'+k.removeprefix('diffusion_model.')) if k.startswith('diffusion_model.') else k:v for k,v in state.items()}
    expected = set(get_peft_model_state_dict(model))
    if set(state)!=expected:
        raise ValueError('LTX preview checkpoint does not match all expected adapter tensors for this rank/model.')
    result = set_peft_model_state_dict(model,state)
    if result.unexpected_keys:
        raise ValueError('LTX preview LoRA contains unexpected weights: '+str(result.unexpected_keys[:8]))
    from peft.tuners.lora.layer import LoraLayer
    for layer in model.modules():
        if isinstance(layer,LoraLayer):
            layer.scale_layer(.7)
    model.eval().to(device)
    output = runner.run(model,step=0,output_dir=path.parent,device=device,progress=TrainingProgress(enabled=False,total_steps=1))
    if not output or not output[0][1].is_file():
        raise RuntimeError('LTX I2V pipeline produced no saved preview.')
    shutil.copy2(output[0][1],path.parent/'preview.mp4')
    print('LTX_REFERENCE_IMAGE_LORA_PREVIEW_SAVED',flush=True)

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project',required=True)
    infer(p.parse_args().project)
