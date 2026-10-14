import argparse
import json
import os
from pathlib import Path

def infer(project_path):
    import torch
    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import export_to_video
    from PIL import Image
    from .assets import model_path
    p = json.loads(Path(project_path).read_text())
    plan = p['plan']
    if plan['model']['kind']=='t2v':
        raise ValueError('This preset is T2V. Choose an I2V/TI2V base to use characters from a reference image.')
    device = int(os.environ.get('STUDIO_DEVICE_INDEX','0'))
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipeline = WanImageToVideoPipeline.from_pretrained(model_path(plan['model']['repo']),torch_dtype=dtype,local_files_only=True)
    checkpoint = Path(p['checkpoints'][-1])
    pipeline.load_lora_weights(checkpoint.parent,weight_name=checkpoint.name,adapter_name='motion')
    pipeline.set_adapters(['motion'],adapter_weights=[.7])
    pipeline.enable_model_cpu_offload(gpu_id=device)
    pipeline.vae.enable_tiling()
    prompt = f'{plan["trigger"]}, {plan["instructions"] or "perform the learned motion"}, retain the number and appearance of the characters in the reference image'
    image = Image.open(plan['reference']).convert('RGB')
    output = pipeline(image=image,prompt=prompt,negative_prompt='distorted anatomy, disappearing characters, identity change, flicker',height=plan['height'],width=plan['width'],num_frames=plan['frames'],num_inference_steps=plan.get('preview_steps',30),guidance_scale=5.0,generator=torch.Generator(device=f'cuda:{device}').manual_seed(plan['seed'])).frames[0]
    export_to_video(output,str(Path(project_path).parent/'preview.mp4'),fps=plan['fps'])
    print('Reference-image preview saved. Review character count, identity, motion, and temporal consistency.',flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project',required=True)
    infer(p.parse_args().project)
