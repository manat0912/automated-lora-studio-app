"""Native Wan LoRA flow-matching trainer. PyTorch SDPA/AdamW work on HIP and CUDA.

One GPU, batch size one, sequential encoder/cache/transformer stages. No full fine-tune.
"""
import argparse
import gc
import hashlib
import json
import os
import random
from pathlib import Path

def normalized(latent, vae):
    import torch
    mean = torch.tensor(vae.config.latents_mean,device=latent.device,dtype=latent.dtype).view(1,-1,1,1,1)
    inverse_std = 1/torch.tensor(vae.config.latents_std,device=latent.device,dtype=latent.dtype).view(1,-1,1,1,1)
    return (latent-mean)*inverse_std

def conditioning(noisy, t, entry, kind):
    import torch
    mask = torch.ones_like(noisy[:,:1])
    image_embeds = None
    if kind == 'ti2v':
        # Wan 2.2 5B embeds a timestep per spatial/temporal patch.
        first = entry['first_latent'].to(noisy)
        mask[:,:,:first.shape[2]] = 0
        noisy = noisy.clone()
        noisy[:,:,:first.shape[2]] = first
        timestep = (mask[0,0,:,::2,::2]*t[0]).flatten().unsqueeze(0)
    elif kind == 'i2v':
        condition = entry['image_condition'].to(noisy)
        noisy = torch.cat([noisy,condition],dim=1)
        image_embeds = entry['image_embeds'].to(noisy)
        timestep = t
    else:
        timestep = t
    return noisy,timestep,mask,image_embeds

def loss_step(transformer,latent,text,entry,kind,dtype):
    import torch
    sigma = torch.sigmoid(torch.randn((latent.shape[0],),device=latent.device))
    noise = torch.randn_like(latent)
    sigma = sigma.view(-1,1,1,1,1)
    noisy = (1-sigma)*latent + sigma*noise
    target = noise-latent
    x,t,mask,image = conditioning(noisy,sigma.flatten()*1000,entry,kind)
    kwargs = {'hidden_states':x,'timestep':t,'encoder_hidden_states':text,'return_dict':False}
    if image is not None:
        kwargs['encoder_hidden_states_image'] = image
    with torch.autocast('cuda',dtype=dtype,enabled=latent.device.type=='cuda'):
        prediction = transformer(**kwargs)[0]
    diff = (prediction.float()-target.float()).square()*mask
    return diff.sum()/(mask.sum()*latent.shape[1]).clamp_min(1)

def cache_dataset(project,model_path,directory,device,dtype):
    import torch
    import cv2
    import numpy as np
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer, UMT5EncoderModel
    from diffusers import AutoencoderKLWan
    from PIL import Image
    cache = directory/'cache'
    cache.mkdir(exist_ok=True)
    plan = project['plan']
    entries = []
    for clip in project['clips']:
        path = Path(clip['path'])
        caption = path.with_suffix('.txt').read_text(encoding='utf-8')
        key = hashlib.sha256((caption+plan['model']['repo']+str(plan['frames'])+str(plan['width'])+str(plan['height'])).encode()+path.read_bytes()).hexdigest()[:24]
        entries.append({'video':path,'caption':caption,'path':cache/f'{key}.safetensors'})
    missing = [e for e in entries if not e['path'].exists()]
    if not missing:
        print('Reusing validated latent/text cache.',flush=True)
        return entries
    tokenizer = AutoTokenizer.from_pretrained(model_path,subfolder='tokenizer',local_files_only=True)
    encoder = UMT5EncoderModel.from_pretrained(model_path,subfolder='text_encoder',dtype=dtype,local_files_only=True).to(device).eval()
    text_cache = {}
    with torch.inference_mode():
        for e in missing:
            inputs = tokenizer(e['caption'],padding='max_length',max_length=512,truncation=True,return_tensors='pt').to(device)
            embeds = encoder(**inputs).last_hidden_state
            # The Wan pipeline zero-pads beyond the actual prompt length.
            length = int(inputs.attention_mask.sum())
            embeds[:,length:] = 0
            text_cache[str(e['path'])] = embeds.cpu().contiguous()
    del encoder,tokenizer,inputs,embeds
    gc.collect()
    torch.cuda.empty_cache()
    vae = AutoencoderKLWan.from_pretrained(model_path,subfolder='vae',torch_dtype=torch.float32,local_files_only=True).to(device).eval()
    vae.enable_tiling()
    image_cache = {}
    if plan['model']['kind'] == 'i2v':
        from transformers import CLIPVisionModel, CLIPImageProcessor
        processor = CLIPImageProcessor.from_pretrained(model_path,subfolder='image_processor',local_files_only=True)
        image_encoder = CLIPVisionModel.from_pretrained(model_path,subfolder='image_encoder',dtype=dtype,local_files_only=True).to(device).eval()
        with torch.inference_mode():
            for e in missing:
                capture = cv2.VideoCapture(str(e['video']))
                ok,frame = capture.read()
                capture.release()
                if not ok:
                    raise ValueError('Cannot read reference frame from clip.')
                im = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                pixels = processor(images=im,return_tensors='pt').pixel_values.to(device,dtype)
                image_cache[str(e['path'])] = image_encoder(pixels,output_hidden_states=True).hidden_states[-2].cpu().contiguous()
        del image_encoder,processor,pixels
        gc.collect()
        torch.cuda.empty_cache()
    with torch.inference_mode():
        for e in missing:
            cap = cv2.VideoCapture(str(e['video']))
            frames = []
            try:
                while True:
                    ok,bgr = cap.read()
                    if not ok:
                        break
                    frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            finally:
                cap.release()
            if len(frames) != plan['frames']:
                raise ValueError(f'Training clip frame count changed: {e["video"]}')
            video = torch.from_numpy(np.stack(frames)).permute(3,0,1,2).unsqueeze(0).to(device,torch.float32)/127.5-1
            latent = normalized(vae.encode(video).latent_dist.sample(),vae)
            tensors = {'latent':latent.cpu().contiguous(),'text':text_cache[str(e['path'])]}
            if plan['model']['kind'] == 'ti2v':
                tensors['first_latent'] = normalized(vae.encode(video[:,:,:1]).latent_dist.mode(),vae).cpu().contiguous()
            elif plan['model']['kind'] == 'i2v':
                zero_video = torch.zeros_like(video)
                zero_video[:,:,:1] = video[:,:,:1]
                condition = normalized(vae.encode(zero_video).latent_dist.mode(),vae)
                temporal = 2**sum(vae.temperal_downsample)
                mask = torch.zeros((1,1,video.shape[2],latent.shape[3],latent.shape[4]),device=device)
                mask[:,:,0] = 1
                mask = torch.cat([mask[:,:,:1].repeat(1,1,temporal,1,1),mask[:,:,1:]],dim=2)
                mask = mask.reshape(1,-1,temporal,latent.shape[3],latent.shape[4]).transpose(1,2)
                tensors['image_condition'] = torch.cat([mask,condition],dim=1).cpu().contiguous()
                tensors['image_embeds'] = image_cache[str(e['path'])]
            save_file(tensors,e['path'])
            print(f'Encoded video, caption and conditioning: {e["video"].name}',flush=True)
            del video,latent,frames,tensors
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    return entries

def train(project_path):
    import torch
    from diffusers import WanTransformer3DModel, WanPipeline
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from .assets import model_path
    from .catalog import MODELS
    p = json.loads(Path(project_path).read_text())
    plan = p['plan']
    m = MODELS[plan['model']['key']]
    if not m.native:
        raise ValueError('Selected architecture needs a different backend.')
    if not torch.cuda.is_available():
        raise RuntimeError('No working HIP/CUDA GPU.')
    index = int(os.environ.get('STUDIO_DEVICE_INDEX','0'))
    device = torch.device(f'cuda:{index}')
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.manual_seed(plan['seed'])
    random.seed(plan['seed'])
    location = model_path(m.repo)
    directory = Path(project_path).parent
    entries = cache_dataset(p,location,directory,device,dtype)
    transformer = WanTransformer3DModel.from_pretrained(location,subfolder='transformer',torch_dtype=dtype,local_files_only=True)
    transformer.requires_grad_(False)
    transformer.add_adapter(LoraConfig(r=plan['rank'],lora_alpha=plan['rank'],init_lora_weights='gaussian',target_modules=['to_q','to_k','to_v','to_out.0']))
    for param in transformer.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    transformer.enable_gradient_checkpointing()
    transformer.to(device)
    transformer.train()
    parameters = [param for param in transformer.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=plan['learning_rate'],weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    checkpoints = directory/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    state_path = directory/'resume.pt'
    start = 0
    if state_path.exists():
        state = torch.load(state_path,map_location='cpu',weights_only=True)
        # Resume is accepted only for the same dataset/caption/model/shape/rank.
        fingerprint = hashlib.sha256(json.dumps({'entries':[str(e['path']) for e in entries],'plan':{k:plan[k] for k in ['rank','learning_rate','model','width','height','frames','seed']}},sort_keys=True).encode()).hexdigest()
        if state.get('fingerprint') == fingerprint:
            set_peft_model_state_dict(transformer,state['adapter'])
            optimizer.load_state_dict(state['optimizer'])
            scaler.load_state_dict(state['scaler'])
            torch.set_rng_state(state['rng'])
            torch.cuda.set_rng_state_all(state['cuda_rng'])
            start = state['step']
            print(f'Resuming step {start}',flush=True)
    fingerprint = hashlib.sha256(json.dumps({'entries':[str(e['path']) for e in entries],'plan':{k:plan[k] for k in ['rank','learning_rate','model','width','height','frames','seed']}},sort_keys=True).encode()).hexdigest()
    for step in range(start,plan['steps']):
        e = entries[step % len(entries)]
        tensors = load_file(e['path'],device=str(device))
        latent,text = tensors['latent'].to(dtype=dtype),tensors['text'].to(dtype=dtype)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_step(transformer,latent,text,tensors,m.kind,dtype)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at step {step+1}. Training stopped; existing checkpoints kept.')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        if step == start or (step+1)%10 == 0:
            print(json.dumps({'step':step+1,'total':plan['steps'],'loss':round(loss.item(),6),'peak_vram_gb':round(torch.cuda.max_memory_allocated()/2**30,2)}),flush=True)
        if (step+1)%plan['checkpoint_every']==0 or step+1==plan['steps']:
            state_dict = {k:v.detach().cpu() for k,v in get_peft_model_state_dict(transformer).items()}
            name = f'{plan["trigger"]}_{step+1:06d}.safetensors'
            WanPipeline.save_lora_weights(checkpoints,transformer_lora_layers=state_dict,weight_name=name,safe_serialization=True)
            # Only local app-created optimizer state is loaded, using weights_only=True.
            temp = state_path.with_suffix('.tmp')
            torch.save({'step':step+1,'fingerprint':fingerprint,'adapter':state_dict,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},temp)
            temp.replace(state_path)
            (checkpoints/(name+'.json')).write_text(json.dumps({'base_model':m.repo,'architecture':m.arch,'rank':plan['rank'],'step':step+1,'trigger':plan['trigger'],'audio_trained':False},indent=2))
            print(f'Saved {name}',flush=True)
        del tensors,latent,text,loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',required=True)
    train(parser.parse_args().project)
