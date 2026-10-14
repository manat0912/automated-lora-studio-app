"""Validate a real local Quanto LTX checkpoint without changing its source.

Random latent/context inputs check pretrained weight loading and trainable
adapters, not caption encoding, media preprocessing or generation quality.
"""
import argparse
import base64
import json
from pathlib import Path
import torch
from accelerate import init_empty_weights
from mmgp import offload
from peft import LoraConfig,get_peft_model
from peft.utils import get_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import save_file
from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
from ltx_core.model.transformer.modality import Modality

ROOT = Path(__file__).resolve().parent

def validate(path):
    report_path = ROOT/'validation/ltx-local-full-report.json'
    report_path.parent.mkdir(exist_ok=True)
    try:
        with safe_open(path,framework='pt',device='cpu') as f:
            metadata = f.metadata()
        if metadata.get('quantization_format')!='quanto':
            raise ValueError('This loader requires the actual Quanto format and its quantization map.')
        # MMGP's safe_open normalizes base64 metadata to a config dictionary.
        config = metadata.get('config')
        if config is None:
            config = json.loads(base64.b64decode(metadata['config_base64']))
        elif isinstance(config,str):
            config = json.loads(config)
        h = json.loads((ROOT/'hardware.json').read_text())
        device = torch.device(f'cuda:{h.get("device_index",0)}')
        torch.cuda.set_device(device)
        report_path.write_text(json.dumps({'status':'loading','source':str(path),'source_modified':False},indent=2))
        with init_empty_weights():
            model = LTXModelConfigurator.from_metadata({'config':config})
        offload.load_model_data(model,str(path),default_dtype=torch.bfloat16,writable_tensors=False,ignore_missing_keys=False,ignore_unused_weights=False)
        # We use PEFT directly. Restore the real torch state_dict signature;
        # MMGP's Diffusers-specific serialization hook is not needed here.
        if hasattr(model,'_real_state_dict'):
            model.state_dict = model._real_state_dict
        model.requires_grad_(False)
        model = get_peft_model(model,LoraConfig(r=2,lora_alpha=2,target_modules=['to_q','to_k','to_v','to_out.0']))
        model.to(device)
        model.base_model.model.set_gradient_checkpointing(True)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters,lr=5e-5)
        c = config['transformer']
        def modality(channels,context_dim,axes,tokens):
            pos = torch.zeros(1,axes,tokens,2,device=device)
            pos[:,:,:,1] = 1
            return Modality(latent=torch.randn(1,tokens,channels,device=device,dtype=torch.bfloat16),sigma=torch.tensor([.5],device=device),timesteps=torch.full((1,tokens),.5,device=device),positions=pos,context=torch.randn(1,4,context_dim,device=device,dtype=torch.bfloat16))
        video = modality(c.get('in_channels',128),c.get('cross_attention_dim',4096),3,4)
        audio = modality(c.get('audio_in_channels',128),c.get('audio_cross_attention_dim',2048),1,4)
        before = [p.detach().clone() for p in parameters]
        report_path.write_text(json.dumps({'status':'backward','source':str(path),'source_modified':False},indent=2))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            video_output,audio_output = model(video=video,audio=audio,perturbations=None)
            loss = video_output.float().square().mean()+audio_output.float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in parameters)
        optimizer.step()
        assert any(not torch.equal(a,b) for a,b in zip(before,parameters))
        target = ROOT/'validation/ltx-local-test.safetensors'
        save_file({k:v.detach().contiguous().cpu() for k,v in get_peft_model_state_dict(model).items()},target)
        torch.cuda.synchronize()
        report_path.write_text(json.dumps({'status':'passed','checkpoint':str(path),'source_modified':False,'device':torch.cuda.get_device_name(device),'torch':torch.__version__,'loss':loss.item(),'audio_and_video_backward':True,'adapter_saved':str(target),'pretrained_weights_tested':True,'real_media_preprocessing_tested':False,'i2v_generation_tested':False,'quality_validated':False,'peak_vram_gib':round(torch.cuda.max_memory_allocated()/2**30,2)},indent=2))
        print('FULL_PRETRAINED_LTX_QUANTO_LORA_BACKWARD_PASSED',flush=True)
    except Exception as e:
        report_path.write_text(json.dumps({'status':'failed','error':str(e),'source_modified':False},indent=2))
        raise

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    validate(Path(parser.parse_args().checkpoint))
