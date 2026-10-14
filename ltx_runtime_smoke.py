"""Actual official audio/video transformer + int8 LoRA backward on the PC GPU.

Reduced model dimensions validate kernels and serialization, not full 22B weights.
"""
import json
import tempfile
from pathlib import Path
import torch
from peft import LoraConfig,inject_adapter_in_model
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file,load_file
from ltx_core.model.transformer.model import LTXModel,LTXModelType
from ltx_core.model.transformer.modality import Modality
from ltx_trainer.quantization import quantize_model

ROOT = Path(__file__).resolve().parent
report = ROOT/'validation/ltx-runtime-report.json'
report.parent.mkdir(exist_ok=True)
try:
    h = json.loads((ROOT/'hardware.json').read_text())
    device = torch.device(f'cuda:{h.get("device_index",0)}')
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    model = LTXModel(model_type=LTXModelType.AudioVideo,num_attention_heads=1,attention_head_dim=64,in_channels=32,out_channels=32,num_layers=1,cross_attention_dim=64,audio_num_attention_heads=1,audio_attention_head_dim=32,audio_in_channels=16,audio_out_channels=16,audio_cross_attention_dim=32)
    # The official constructor contains torch.empty parameters, expecting a
    # checkpoint loader to populate them. Initialize every tiny-fixture parameter
    # explicitly; otherwise uninitialized scale/shift tables make the test flaky.
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'norm' in name and name.endswith('weight'):
                p.fill_(1)
            elif 'norm' in name and name.endswith('bias'):
                p.zero_()
            else:
                torch.nn.init.normal_(p,mean=0,std=.02)
    model.requires_grad_(False)
    model.to(dtype=torch.bfloat16)
    model = quantize_model(model,'int8-quanto',device=device)
    model = inject_adapter_in_model(LoraConfig(r=2,lora_alpha=2,target_modules=['to_q','to_k','to_v','to_out.0']),model)
    model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert parameters
    for p in parameters:
        p.data = p.data.float()
    optimizer = torch.optim.AdamW(parameters,lr=.001)
    def modality(channels,context_dim,axes,tokens):
        positions = torch.zeros(1,axes,tokens,2,device=device)
        positions[:,:,:,1] = 1
        return Modality(latent=torch.randn(1,tokens,channels,device=device,dtype=torch.bfloat16),sigma=torch.tensor([.5],device=device),timesteps=torch.full((1,tokens),.5,device=device),positions=positions,context=torch.randn(1,4,context_dim,device=device,dtype=torch.bfloat16))
    video = modality(32,64,3,8)
    audio = modality(16,32,1,4)
    before = [p.detach().clone() for p in parameters]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        video_output,audio_output = model(video=video,audio=audio,perturbations=None)
        loss = video_output.float().square().mean()+audio_output.float().square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in parameters)
    optimizer.step()
    assert any(not torch.equal(a,b) for a,b in zip(before,parameters))
    with tempfile.TemporaryDirectory(dir=report.parent) as temp:
        path = Path(temp)/'adapter.safetensors'
        save_file({k:v.detach().contiguous().cpu() for k,v in get_peft_model_state_dict(model).items()},path)
        assert load_file(path)
        import numpy as np
        import soundfile as sf
        from ltx_portable_audio import load
        wav = Path(temp)/'audio.wav'
        original = np.sin(np.arange(4800)*2*np.pi*440/48000).astype('float32')
        sf.write(wav,original,48000,subtype='FLOAT')
        decoded,rate = load(str(wav))
        assert rate==48000 and decoded.shape==(1,4800)
        assert torch.allclose(decoded[0],torch.from_numpy(original))
    torch.cuda.synchronize()
    report.write_text(json.dumps({'status':'passed','device':torch.cuda.get_device_name(device),'torch':torch.__version__,'hip':torch.version.hip,'quantization':'int8-quanto','audio_and_video_backward':True,'loss':loss.item(),'full_pretrained_model_tested':False},indent=2))
    print('LTX_AUDIO_VIDEO_INT8_LORA_BACKWARD_PASSED',flush=True)
except Exception as e:
    report.write_text(json.dumps({'status':'failed','error':str(e),'full_pretrained_model_tested':False},indent=2))
    raise
