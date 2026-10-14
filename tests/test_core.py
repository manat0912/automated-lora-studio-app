import json
import shutil
import subprocess
import threading
from pathlib import Path
import pytest
from studio import media
from studio.catalog import MODELS
from studio.caption import compose_caption,parse_json
from studio.train_native import conditioning,loss_step

def test_model_frame_and_spatial_rules():
    for m in MODELS.values():
        frames = media.frame_count(5.5,m.fps,m.stride)
        assert (frames-1)%m.stride == 0
        assert frames/m.fps <= 5.5
        w,h = media.dimensions(1920,1080,512,m.spatial)
        assert w%m.spatial==h%m.spatial==0
    assert media.windows([0,4,12],5)==[4]

def test_installer_boolean_arguments_are_respected():
    from setup import option_flag
    assert not option_flag('false')
    assert not option_flag(False)
    assert option_flag('true')
    with pytest.raises(ValueError):
        option_flag('invalid')

def test_ltx_catalog_contains_real_distilled_11_and_quantized_variants():
    from studio.ltx_assets import catalog,default_asset
    assets = catalog()['assets']
    assert any(a['family']=='ltx23' and a['name']=='ltx-2.3-22b-distilled-1.1.safetensors' for a in assets)
    assert any(a['quantization']=='FP8' and a['family']=='ltx23' for a in assets)
    assert any(a['format']=='GGUF' and a['family']=='ltx23' for a in assets)
    assert any(a['quantization']=='NVFP4' and a['family']=='ltx25' for a in assets)
    assert default_asset('ltx23')['role']=='checkpoint'
    assert len({a['id'] for a in assets})==len(assets)

def test_audio_not_labeled_as_visual_training():
    r = {'analysis':{'visual':'three people','motion':'stomp in sequence','camera':'wide shot'},'speech':[{'text':'a chant'}],'audio_description':'audible impacts'}
    assert 'audible impacts' not in compose_caption(r,'hkchoreo')
    assert 'audible impacts' in compose_caption(r,'hkchoreo',True)
    assert 'a chant' in compose_caption(r,'hkchoreo',True)

def test_ltx_full_precision_ram_gate_keeps_quanto_available(tmp_path,monkeypatch):
    from studio import ltx_adapter
    from studio.local_ltx import FILENAME
    monkeypatch.setattr(ltx_adapter,'ROOT',tmp_path)
    marker = tmp_path/'backends/ltx-official/backend-ready.json'
    marker.parent.mkdir(parents=True)
    marker.write_text('{}')
    hardware = {'selected_gpu':'amd','vram_gb':31.9,'ram_gb':31.8}
    variant = {'family':'ltx23','official':True,'quantization':'BF16/FP16','role':'checkpoint'}
    with pytest.raises(ValueError,match='64GiB'):
        ltx_adapter.check({'plan':{'variant':variant}},hardware)
    variant.update(official=False,repo='DeepBeepMeep/LTX-2',file=FILENAME,quantization='INT8 Quanto')
    assert ltx_adapter.check({'plan':{'variant':variant}},hardware)=={}

def test_ltx_32gb_plan_and_retry_preserve_original(tmp_path,monkeypatch):
    from studio import engine
    monkeypatch.setattr(engine,'JOBS',tmp_path/'jobs')
    monkeypatch.setattr(engine,'hardware',lambda:{'vram_gb':31.9,'ram_gb':31.8})
    monkeypatch.setattr(media,'probe',lambda path:{'width':1920,'height':1080,'duration':50,'fps':24,'audio':True})
    calls = []
    monkeypatch.setattr(engine,'dispatch',lambda *args:calls.append(args))
    source = tmp_path/'source.mp4'
    source.write_bytes(b'preserve source bytes')
    original = engine.create(source,'ltx23','hkchoreo','motion','group stomps')
    assert original['plan']['frames']==49
    assert max(original['plan']['width'],original['plan']['height'])==320
    assert original['plan']['rank']==8
    original['plan'].update(width=512,height=288,frames=129,rank=16)
    original['clips'] = [{'source_start':0,'caption':'original caption'}]
    engine.save(original)
    before = engine.jobdir(original['id']).joinpath('project.json').read_bytes()
    result = engine.retry_smaller(original['id'])
    retry = engine.load(result['project_id'])
    assert retry['plan']['frames']==49 and retry['plan']['rank']==8
    assert retry['candidate_starts']==[0]
    assert Path(retry['plan']['source']).read_bytes()==b'preserve source bytes'
    assert engine.jobdir(original['id']).joinpath('project.json').read_bytes()==before
    assert calls==[(retry['id'],'prepare',True)]

def test_audio_model_duration_claim_is_not_used_as_measured_timing():
    from studio.caption import audio_caption
    observed = 'The audio contains a sine wave occurring from the beginning to the end (0.00 to 9.08 seconds). There is no speech.'
    assert '9.08' not in audio_caption(observed)
    assert 'sine wave' in audio_caption(observed)
    assert '0.2 to 12 seconds' not in audio_caption('A tone from 0.2 to 12 seconds.')

def test_caption_contract_rejects_invalid_model_response():
    with pytest.raises(ValueError):
        parse_json('a confident but unstructured response')
    with pytest.raises(ValueError):
        parse_json('{"visual":"v","motion":"m","camera":"c","characters":{},"tags":"wrong"}')

def test_real_ffmpeg_roundtrip(tmp_path):
    source = tmp_path/'source.mp4'
    subprocess.run([shutil.which('ffmpeg'),'-v','error','-y','-f','lavfi','-i','testsrc2=size=320x180:rate=30','-f','lavfi','-i','sine=frequency=300:sample_rate=48000','-t','7','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',str(source)],check=True)
    dest = tmp_path/'clip.mp4'
    result = media.split(source,dest,.5,81,16,256,160)
    assert result['frames']==81 and result['audio'] and result['fps']==16
    wav = tmp_path/'clip.wav'
    media.extract_audio(dest,wav)
    assert media.audio_events(wav)['rms']>0

@pytest.mark.parametrize('audio',[False,True])
def test_fractional_seek_keeps_all_ltx_frames(tmp_path,audio):
    source = tmp_path/'fractional.mp4'
    command = [shutil.which('ffmpeg'),'-v','error','-y','-f','lavfi','-i','testsrc2=size=320x180:rate=30000/1001']
    if audio:
        command += ['-f','lavfi','-i','sine=frequency=440:sample_rate=48000']
    subprocess.run(command+['-t','8','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',str(source)],check=True)
    result = media.split(source,tmp_path/'49frames.mp4',5.375,49,24,320,192)
    assert result['frames']==49 and result['fps']==24
    assert result['audio']==audio
    assert abs(result['duration']-49/24)<.01

def test_ti2v_conditioning_masks_first_frame():
    import torch
    x = torch.randn(1,4,3,4,4)
    first = torch.zeros(1,4,1,4,4)
    y,t,mask,_ = conditioning(x,torch.tensor([500.]),{'first_latent':first},'ti2v')
    assert y[:,:,:1].abs().sum()==0
    assert torch.all(t[:,:4]==0) and torch.all(t[:,4:]==500)
    assert mask[:,:,:1].sum()==0

def test_tiny_wan_real_lora_backward_and_serialization(tmp_path):
    import torch
    from diffusers import WanTransformer3DModel,WanPipeline
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file
    # Actual architecture with tiny dimensions, not a fake training checkpoint.
    model = WanTransformer3DModel(patch_size=(1,2,2),num_attention_heads=2,attention_head_dim=16,in_channels=4,out_channels=4,text_dim=16,freq_dim=16,ffn_dim=64,num_layers=1)
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(r=2,lora_alpha=2,target_modules=['to_q','to_k','to_v','to_out.0']))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    latent = torch.randn(1,4,3,4,4,device=device)
    text = torch.randn(1,8,16,device=device)
    entry = {'first_latent':torch.zeros(1,4,1,4,4,device=device)}
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,lr=.001)
    before = [p.detach().clone() for p in params]
    dtype = torch.bfloat16 if device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16 if device.type=='cuda' else torch.float32
    loss = loss_step(model,latent,text,entry,'ti2v',dtype)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in params)
    optimizer.step()
    assert any(not torch.equal(a,b) for a,b in zip(before,params))
    WanPipeline.save_lora_weights(tmp_path,transformer_lora_layers=get_peft_model_state_dict(model),weight_name='test.safetensors',safe_serialization=True)
    saved = load_file(tmp_path/'test.safetensors')
    assert saved and all('lora_' in k for k in saved)
    restored = WanTransformer3DModel.from_config(model.config)
    WanPipeline.load_lora_into_transformer(saved,restored,adapter_name='motion')
    assert get_peft_model_state_dict(restored,adapter_name='motion')

def test_ui_builds_and_exposes_actions():
    import app
    ui = app.build_ui()
    config = ui.get_config_file()
    api = {d.get('api_name') for d in config['dependencies']}
    assert {'prepare','prepare_and_train','train','generate','hardware','export'}.issubset(api)

def test_ltx_quanto_publisher_and_component_roles():
    from studio.ltx_assets import catalog,classify
    from studio.local_ltx import FILENAME
    assert any(a['repo']=='DeepBeepMeep/LTX-2' and a['file']==FILENAME and a['quantization']=='INT8 Quanto' for a in catalog()['assets'])
    assert classify('DeepBeepMeep/LTX-2','ltx-2.3-22b_text_embedding_projection.safetensors',1,'revision')['role']=='component'

def test_ltx_config_against_installed_official_schema(tmp_path):
    import os
    import yaml
    from studio.engine import ROOT
    from studio.ltx_adapter import configuration
    marker = ROOT/'backends/ltx-official/backend-ready.json'
    if not marker.exists():
        pytest.skip('Optional LTX backend is not installed; real LTX validation runs through verify-ltx.js.')
    backend = json.loads(marker.read_text())
    data = tmp_path/'cache'
    for name in ['latents','conditions','audio_latents']:
        (data/name).mkdir(parents=True)
    checkpoint = tmp_path/'path-placeholder.safetensors'
    checkpoint.write_bytes(b'configuration path placeholder; not a model')
    text = tmp_path/'text'
    text.mkdir()
    p = {'plan':{'rank':16,'learning_rate':5e-5,'steps':2,'checkpoint_every':1,'width':64,'height':64,'frames':9,'fps':24,'seed':42},'clips':[{'media':{'audio':True}}]}
    config = configuration(p,{'local_file':str(checkpoint),'text_encoder_path':str(text)},data,tmp_path/'output',{'vram_gb':31.9,'selected_gpu':'amd'})
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    env = {**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'}
    subprocess.run([backend['python'],'-c','import sys,yaml; from ltx_trainer.config import LtxTrainerConfig; c=LtxTrainerConfig(**yaml.safe_load(open(sys.argv[1]))); assert c.training_strategy.audio.is_generated; assert c.acceleration.quantization=="int8-quanto"; assert c.optimization.optimizer_type=="adamw"',str(path)],check=True,env=env)
