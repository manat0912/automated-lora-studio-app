import gc
import json
import re
import os
from pathlib import Path
from .assets import model_path, download
from .catalog import VISION_REPO, SPEECH_REPO, AUDIO_REPO
from . import media

def parse_json(text):
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end < start:
        raise ValueError('Vision model did not return JSON. Retry captioning or edit the caption manually.')
    value = json.loads(text[start:end+1])
    for key in ['visual','motion','characters','camera','tags']:
        if key not in value:
            raise ValueError(f'Vision response is missing {key}.')
    if not isinstance(value['tags'],list) or not all(isinstance(x,str) for x in value['tags']):
        raise ValueError('Vision tags must be a list of strings.')
    for key in ['visual','motion','camera']:
        if not isinstance(value[key],str):
            raise ValueError(f'Vision {key} must be a string.')
        if value[key].strip() in ('','...','…'):
            raise ValueError(f'Vision {key} contains a placeholder, not an observation.')
    return value

def free(model):
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def audio_caption(text):
    # Qwen audio can hallucinate numeric durations. Measured/ASR timing lives in
    # separate metadata; remove generated time ranges from training prose.
    text = re.sub(r'\([^)]*(?:seconds?|\d\s*s\b)[^)]*\)','',text,flags=re.I)
    text = re.sub(r'\b(?:from\s+)?\d+(?:\.\d+)?\s*(?:to|[-–])\s*\d+(?:\.\d+)?\s*(?:seconds?|s)\b','',text,flags=re.I)
    return re.sub(r'\s+',' ',text).strip()

def caption_clips(records, trigger, purpose, instructions, semantic_audio, device, cancel, log):
    # Speech, audio events, and vision use separate sequential models.
    speech_path = download(SPEECH_REPO,['*.json','*.bin','*.txt'],log)
    for record in records:
        if cancel.is_set():
            raise InterruptedError('Cancelled during speech analysis.')
        path = Path(record['path'])
        if record['media']['audio']:
            wav = path.with_suffix('.wav')
            media.extract_audio(path,wav,cancel)
            record['audio_events'] = media.audio_events(wav)
            import sys
            output = path.with_suffix('.speech.json')
            media.run([sys.executable,'-m','studio.speech_worker','--model',str(speech_path),'--audio',str(wav),'--output',str(output)],cancel)
            transcript = json.loads(output.read_text(encoding='utf-8'))
            record['speech'] = transcript['segments']
            record['speech_language'] = transcript['language']
        else:
            record['speech'] = []
            record['audio_events'] = {'transients_seconds':[],'method':'No audio stream'}
        log(f'Speech and audio timing: {path.name}')
    gc.collect()

    if semantic_audio and any(r['media']['audio'] for r in records):
        import torch
        import soundfile as sf
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        location = download(AUDIO_REPO,['*.json','*.safetensors','*.txt','*.model','*.jinja'],log)
        processor = AutoProcessor.from_pretrained(location,local_files_only=True)
        dtype = torch.float16 if device.startswith('cuda') else torch.float32
        load_options = {'dtype':dtype,'local_files_only':True}
        offload = False
        if device.startswith('cuda'):
            index = int(device.split(':')[1])
            available,total = torch.cuda.mem_get_info(index)
            budget = min(int(total*.70),max(2*2**30,available-3*2**30))
            requested = os.environ.get('STUDIO_AUDIO_MAX_VRAM_GIB')
            if requested:
                budget = min(budget,int(float(requested)*2**30))
            if budget<18*2**30:
                import psutil
                folder = location/'inference-offload'
                folder.mkdir(exist_ok=True)
                load_options.update(device_map='auto',max_memory={index:budget,'cpu':int(psutil.virtual_memory().available*.70)},offload_folder=str(folder))
                offload = True
                log('Audio model uses CPU/disk weight offload to fit the available GPU memory.')
        model = Qwen2AudioForConditionalGeneration.from_pretrained(location,**load_options).eval()
        if not offload:
            model.to(device)
        for r in records:
            if cancel.is_set():
                raise InterruptedError('Cancelled during audio captioning.')
            if not r['media']['audio']:
                continue
            audio, rate = sf.read(Path(r['path']).with_suffix('.wav'))
            conversation = [{'role':'user','content':[{'type':'audio','audio_url':'local.wav'},{'type':'text','text':f'This clip lasts {r["media"]["duration"]:.2f} seconds. Describe audible sounds and rhythm in words only. Distinguish speech, music, and percussive impacts. Do not infer the source of an impact unless clear. Do not report timestamps or estimate durations; measured timing is recorded separately.'}]}]
            text = processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=False)
            inputs = processor(text=text,audio=[audio],sampling_rate=rate,return_tensors='pt',padding=True).to(device)
            with torch.inference_mode():
                result = model.generate(**inputs,max_new_tokens=160,do_sample=False)
            r['audio_description_raw'] = processor.batch_decode(result[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0].strip()
            r['audio_description'] = audio_caption(r['audio_description_raw'])
            log(f'Audio description: {Path(r["path"]).name}')
        del model, processor, inputs, result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    location = download(VISION_REPO,['*.json','*.safetensors','*.txt','*.model','*.jinja'],log)
    processor = AutoProcessor.from_pretrained(location,local_files_only=True,min_pixels=128*28*28,max_pixels=384*28*28)
    dtype = torch.float16 if device.startswith('cuda') else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(location,dtype=dtype,attn_implementation='sdpa',local_files_only=True).to(device).eval()
    for r in records:
        if cancel.is_set():
            raise InterruptedError('Cancelled during visual captioning.')
        frames = media.storyboard(r['path'])
        identity_rule = ('Describe performers generically by count, positions, and roles. Omit distinctive faces, names, clothing identity, ethnicity, and source identity so the motion trigger does not intentionally represent those people.' if purpose=='motion' else 'Describe consistent visible appearance and positions; do not guess identities or names.')
        prompt = f'''These images are chronological samples of ONE video clip. Their timestamps are {[t for t,_ in frames]}. Describe observable changes, coordination, body movement, camera motion and multi-person relationships. Do not claim skeletal/pose tracking, unseen joints, or exact synchronization from sparse frames. {identity_rule}
User movement instructions: {instructions or 'Describe the visible motion faithfully.'}
Return ONLY a JSON object. Required keys: visual (specific scene description), motion (specific observed changes), characters (object with count integer or null, roles list, relationships text), camera (observable camera/framing), tags (list of descriptive strings), uncertainty (what cannot be established). Use concrete observations, never placeholders or an echoed schema. Count may be null when uncertain. Do not invent audio or speech.'''
        content = [{'type':'image','image':im} for _,im in frames] + [{'type':'text','text':prompt}]
        text = processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
        inputs = processor(text=[text],images=[im for _,im in frames],return_tensors='pt',padding=True).to(device)
        parsed = None
        for attempt in range(2):
            with torch.inference_mode():
                result = model.generate(**inputs,max_new_tokens=640,do_sample=False)
            decoded = processor.batch_decode(result[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
            try:
                parsed = parse_json(decoded)
                break
            except (ValueError,json.JSONDecodeError):
                if attempt == 1:
                    raise ValueError(f'Could not create a valid caption for {Path(r["path"]).name}. Raw response: {decoded[:500]}')
        r['analysis'] = parsed
        r['caption'] = compose_caption(r,trigger)
        r['caption_status'] = 'AI generated; review recommended'
        Path(r['path']).with_suffix('.txt').write_text(r['caption'],encoding='utf-8')
        Path(r['path']).with_suffix('.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
        log(f'Caption and motion tags: {Path(r["path"]).name}')
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records

def compose_caption(record,trigger,train_audio=False):
    a = record['analysis']
    parts = [trigger,a['visual'],a['motion'],a['camera']]
    # Wan learns visual motion only. Audio descriptions are retained in metadata.
    if train_audio:
        parts += [record.get('audio_description','')]
        speech = ' '.join(s['text'] for s in record.get('speech',[]))
        if speech:
            parts += ['Audible speech: '+speech]
    return ', '.join(x.strip().rstrip('.,') for x in parts if isinstance(x,str) and x.strip())
