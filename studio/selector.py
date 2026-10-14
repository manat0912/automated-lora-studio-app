"""AI ranks candidate clips; model geometry and scene validation remain deterministic."""
import gc
import json
from pathlib import Path

def select(source,starts,seconds,instructions,device,cancel,log):
    import torch
    import cv2
    from PIL import Image
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from .assets import download
    from .catalog import VISION_REPO
    location = download(VISION_REPO,['*.json','*.safetensors','*.txt','*.model','*.jinja'],log)
    processor = AutoProcessor.from_pretrained(location,local_files_only=True,min_pixels=128*28*28,max_pixels=256*28*28)
    dtype = torch.float16 if device.startswith('cuda') else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(location,dtype=dtype,attn_implementation='sdpa',local_files_only=True).to(device).eval()
    cap = cv2.VideoCapture(str(source))
    decisions = []
    try:
        for index,start in enumerate(starts):
            if cancel.is_set():
                raise InterruptedError('Cancelled during AI motion selection.')
            images = []
            for fraction in (0,.33,.66,.95):
                cap.set(cv2.CAP_PROP_POS_MSEC,(start+seconds*fraction)*1000)
                ok,frame = cap.read()
                if ok:
                    image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    image.thumbnail((384,384))
                    images.append(image)
            if len(images)<4:
                decisions.append({'start':start,'keep':False,'reason':'Could not sample all clip frames'})
                continue
            prompt = f'''Four chronological frames from a {seconds:.2f}s clip. Decide whether this clip provides a visible, usable example of motion relevant to: {instructions or 'the clearly visible action'}. Reject title cards, unrelated footage, and stationary empty shots. Sparse frames cannot establish exact skeletal tracks or rhythm. Return a JSON object with keys keep (boolean), reason (specific sentence explaining the visible evidence), motion (specific visible movement), people_count (integer or null). Never echo placeholders or an example schema.'''
            content = [{'type':'image','image':im} for im in images]+[{'type':'text','text':prompt}]
            text = processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
            inputs = processor(text=[text],images=images,return_tensors='pt',padding=True).to(device)
            with torch.inference_mode():
                result = model.generate(**inputs,max_new_tokens=160,do_sample=False)
            decoded = processor.batch_decode(result[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
            try:
                decision = json.loads(decoded[decoded.index('{'):decoded.rindex('}')+1])
                if not isinstance(decision.get('keep'),bool):
                    raise ValueError('Invalid keep field')
            except (ValueError,json.JSONDecodeError):
                # Invalid AI output must not silently delete useful training footage.
                decision = {'keep':True,'reason':'AI selection response could not be parsed; kept for caption review','motion':''}
            decision['start'] = start
            decisions.append(decision)
            log(f'AI candidate {index+1}/{len(starts)}: {"keep" if decision["keep"] else "skip"} · {decision["reason"]}')
    finally:
        cap.release()
        del model,processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return decisions
