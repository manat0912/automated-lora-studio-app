"""Load real prequantized LTX/Gemma weights; used by the explicit source port."""
import base64
import json
import struct
from pathlib import Path

def metadata(path):
    with Path(path).open('rb') as file:
        length = struct.unpack('<Q',file.read(8))[0]
        if length>64*1024**2:
            raise ValueError('Safetensors header is unexpectedly large.')
        return json.loads(file.read(length)).get('__metadata__',{})

def is_quanto(path):
    path = Path(path)
    return path.is_file() and path.suffix=='.safetensors' and metadata(path).get('quantization_format')=='quanto'

def load_transformer(path,device,dtype):
    import torch
    from accelerate import init_empty_weights
    from mmgp import offload
    from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
    meta = metadata(path)
    config = json.loads(base64.b64decode(meta['config_base64']))
    with init_empty_weights():
        model = LTXModelConfigurator.from_metadata({'config':config})
    offload.load_model_data(model,str(path),default_dtype=dtype,writable_tensors=False,ignore_missing_keys=False,ignore_unused_weights=False)
    if hasattr(model,'_real_state_dict'):
        model.state_dict = model._real_state_dict
    return model.to(device)

def load_gemma(root,device,dtype):
    import torch
    from accelerate import init_empty_weights
    from mmgp import offload
    from ltx_core.text_encoders.gemma import GemmaTextEncoderConfigurator,get_gemma_ops
    root = Path(root)
    bundle = json.loads((root/'studio-local-ltx.json').read_text())
    source = Path(bundle['gemma_weights'])
    if not is_quanto(source):
        raise ValueError('The local Gemma file is no longer a valid Quanto checkpoint.')
    ops,module_ops = get_gemma_ops(str(root))
    with init_empty_weights():
        model = GemmaTextEncoderConfigurator.with_gemma_model_path(str(root)).from_metadata({})
    for op in module_ops:
        if op.matcher(model):
            model = op.mutator(model)
    # Existing Gemma is saved as an HF conditional-generation model. Normalize
    # its 4.x names to the official 5.8+ wrapper tree, including quantization keys.
    def key(name):
        if name.startswith('model.vision_tower.vision_model.'):
            return name.replace('model.vision_tower.vision_model.','model.vision_tower.',1)
        if name.startswith('model.language_model.'):
            return name
        if name.startswith('model.multi_modal_projector.'):
            return name
        if name.startswith('lm_head.'):
            return name
        raise ValueError('Unrecognized local Gemma weight layout: '+name)
    def postprocess(state,quantization_map):
        return {key(k):v for k,v in state.items()},{key(k):v for k,v in quantization_map.items()}
    offload.load_model_data(model.model,str(source),postprocess_sd=postprocess,default_dtype=dtype,writable_tensors=False,ignore_missing_keys=False,ignore_unused_weights=False)
    if hasattr(model.model,'_real_state_dict'):
        model.model.state_dict = model.model._real_state_dict
    return model.to(device)

def embedding_paths(root):
    path = Path(root)/'studio-local-ltx.json'
    if not path.exists():
        return None
    bundle = json.loads(path.read_text())
    return tuple(bundle['embedding_weights'])
