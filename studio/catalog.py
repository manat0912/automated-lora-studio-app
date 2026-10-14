from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Model:
    key: str
    label: str
    repo: str
    kind: str
    stride: int
    spatial: int
    fps: int
    min_vram: int
    arch: str
    native: bool = True
    audio: bool = False

MODELS = {
    m.key: m for m in [
        Model('wan21-t2v-1.3b', 'Wan 2.1 · 1.3B · video motion', 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers', 't2v', 4, 16, 16, 12, 'wan21'),
        Model('wan22-ti2v-5b', 'Wan 2.2 · 5B · image to video', 'Wan-AI/Wan2.2-TI2V-5B-Diffusers', 'ti2v', 4, 32, 24, 24, 'wan22_5b'),
        Model('wan21-i2v-14b', 'Wan 2.1 · 14B · image to video', 'Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', 'i2v', 4, 16, 16, 48, 'wan21_i2v'),
        Model('wan22-i2v-14b', 'Wan 2.2 · A14B · image to video · AI Toolkit', 'Wan-AI/Wan2.2-I2V-A14B-Diffusers', 'i2v', 4, 16, 16, 48, 'wan22_14b_i2v', False),
        Model('ltx2', 'LTX-2 · audio/video LoRA · isolated HIP/CUDA trainer', 'Lightricks/LTX-2', 'i2v', 8, 32, 24, 31, 'ltx2', False, True),
        Model('ltx23', 'LTX-2.3 · includes distilled v1.1 and quantized variants', 'Lightricks/LTX-2.3', 'i2v', 8, 32, 24, 31, 'ltx23', False, True),
        Model('ltx25', 'LTX-2.5 · audio/video LoRA · matching Gemma 4', 'Lightricks/LTX-2.5', 'i2v', 8, 32, 24, 31, 'ltx25', False, True),
        Model('ltxv', 'LTX-Video · legacy 2B/13B dev, distilled, FP8 and GGUF', 'Lightricks/LTX-Video', 'i2v', 8, 32, 24, 32, 'ltxv', False, False),
    ]
}
VISION_REPO = 'Qwen/Qwen2.5-VL-3B-Instruct'
SPEECH_REPO = 'Systran/faster-whisper-small'
AUDIO_REPO = 'Qwen/Qwen2-Audio-7B-Instruct'

# Wrappers, tutorials, and trainers remain separate projects, with their licenses.
INTEGRATIONS = [
    ('AI Video Clipper Pinokio', 'manat0912/AI-Video-Clipper-LoRA-Pinokio', 'Pinokio wrapper reference; native split/caption pipeline; exported .mp4/.txt dataset interoperability'),
    ('AI Toolkit Pinokio', 'pinokiofactory/ai-toolkit', 'Installer reference; isolated upstream CLI adapter'),
    ('AI Toolkit', 'ostris/ai-toolkit', 'Optional Wan A14B trainer adapter; isolated runtime. LTX training remains unvalidated'),
    ('OneTrainer Pinokio', '6Morpheus6/Onetrainer', 'Wrapper for Nerogar/OneTrainer; source library only'),
    ('LoRA Training Toolkit', 'miroleon/lora-training-toolkit', 'Learning material; source library only'),
    ('Realtime LoRA', 'shootthesound/comfyUI-Realtime-Lora', 'ComfyUI custom nodes; execute user-supplied API workflow through ComfyUI bridge'),
    ('Stable Diffusion guides', 'FurkanGozukara/Stable-Diffusion', 'Tutorial/reference collection; no unified trainer API'),
    ('OmniChar', 'omnichar/OmniChar', 'Character format reference; image reference handoff, no .char encoder implemented'),
    ('Fizgig', 'shootthesound/Fizgig', 'Independent trainer; source library only'),
    ('LoRA Scripts', 'Akegarasu/lora-scripts', 'Independent image trainer; source library only'),
    ('LoRA Studio', 'michaelringholm/lora-studio', 'Independent image trainer; source library only'),
    ('FlyMyAI trainer', 'FlyMyAI/flymyai-lora-trainer', 'Qwen Image trainer; source library only'),
    ('Newbie trainer', 'NewBieAI-Lab/NewbieLoraTrainer', 'Independent trainer; source library only'),
    ('LTX-2 official', 'Lightricks/LTX-2', 'Pinned isolated trainer source port: Windows PCM audio, local Quanto loading; actual runtime/model tests recorded separately'),
    ('ComfyUI LTX2 Trainer', 'jaimitoes/ComfyUI-LTX2-TRAINER', 'Custom nodes; execute user-supplied API workflow through ComfyUI bridge'),
]

def model_dict(key):
    return asdict(MODELS[key])
