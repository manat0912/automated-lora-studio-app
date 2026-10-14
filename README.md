# Automated LoRA Studio app source

Local Gradio studio for motion dataset preparation, captions, speech analysis, native Wan LoRA training and an isolated LTX audio/video trainer.

Install through the [Pinokio launcher](https://github.com/manat0912/automated-lora-studio), which clones this repository into local `app/`, scans hardware, installs vendor dependencies, obtains Hugging Face access when needed and downloads models locally. This repository contains source and catalog metadata only. Models, media, environments, jobs, credentials and settings are ignored by Git.

Windows/Linux x64 NVIDIA CUDA and supported AMD ROCm use actual GPU checks. Apple Silicon macOS prepares datasets on CPU. Intel macOS and Windows/Linux ARM64 are unavailable with the pinned official wheels. See the launcher's [compatibility matrix](https://github.com/manat0912/automated-lora-studio/blob/main/COMPATIBILITY.md) and [usage/API documentation](https://github.com/manat0912/automated-lora-studio#api).

For development, use the launcher-managed Python 3.12 environment. `python validate.py` runs regression tests; `setup.py verify` checks imports, FFmpeg and GPU forward/backward. Installer tests run without downloading weights. Optional LTX tests require its separate backend and matching model components.

Use the actual URL displayed by Pinokio. **View API** provides the live contract:

```python
from gradio_client import Client
hardware, compatibility = Client('http://127.0.0.1:7860').predict(api_name='/hardware')
```

```javascript
import { Client } from '@gradio/client';
const client = await Client.connect('http://127.0.0.1:7860');
console.log((await client.predict('/hardware', [])).data);
```

```sh
curl -X POST http://127.0.0.1:7860/gradio_api/call/hardware -H 'Content-Type: application/json' -d '{"data":[]}'
curl -N http://127.0.0.1:7860/gradio_api/call/hardware/EVENT_ID
```

`/create_project`, `/project_status`, `/train_project`, `/cancel_project`, `/generate_project` and `/export_project` operate on local projects. The launcher README provides payload examples. MIT licensing covers this source; upstream libraries, trainers and models retain their licenses/access conditions.
