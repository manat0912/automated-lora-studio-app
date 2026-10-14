import json
import os
import shutil
from pathlib import Path
import gradio as gr
from studio import engine
from studio.catalog import MODELS, INTEGRATIONS
from studio.assets import ensure_base

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
ROOT = Path(__file__).resolve().parent

CSS = '''
body { background: #0c1715; }
.gradio-container { max-width: 1460px !important; }
#hero { padding: 24px 30px; border: 1px solid #35453b; border-radius: 18px; background: linear-gradient(120deg,#17352d,#1c2824); margin-bottom: 18px; }
#hero h1 { font-size: 34px; letter-spacing: -1.2px; margin: 8px 0; color:#eef4df !important; }
.gradio-container .prose h1, .gradio-container .prose h2, .gradio-container .prose h3 { color:#eef4df !important; }
#hero p { color: #bdc9bf; }
#hero .eyebrow { color:#c0e68b; text-transform:uppercase; letter-spacing: 3px; font-size: 11px; }
#workspace { border: 1px solid #35453b; border-radius: 16px; padding: 18px; }
.primary { background:#b8e678 !important; color:#142016 !important; font-weight:700 !important; }
#progress { border-left: 3px solid #b8e678; padding-left: 16px; }
footer { display:none !important; }
'''

def safe(fn):
    def wrapped(*args,**kwargs):
        try:
            return fn(*args,**kwargs)
        except (ValueError,RuntimeError,FileNotFoundError) as e:
            raise gr.Error(str(e)) from e
    return wrapped

def make_project(video,key,trigger,purpose,instructions,reference,audio,steps,rank,variant=None):
    if not video:
        raise ValueError('Upload a source video first.')
    p = engine.create(video,key,trigger,purpose,instructions,reference,audio)
    if key.startswith('ltx'):
        from studio.ltx_assets import asset,default_asset
        selected = asset(variant) if variant else default_asset(key)
        if not selected or selected['family']!=key or selected['role']!='checkpoint':
            raise ValueError('Choose a checkpoint variant belonging to the selected LTX family.')
        p['plan']['variant'] = selected
    p['plan']['steps'] = int(steps)
    p['plan']['rank'] = int(rank)
    engine.save(p)
    return p

@safe
def api_create(video,key,trigger,purpose,instructions,reference,audio,steps,rank,variant,auto_train):
    p = make_project(video,key,trigger,purpose,instructions,reference,audio,steps,rank,variant)
    engine.dispatch(p['id'],'prepare',auto_train)
    return {'project_id':p['id'],'status':'queued','plan':p['plan']}

@safe
def begin(video,key,trigger,purpose,instructions,reference,audio,steps,rank,variant,auto):
    p = make_project(video,key,trigger,purpose,instructions,reference,audio,steps,rank,variant)
    message = engine.dispatch(p['id'],'prepare',bool(auto))
    return p['id'],message,p['plan'],project_choices(p['id'])

def project_choices(selected=None):
    options = {'choices':[(f'{p["name"]} · {p["status"]} · {p["id"]}',p['id']) for p in engine.list_projects()]}
    if selected is not None:
        options['value'] = selected
    return gr.update(**options)

@safe
def retry_current(id):
    result = engine.retry_smaller(id)
    return result['project_id'],'Smaller LTX retry queued; original project preserved.',result['plan'],project_choices(result['project_id'])

@safe
def refresh(id):
    if not id:
        return 'Upload a video, choose a model, and begin.',{},[], '',gr.update(choices=[]),None,gr.update(choices=[])
    p,logs = engine.status(id)
    status = f'### {p["status"].replace("_"," ").title()}\nProject `{id}`'
    if p.get('error'):
        status += '\n\n'+p['error']
    rows = [[c['index'],Path(c['path']).name,c['media']['frames'],c['media']['fps'],c.get('caption_status','Pending')] for c in p['clips']]
    return status,p['plan'],rows,logs,gr.update(choices=[(Path(c['path']).name,c['index']) for c in p['clips']]),p.get('preview'),gr.update(choices=p['checkpoints'],value=p['checkpoints'][-1] if p['checkpoints'] else None)

@safe
def select_clip(id,index):
    if index is None or not id:
        return None,'',{}
    clip = engine.load(id)['clips'][int(index)]
    return clip['path'],clip.get('caption',''),{k:v for k,v in clip.items() if k not in ('path','caption')}

def compatibility():
    h = engine.hardware()
    lock_path = ROOT/'models/downloads.json'
    downloads = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    rows = []
    for key,m in MODELS.items():
        supported = bool(h.get('gpu_verified') and h.get('vram_gb',0)>=m.min_vram)
        runtime = 'Native HIP/CUDA' if m.native else 'Separate AI Toolkit (NVIDIA/Linux)'
        if m.arch.startswith('ltx'):
            runtime = 'Isolated LTX HIP/CUDA'
            installed = (ROOT/'backends/ltx-official/backend-ready.json').exists()
            status = 'LTX backend installed; exact variant validation is recorded separately' if installed else 'Install LTX backend from Pinokio menu'
            if h.get('vram_gb',0)<31:
                status = 'Current 22B training preset needs a nominal 32GiB GPU'
            if key=='ltxv':
                status = 'Legacy dataset / compatible runner handoff; separate trainer required'
        elif not m.native and not (ROOT/'backends/ai-toolkit/backend-ready.json').exists():
            status = 'Optional backend not installed'
        elif m.native and not downloads.get(m.repo,{}).get('complete'):
            status = 'Base weights pending download'
        else:
            status = 'Capacity check passed; real training still needs validation' if supported else f'Needs {m.min_vram}GiB verified VRAM'
        report_file = ROOT/'validation'/('ltx-pipeline-report.json' if key=='ltx23' else 'full-model-report.json' if key=='wan22-ti2v-5b' else 'no-report')
        if report_file.exists() and json.loads(report_file.read_text()).get('status')=='passed':
            status = 'Training + I2V smoke passed on this PC (small synthetic media)'
        rows.append([m.label,runtime,f'{m.stride}n+1',m.spatial,m.fps,'Audio + video' if m.audio else 'Video only',status])
    return h,rows

@safe
def download_model(key):
    with engine.GPU_LOCK:
        path = ensure_base(key)
    return f'Model files verified at {path}'

def model_variants(key):
    from studio.ltx_assets import choices,default_asset
    if not key.startswith('ltx'):
        return gr.update(choices=[],value=None,visible=False),'Native training uses this exact Wan base variant.' if MODELS[key].native else 'This model uses the separate AI Toolkit backend.'
    selected = default_asset(key)
    return gr.update(choices=choices(key,True),value=selected['id'] if selected else None,visible=True),'LTX-2/2.3/2.5 training uses the isolated HIP/CUDA backend. Select BF16/FP16 or a complete local INT8 Quanto bundle for training. Other quantized formats remain available for compatible inference loaders. Exact version and media tests are recorded separately.'

def ltx_table():
    from studio.ltx_assets import catalog
    return [[a['family'],a['name'],a['role'],a['quantization'],a['size_gib'],'Local bundle' if a.get('local') else 'Official' if a['official'] else 'Community',a['backend_requirement']] for a in catalog()['assets']]

@safe
def ltx_refresh():
    from studio.ltx_assets import refresh,choices
    result = refresh()
    return ltx_table(),gr.update(choices=choices()),result

@safe
def ltx_download(id,components):
    from studio.ltx_assets import download_asset
    return download_asset(id,components)

def build_ui():
    h = engine.hardware()
    initial = h.get('model','wan22-ti2v-5b')
    if (ROOT/'backends/ltx-official/backend-ready.json').exists():
        initial = 'ltx23'
    theme = gr.themes.Soft(primary_hue='lime',secondary_hue='emerald',neutral_hue='slate',font=['Inter','ui-sans-serif','sans-serif']).set(body_background_fill='#0c1715',body_background_fill_dark='#0c1715')
    with gr.Blocks(title='Automated LoRA Studio',theme=theme,css=CSS,js="() => { document.body.classList.add('dark'); }") as app:
        gr.HTML('<div id="hero"><div class="eyebrow">AUTOMATED LORA STUDIO / LOCAL WORKSPACE</div><h1>Teach the motion. Bring your own characters.</h1><p>Source video → model-aware clips → AI captions → compatible LoRA → reference-image preview</p></div>')
        state = gr.State('')
        with gr.Tabs():
            with gr.Tab('Create & train'):
                with gr.Row():
                    with gr.Column(scale=4,elem_id='workspace'):
                        gr.Markdown('### 01 · Choose your model and source')
                        model = gr.Dropdown([(m.label,k) for k,m in MODELS.items()],value=initial,label='Target generation model')
                        from studio.ltx_assets import choices,default_asset,catalog
                        selected_ltx = default_asset(initial) if initial.startswith('ltx') else None
                        variant = gr.Dropdown(choices=choices(initial,True) if selected_ltx else [],value=selected_ltx['id'] if selected_ltx else None,label='LTX checkpoint / quantized variant',visible=initial.startswith('ltx'))
                        variant_info = gr.Markdown()
                        model.change(model_variants,model,[variant,variant_info])
                        video = gr.File(file_types=['.mp4','.mov','.mkv','.webm','.avi'],type='filepath',label='Source recording')
                        with gr.Row():
                            trigger = gr.Textbox(value='motionstyle',label='Motion trigger',placeholder='e.g. hkchoreo')
                            purpose = gr.Dropdown([('Transfer motion to new characters','motion'),('Learn appearance and motion','character_motion')],value='motion',label='What should this LoRA learn?')
                        instructions = gr.Textbox(label='Movement instructions',lines=3,placeholder='e.g. A group moves in unison, with rhythmic stomps and chest slaps. Describe relationships and timing.')
                        reference = gr.Image(type='filepath',label='Characters for an I2V preview (optional)')
                        audio = gr.Checkbox(value=h.get('audio_semantics',False),label='Add AI audio descriptions (speech timing is included)')
                        with gr.Accordion('Training controls',open=False):
                            steps = gr.Slider(1,8000,value=2000,step=1,label='Training steps')
                            rank = gr.Dropdown([8,16,32],value=8 if initial.startswith('ltx') else 16,label='LoRA rank')
                            gr.Markdown('Automatic planning uses conservative hardware heuristics. Exact memory use depends on your footage and model.')
                        with gr.Row():
                            prepare = gr.Button('Prepare dataset',variant='secondary')
                            auto = gr.Button('Prepare + train automatically',variant='primary',elem_classes='primary')
                        notice = gr.Markdown()
                        gr.Markdown('For motion transfer, captions describe roles and movement generically. Diverse source performers help reduce appearance memorization. Review generated samples to check the result.')
                    with gr.Column(scale=5):
                        gr.Markdown('### 02 · Your project',elem_id='progress')
                        projects = gr.Dropdown(label='Open a saved project',choices=[(f'{p["name"]} · {p["id"]}',p['id']) for p in engine.list_projects()])
                        refresh_projects = gr.Button('Refresh project list',size='sm')
                        status = gr.Markdown('Upload a video and begin.')
                        with gr.Row():
                            train = gr.Button('Train / resume')
                            cancel = gr.Button('Cancel job')
                            export = gr.Button('Export dataset & LoRA')
                        smaller = gr.Button('Retry with smaller LTX clips')
                        artifact = gr.File(label='Export package')
                        clips = gr.Dataframe(headers=['Clip','File','Frames','FPS','Caption status'],datatype=['number','str','number','number','str'],interactive=False,label='Dataset audit')
                        with gr.Accordion('Automatic plan',open=False):
                            plan_view = gr.JSON(label='Plan and compatibility notes')
                        logs = gr.Textbox(label='Live job log',lines=12,max_lines=18,interactive=False)
                timer = gr.Timer(3)
                inputs = [video,model,trigger,purpose,instructions,reference,audio,steps,rank,variant]
                model.change(lambda key:gr.update(value=8 if key.startswith('ltx') else 16),model,rank,api_name=False)
                prepare.click(lambda *a:begin(*a,False),inputs,[state,notice,plan_view,projects],api_name='prepare')
                auto.click(lambda *a:begin(*a,True),inputs,[state,notice,plan_view,projects],api_name='prepare_and_train')
                projects.change(lambda id:id or '',projects,state)
                refresh_projects.click(project_choices,outputs=projects,api_name=False)
                train.click(safe(lambda id:engine.dispatch(id,'train')),state,notice,api_name='train')
                cancel.click(safe(engine.cancel_job),state,notice,api_name='cancel')
                export.click(safe(engine.export),state,artifact,api_name='export')
                smaller.click(retry_current,state,[state,notice,plan_view,projects],api_name=False)
            with gr.Tab('Review captions'):
                gr.Markdown('### Inspect every clip before a longer training run\nAudio and speech timing are kept as metadata. Visual-only models receive visual captions.')
                clip_select = gr.Dropdown(label='Clip')
                with gr.Row():
                    clip_video = gr.Video(label='Verified clip')
                    with gr.Column():
                        caption = gr.Textbox(label='Training caption',lines=7)
                        caption_save = gr.Button('Save reviewed caption',variant='primary')
                        caption_notice = gr.Markdown()
                analysis = gr.JSON(label='Motion, characters, tags, speech, and audio analysis')
                clip_select.change(select_clip,[state,clip_select],[clip_video,caption,analysis])
                caption_save.click(safe(engine.edit_caption),[state,clip_select,caption],caption_notice,api_name='save_caption')
            with gr.Tab('Test with your image'):
                gr.Markdown('### Use a compatible I2V model and the characters in your image\nThe latest checkpoint is applied at strength 0.7. Compare character count, appearance, timing, and motion. A successful training loss alone does not establish transfer quality.')
                checkpoint = gr.Dropdown(label='Latest trained checkpoint',interactive=False)
                generate = gr.Button('Generate reference-image preview',variant='primary')
                generate_notice = gr.Markdown()
                preview = gr.Video(label='Reference-image motion test')
                generate.click(safe(lambda id:engine.dispatch(id,'generate')),state,generate_notice,api_name='generate')
            with gr.Tab('Hardware & models'):
                gr.Markdown('### Installation evidence and model compatibility\nThe installer scans the PC before creating the Python environment, installing packages, or downloading models. It then checks imports and runs a real GPU forward/backward operation.')
                hardware_view = gr.JSON(h,label='PC scan and runtime report')
                matrix = gr.Dataframe(value=compatibility()[1],headers=['Model','Backend','Frames','Dimension multiple','FPS','Training modality','Status'],interactive=False)
                refresh_hardware = gr.Button('Refresh compatibility report')
                refresh_hardware.click(compatibility,outputs=[hardware_view,matrix],api_name='hardware')
                model_download = gr.Dropdown([(m.label,k) for k,m in MODELS.items() if m.native],value=initial if MODELS[initial].native else 'wan22-ti2v-5b',label='Download another native base model')
                download = gr.Button('Download & verify model files')
                download_notice = gr.Markdown()
                download.click(safe(download_model),model_download,download_notice,api_name='download_model')
                gr.Markdown('### LTX versions, distilled checkpoints and quantized files\nIncludes official LTX-2, LTX-2.3 (including distilled v1.1), LTX-2.5, legacy LTX-Video and discovered GGUF files. Refresh checks the published repositories for new variants. A listed or downloaded file is not a validated training loader.')
                ltx_grid = gr.Dataframe(value=ltx_table(),headers=['Family','File','Role','Precision','GiB','Publisher','Required loader'],interactive=False)
                ltx_select = gr.Dropdown(choices=choices(),label='LTX file to download (checkpoints and components)')
                include_components = gr.Checkbox(value=True,label='Include the required matching text encoder and split VAEs where mapped')
                with gr.Row():
                    catalog_refresh = gr.Button('Refresh published LTX variants')
                    ltx_get = gr.Button('Download selected LTX asset + components')
                ltx_result = gr.JSON(label='Exact files, sizes, loader requirements and download results')
                catalog_refresh.click(ltx_refresh,outputs=[ltx_grid,ltx_select,ltx_result],api_name='ltx_catalog_refresh')
                ltx_get.click(ltx_download,[ltx_select,include_components],ltx_result,api_name='ltx_download')
            with gr.Tab('Integrations'):
                gr.Markdown('### One workspace, explicit integration status\nNative Wan and an isolated LTX source port provide automated video training. The LTX port uses real SoundFile/FFmpeg audio I/O on Windows and validates quantized LoRA gradients. AI Toolkit is an optional isolated CLI backend. ComfyUI custom nodes use the workflow bridge below. Other listed projects are a source/reference library; their independent runtimes are not installed by cloning their source.')
                gr.Dataframe(value=[[name,f'https://github.com/{repo}',role] for name,repo,role in INTEGRATIONS],headers=['Project','Upstream','Integration scope'],interactive=False)
                with gr.Accordion('ComfyUI API workflow bridge',open=False):
                    gr.Markdown('Upload an exported **API format** workflow containing your installed training or generation nodes. The bridge checks every node class and input against the running ComfyUI server before submission. Workflow paths must be valid on that server.')
                    comfy_url = gr.Textbox(value='http://127.0.0.1:8188',label='Local ComfyUI URL')
                    workflow = gr.File(file_types=['.json'],type='filepath',label='API workflow JSON')
                    submit = gr.Button('Validate nodes & submit workflow')
                    comfy_result = gr.JSON()
                    from studio.comfy import submit_workflow
                    submit.click(safe(submit_workflow),[comfy_url,workflow],comfy_result,api_name='comfy_submit')
        timer.tick(refresh,state,[status,plan_view,clips,logs,clip_select,preview,checkpoint],queue=False,api_name=False)
        # Explicit project IDs make automation independent of browser session State.
        api_id = gr.Textbox(visible=False)
        api_json = gr.JSON(visible=False)
        api_text = gr.Textbox(visible=False)
        api_auto = gr.Checkbox(value=False,visible=False)
        gr.Button(visible=False).click(api_create,inputs+[api_auto],api_json,api_name='create_project')
        gr.Button(visible=False).click(safe(lambda id:engine.status(id)[0]),api_id,api_json,api_name='project_status')
        gr.Button(visible=False).click(safe(lambda id:engine.dispatch(id,'train')),api_id,api_text,api_name='train_project')
        gr.Button(visible=False).click(safe(engine.retry_smaller),api_id,api_json,api_name='retry_smaller_project')
        gr.Button(visible=False).click(safe(engine.cancel_job),api_id,api_text,api_name='cancel_project')
        gr.Button(visible=False).click(safe(lambda id:engine.dispatch(id,'generate')),api_id,api_text,api_name='generate_project')
        gr.Button(visible=False).click(safe(engine.export),api_id,artifact,api_name='export_project')
    return app.queue(default_concurrency_limit=1)

if __name__ == '__main__':
    # Gradio chooses the next free port; no public sharing or hardcoded port.
    app = build_ui()
    app.launch(server_name='127.0.0.1',share=False,inbrowser=False,allowed_paths=[str(engine.JOBS)],show_error=True)
