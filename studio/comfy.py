import json
import urllib.parse
import urllib.request
import uuid

def request(base,path,payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base.rstrip('/')+path,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=30) as response:
        return json.load(response)

def submit_workflow(url,workflow_path):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost','::1'):
        raise ValueError('This bridge supports local ComfyUI only.')
    if not workflow_path:
        raise ValueError('Upload an API-format workflow JSON.')
    from pathlib import Path
    prompt = json.loads(Path(workflow_path).read_text())
    if 'nodes' in prompt or 'links' in prompt:
        raise ValueError('Export ComfyUI API format, not the visual editor graph.')
    info = request(url,'/object_info')
    if not prompt or not isinstance(prompt,dict):
        raise ValueError('Workflow must be a nonempty node dictionary.')
    for id,node in prompt.items():
        kind = node.get('class_type')
        if kind not in info:
            raise ValueError(f'ComfyUI node {kind} is not installed. Install the matching custom nodes in that ComfyUI runtime.')
        inputs = node.get('inputs',{})
        required = info[kind].get('input',{}).get('required',{})
        missing = set(required)-set(inputs)
        if missing:
            raise ValueError(f'Node {id} is missing inputs: {sorted(missing)}')
    result = request(url,'/prompt',{'prompt':prompt,'client_id':str(uuid.uuid4())})
    if result.get('node_errors'):
        raise ValueError(json.dumps(result['node_errors']))
    return {'status':'Submitted to ComfyUI; completion must be checked in ComfyUI history','prompt_id':result['prompt_id'],'history_url':url.rstrip('/')+'/history/'+result['prompt_id']}
