import argparse
import json

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='action',required=True)
    download = sub.add_parser('download')
    download.add_argument('--model',required=True)
    sub.add_parser('hardware')
    sub.add_parser('download-request')
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--video',required=True)
    prepare.add_argument('--model',default='wan22-ti2v-5b')
    prepare.add_argument('--trigger',default='motionstyle')
    prepare.add_argument('--instructions',default='')
    prepare.add_argument('--reference')
    prepare.add_argument('--audio',action='store_true')
    prepare.add_argument('--train',action='store_true')
    a = p.parse_args()
    if a.action in ('download','download-request'):
        from .assets import ensure_base
        from pathlib import Path
        options = json.loads(Path('model-request.json').read_text()) if a.action=='download-request' else {'model':a.model,'component':'model'}
        if options.get('component')=='sources':
            from .assets import download_sources
            download_sources(force=True)
        elif options.get('component','model')=='model':
            if options['model'].startswith('ltx'):
                from .ltx_assets import default_asset,asset,download_asset
                chosen = asset(options['ltx_asset']) if options.get('ltx_asset') else default_asset(options['model'])
                if not chosen or chosen['family']!=options['model'] or chosen['role']!='checkpoint':
                    raise ValueError('Select a checkpoint from the requested LTX family.')
                print(json.dumps(download_asset(chosen['id'],True),indent=2))
            else:
                print(ensure_base(options['model']))
        else:
            raise ValueError('Unknown download component.')
    elif a.action=='hardware':
        from .engine import hardware
        print(json.dumps(hardware(),indent=2))
    else:
        import time
        from . import engine
        project = engine.create(a.video,a.model,a.trigger,'motion',a.instructions,a.reference,a.audio)
        engine.dispatch(project['id'],'prepare',a.train)
        engine.ACTIVE[project['id']]['thread'].join()
        p = engine.load(project['id'])
        print(json.dumps(p,indent=2))
        if p['status'] in ('failed','cancelled'):
            raise SystemExit(1)

if __name__=='__main__':
    main()
