"""Bounded real LTX training benchmark; source datasets/checkpoints are read-only."""
import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def benchmark(project,profile,steps=3):
    if not re.fullmatch(r'[a-f0-9]{12}',project) or not 3 <= steps <= 32:
        raise ValueError('Invalid project ID.')
    removed = []
    if profile == 'native':
        if os.name != 'nt':
            raise ValueError('Native comparison is specifically for the Windows AMD runtime.')
        from studio.runtime_env import ltx_environment
        hardware = json.loads((ROOT/'hardware.json').read_text())
        if hardware.get('selected_gpu') != 'amd':
            raise ValueError('Native comparison is specifically for Windows AMD.')
        _,removed = ltx_environment(os.environ,hardware['selected_gpu'],sys.platform)
        for name in removed:
            os.environ.pop(name)
    # Configure the environment before importing torch or loading a runtime.
    import torch
    import yaml
    from ltx_trainer.config import LtxTrainerConfig
    from ltx_trainer.trainer import LtxvTrainer
    from torch.utils.data import DataLoader
    source = ROOT/'jobs'/project/'ltx-train.yaml'
    config = yaml.safe_load(source.read_text())
    if not Path(config['data']['preprocessed_data_root']).is_dir():
        raise ValueError('Benchmark needs the project\'s already-preprocessed dataset.')
    output = ROOT/'validation/performance'/f'{int(time.time())}-{project}-{profile}'
    output.mkdir(parents=True)
    config['output_dir'] = str(output)
    config['model']['load_checkpoint'] = None
    config['optimization']['steps'] = steps
    config['checkpoints']['interval'] = steps
    config['validation']['samples'] = []
    config['validation']['interval'] = None
    config['validation']['skip_initial_validation'] = True
    config['hub']['push_to_hub'] = False
    config['wandb']['enabled'] = False
    phases = {}
    report = {'profile':profile,'source_project':project,'source_modified':False,
        'output':str(output),'benchmark_steps':steps,'removed_environment_keys':removed,'status':'loading',
        'torch':torch.__version__,'phases':phases}
    destination = output/'report.json'
    def persist():
        destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    def timed(label,fn):
        def wrapper(*args,**kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            value = fn(*args,**kwargs)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            phases.setdefault(label,[]).append(round(seconds,4))
            print(f'BENCHMARK_PHASE {label} {seconds:.4f}s',flush=True)
            persist()
            return value
        return wrapper
    persist()
    print('BENCHMARK_LOADING '+str(destination),flush=True)
    start = time.perf_counter()
    trainer = LtxvTrainer(LtxTrainerConfig(**config))
    torch.cuda.synchronize()
    report['load_seconds'] = time.perf_counter()-start
    trainer._training_step = timed('forward',trainer._training_step)
    trainer._accelerator.backward = timed('backward',trainer._accelerator.backward)
    trainer._accelerator.clip_grad_norm_ = timed('clip_grad',trainer._accelerator.clip_grad_norm_)
    trainer._save_checkpoint = timed('checkpoint',trainer._save_checkpoint)
    original_init = trainer._init_optimizer
    def init_optimizer():
        original_init()
        trainer._optimizer.step = timed('optimizer',trainer._optimizer.step)
    trainer._init_optimizer = init_optimizer
    original_iter = DataLoader.__iter__
    class TimedIterator:
        def __init__(self,iterator): self.iterator = iterator
        def __iter__(self): return self
        def __next__(self): return timed('data_loading',lambda:next(self.iterator))()
    DataLoader.__iter__ = lambda self:TimedIterator(original_iter(self))
    report['status'] = 'training'
    persist()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        _, stats = trainer.train(disable_progress_bars=True)
        torch.cuda.synchronize()
        report.update(status='passed',wall_seconds=time.perf_counter()-start,
            stats=stats.model_dump(),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            phase_medians={key:statistics.median(values) for key,values in phases.items() if values})
        report['projected_2000_hours'] = stats.total_time_seconds/steps*2000/3600
        steady = [sum(parts) for parts in zip(*(phases.get(name,[]) for name in ('forward','backward','clip_grad','optimizer'))) ][2:]
        if steady:
            report['steady_seconds_per_step'] = statistics.mean(steady)
            report['steady_projected_2000_hours'] = statistics.mean(steady)*2000/3600
        persist()
        print('BENCHMARK_COMPLETE '+json.dumps(report),flush=True)
    except Exception as exc:
        report.update(status='failed',error=str(exc))
        persist()
        raise
    finally:
        DataLoader.__iter__ = original_iter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',required=True)
    parser.add_argument('--profile',choices=['inherited','native'],default='inherited')
    parser.add_argument('--steps',type=int,default=3)
    args = parser.parse_args()
    benchmark(args.project,args.profile,args.steps)
