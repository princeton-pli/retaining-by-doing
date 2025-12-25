import yaml
import json
import copy
import time
import importlib

from addict import Dict
from pathlib import Path
from rich import print
from rich.markup import escape

import torch
import torch.distributed as dist
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
)


def print_rank0(msg):
    if dist.get_rank() == 0:
        print(msg)


def timer(name=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            timer_name = name if name else func.__name__
            print(f'func={timer_name}: {end_time - start_time:.2f} seconds')
            return result
        return wrapper
    return decorator


def print_gpu_usage():
    nvmlInit()
    n_gpus = torch.cuda.device_count()

    print('========== GPU Utilization ==========')
    for gpu_id in range(n_gpus):
        h = nvmlDeviceGetHandleByIndex(gpu_id)
        info = nvmlDeviceGetMemoryInfo(h)
        print(f'GPU {gpu_id}')
        print(f'- Used:       {info.used / 1024 ** 3:>8.2f} B ({info.used / info.total * 100:.1f}%)')
        print(f'- Available:  {info.free / 1024 ** 3:>8.2f} B ({info.free / info.total * 100:.1f}%)')
        print(f'- Total:      {info.total / 1024 ** 3:>8.2f} B')
    print('=====================================')


def load_module(module_path, module_name=None):
    """
    module_path: str
    """
    if module_name is None:
        module_path, module_name = module_path.rsplit('.', 1)
    
    module = importlib.import_module(module_path)
    return getattr(module, module_name)


class ExperimentManager:
    def __init__(self, run_dir, print_to_stdout=True, world_size=None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = self.run_dir / 'predictions'
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        self._writer = None
        self._prediction_writer = None
        self.print_to_stdout = print_to_stdout
        self.world_size = world_size

    def __del__(self):
        if self._writer is not None:
            self._writer.close()
        if self._prediction_writer is not None:
            for writer in self._prediction_writer.values():
                writer.close()
    
    def write(self, text):
        if self._writer is None:
            self._writer = open(self.run_dir / 'log.txt', 'w')
        self._writer.write(text)
        self._writer.write('\n')
        self._writer.flush()
        if self.print_to_stdout:
            print(escape(text))
    
    def save_config(self, config):
        with open(self.run_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)
    
    def save_metrics(self, metrics, save_dir=None):
        if save_dir is not None:
            save_dir = Path(save_dir)
        else:
            save_dir = self.run_dir / 'final'
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / 'eval_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

    def write_prediction(self, prediction):
        prediction_text = json.dumps(prediction)
        local_rank = prediction.get('local_rank', 0)

        if self._prediction_writer is None:
            if self.world_size is None:
                self._prediction_writer = {0: open(self.predictions_dir / 'predictions.jsonl', 'w')}
            else:
                self._prediction_writer = {i: open(self.predictions_dir / f'predictions_rank={i}.jsonl', 'w') for i in range(self.world_size)}
        self._prediction_writer[local_rank].write(prediction_text + '\n')
        self._prediction_writer[local_rank].flush()
