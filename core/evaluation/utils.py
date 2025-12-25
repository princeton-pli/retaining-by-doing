import json
import yaml
import unicodedata
from pathlib import Path
from rich import print
from rich.markup import escape
from rich.console import Console
from transformers import AutoTokenizer
from core.utils import load_module

console = Console(markup=False)


def setup_model_for_inference(
    model_name,
    model_module_path='transformers.AutoModelForCausalLM',
    tokenizer_only=False,
):
    model_module = load_module(model_module_path)
    if tokenizer_only:
        model = None
    else:
        model = model_module.from_pretrained(model_name, device_map='auto', local_files_only=True)
 
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side='left')
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class GenerationManager:
    def __init__(self, run_dir, print_to_stdout=True, overwrite=False, dry_run=False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.print_to_stdout = print_to_stdout
        self.dry_run = dry_run

        if self.dry_run:
            self.print_to_stdout = True
            self._log_writer = None
            self._prediction_writer = None
            self.seen_datapoints = set()
        else:
            self.log_path = self.run_dir / 'log.txt'
            self.predictions_path = self.run_dir / 'predictions.jsonl'
            if overwrite:
                self._log_writer = open(self.log_path, 'w')
                self._prediction_writer = open(self.predictions_path, 'w')
                self.seen_datapoints = set()
            elif self.predictions_path.exists():
                with open(self.predictions_path, 'r') as f:
                    self.seen_datapoints = {json.loads(line)['datapoint_idx'] for line in f}
                self._log_writer = open(self.log_path, 'a')
                self._prediction_writer = open(self.predictions_path, 'a')
            else:
                self._log_writer = open(self.log_path, 'w')
                self._prediction_writer = open(self.predictions_path, 'w')
                self.seen_datapoints = set()

    def __del__(self):
        if self._log_writer is not None:
            self._log_writer.close()
        if self._prediction_writer is not None:
            self._prediction_writer.close()

    def _normalize_text(self, text):
        """Normalize text to ensure it's safe for ASCII output."""
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)  # Keep Unicode if needed
        return unicodedata.normalize('NFKD', text).encode('ascii', 'replace').decode()
 
    def write_log(self, text):
        text = self._normalize_text(text)

        if not self.dry_run:
            self._log_writer.write(text + '\n')
            self._log_writer.flush()
        if self.print_to_stdout:
            console.print(text)
        
    def write_prediction(self, prediction):
        prediction_text = json.dumps(prediction)
        if not self.dry_run:
            self._prediction_writer.write(prediction_text + '\n')
            self._prediction_writer.flush()

    def save_generation_config(self, generation_config):
        if not self.dry_run:
            with open(self.run_dir / 'generation_config.yaml', 'w') as f:
                yaml.dump(generation_config, f)
    
    def save_metrics(self, metrics):
        if not self.dry_run:
            with open(self.run_dir / 'eval_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
