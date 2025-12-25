import os
import sys
import yaml
import json
import math
import signal
import random

import fire
from rich import print
from addict import Dict
from pathlib import Path

import transformers
import torch
import torch.optim as optim
import torch.distributed as dist

from core.utils import ExperimentManager, print_rank0, print_gpu_usage


class BaseTrainer:
    def __init__(self):
        pass

    def run(self, run_name):
        pass


class DistributedBaseTrainer:
    def __init__(self):
        pass

    def run(self, run_name):
        pass

    def save_model(self, local_rank, model, tokenizer, optimizer, run_dir, ckpt_name):
        cpu_state = model.state_dict()
        ckpt_dir = os.path.join(run_dir, ckpt_name)
        print_rank0(f'Saving FSDP model to: {ckpt_dir}')
        if local_rank == 0:
            model.save_pretrained(ckpt_dir, state_dict=cpu_state)
            tokenizer.save_pretrained(ckpt_dir)
        dist.barrier()
        print_rank0(f'Model saved to: {ckpt_dir}')

    def _save_checkpoint_for_signal(self):
        """
        Called whenever SIGTERM/SIGUSR1/SIGINT is caught.
        Saves:
          - FSDP model params (via save_model_and_optimizer)
          - optimizer + scheduler + global_step + epoch + RNG + sampler state
        """
        checkpoint_name = 'checkpoint-signal'
        # 1) Save model weights + tokenizer in FSDP-compatible way
        self.save_model(
            self.local_rank,
            self.model,
            self.tokenizer,
            self.optimizer,
            self.run_dir,
            checkpoint_name,
        )

        # 2) Save optimizer/scheduler/step/epoch/rng/sampler extras
        extra_state = dict(
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            random_state=random.getstate(),
            torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state(),
        )
        extra_info = {
            'global_step': self.global_step,
            'epoch': self.epoch,
        }

        extra_save_path = os.path.join(self.run_dir, checkpoint_name, 'extra.pt')
        extra_info_path = os.path.join(self.run_dir, checkpoint_name, 'extra_info.json')

        torch.save(extra_state, extra_save_path)
        with open(extra_info_path, 'w') as f:
            json.dump(extra_info, f)
        
        print_rank0(f'Checkpoint "{checkpoint_name}" saved (step={self.global_step}, epoch={self.epoch})')
        # Ensure all ranks finish I/O before exit
        dist.barrier()
        sys.exit(0)

    def _register_signal_handlers(self):
        """
        Trap SIGTERM, SIGUSR1 (e.g. from SLURM), and SIGINT (Ctrl+C),
        and call self._save_checkpoint_for_signal() when any arrive.
        """
        def _handle(signum, frame):
            #print_rank0(f'\n[Rank {self.local_rank}] Caught signal {signum}. Saving...')
            print(f'\n[Rank {self.local_rank}] Caught signal {signum}. Saving...')
            self._save_checkpoint_for_signal()

        signal.signal(signal.SIGTERM, _handle)
        signal.signal(signal.SIGUSR1, _handle)
        signal.signal(signal.SIGUSR2, _handle)


class Trainer(BaseTrainer):
    def __init__(self):
        super().__init__()

    def run(self, run_name, seed=0, show_gpu_usage=False, exp_dir=None):
        assert run_name is not None, 'run_name must be specified'

        if exp_dir is None:
            EXP_DIR = 'runs'
        else:
            EXP_DIR = exp_dir

        run_dir = Path(EXP_DIR) / run_name
        exp_manager = ExperimentManager(run_dir)
    
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
        # configs #################################################################
        run_config = dict(
            seed=seed,
            run_dir=str(run_dir),
            run_name=run_name,
            print_every=100,
        )
        model_config = dict(
            num_layers=2,
            vocab_size=100,
            hidden_size=128,
            intermediate_size=1000,
            num_heads=4,
            num_key_value_heads=2,
            max_seq_length=64,
        )
    
        data_config = dict(
            dataset_name='random_index',
            num_datapoints=500,
        )
    
        training_config = dict(
            num_epochs=50,
            batch_size=32,
            lr=5e-3,
        )
        ###########################################################################
        full_config = dict(
            run_config=run_config,
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
        )
        print(full_config)
        run_config = Dict(run_config)
        model_config = Dict(model_config)
        data_config = Dict(data_config)
        training_config = Dict(training_config)
    
        dataloader = RandomIndexSequenceDataset.construct_dataloader(
            num_datapoints=data_config.num_datapoints,
            max_seq_length=model_config.max_seq_length,
            vocab_size=model_config.vocab_size,
            batch_size=training_config.batch_size,
        )
    
        model = LanguageModel(**model_config).cuda()
        optimizer = optim.AdamW(model.parameters(), lr=training_config.lr)
    
        exp_manager.save_config(full_config)
        exp_manager.write('### START ###')
        model = self.run_trian(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            num_epochs=training_config.num_epochs,
            print_every=run_config.print_every,
            exp_manager=exp_manager,
            show_gpu_usage=show_gpu_usage,
        )
        avg_loss = self.run_evaluate(model, dataloader)
        exp_manager.write(f'[eval] loss={avg_loss:.6f}')
        model.save_pretrained(run_dir / 'final')

    def run_trian(
        self,
        model,
        dataloader,
        optimizer,
        num_epochs,
        print_every='none',
        exp_manager=None,
        show_gpu_usage=False,
    ):
        model.train()
    
        total_training_steps = num_epochs * len(dataloader)
        scheduler = transformers.get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=math.ceil(total_training_steps * 0.05),
            num_training_steps=total_training_steps,
        )
    
        global_step = 0
        for epoch_idx in range(num_epochs):
            for batch in dataloader:
                batch = {k: v.cuda() for k, v in batch.items()}
                lr = scheduler.get_last_lr()[0]
                optimizer.zero_grad()
                output = model(**batch)
                loss = output.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if print_every != 'none' and global_step % print_every == 0:
                    if exp_manager is not None:
                        exp_manager.write(f'[train] epoch_idx={epoch_idx} ({global_step} / {total_training_steps}) | loss={loss.item():.6f} | lr={lr:.6f}')
                    if show_gpu_usage:
                        print_gpu_usage()
                global_step += 1
        return model

    @torch.no_grad()
    def run_evaluate(self, model, dataloader):
        model.eval()
        total_loss = []
        for batch in dataloader:
            batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss
        total_loss.append(loss.item())
        avg_loss = sum(total_loss) / len(total_loss)
        return avg_loss


def main(run_name, seed=0):
    trainer = Trainer()
    trainer.run(run_name=run_name, seed=seed)


if __name__ == '__main__':
    fire.Fire(main)
