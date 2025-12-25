"""
Based on FSDP.
"""
import os
import copy
import yaml
import time
import math
import random
import functools
from pathlib import Path
import fire
import wandb
from rich import print

import numpy as np
import transformers
from transformers import AutoTokenizer
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy

from core.training.trainer import BaseTrainer
from core.utils import print_gpu_usage, load_module, ExperimentManager
from core.data import tokenizer_factory


def print_rank0(msg):
    if dist.get_rank() == 0:
        print(msg)


def log_stats(global_step, total_training_steps, epoch, loss, grad_norm, scheduler, wandb_project='none', exp_manager=None):
    lr = scheduler.get_last_lr()[0]
    log_msg = f'\[train] global_step={global_step} / {total_training_steps} | epoch={epoch} | loss={loss:.6f} | lr={lr:.10f} | grad_norm={grad_norm:.6f}'

    if exp_manager is not None:
        exp_manager.write(log_msg)

    if wandb_project != 'none':
        wandb.log({
            'global_step': global_step,
            'total_training_steps': total_training_steps,
            'epoch': epoch,
            'loss': loss,
            'lr': lr,
            'grad_norm': grad_norm,
        })


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def save_model_and_optimizer(local_rank, model, tokenizer, optimizer, run_dir, ckpt_name):
    cpu_state = model.state_dict()
    ckpt_dir = os.path.join(run_dir, ckpt_name)
    print_rank0(f'Saving FSDP model to: {ckpt_dir}')
    if local_rank == 0:
        model.save_pretrained(ckpt_dir, state_dict=cpu_state)
        tokenizer.save_pretrained(ckpt_dir)
    
    # TODO: save optimizer
    ...


def get_optimizer(model, lr):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if 'bias' not in name]
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            'weight_decay': 0.0,
        },
        {
            'params': [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    return optimizer


@torch.no_grad()
def run_evaluation(model, eval_dataloader, local_rank):
    """
    Returns the averaged loss reduced across all processes.
    """
    model.eval()
    print_rank0('Running eval for loss...')

    losses = 0
    for step, batch in enumerate(eval_dataloader):
        inputs = {k: v.cuda() for k, v in batch.items() if k in ('input_ids', 'attention_mask', 'labels')}
        outputs = model(**inputs)
        loss = outputs.loss
        losses += loss.float()
    losses = losses / len(eval_dataloader)
    eval_loss = get_all_reduce_mean(losses.clone()).item()
    return eval_loss


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f'{name}.{n}'
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result


class FSDPTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()

    def run(self, **kwargs):
        run_name = kwargs.get('run_name', None)
        run_base_dir = kwargs.get('run_base_dir', None)
        seed = kwargs.get('seed', 0)
        model_name = kwargs.get('model_name', None)
        ref_model_name = kwargs.get('ref_model_name', None)
        gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', None)
        lr = kwargs.get('lr', None)
        num_epochs = kwargs.get('num_epochs', None)
        warmup_ratio = kwargs.get('warmup_ratio', None)
        scheduler_type = kwargs.get('scheduler_type', None)
        gradient_clipping = kwargs.get('gradient_clipping', None)
        eval_every_n_steps = kwargs.get('eval_every_n_steps', None)
        save_every_n_steps = kwargs.get('save_every_n_steps', None)
        model_module_path = kwargs.get('model_module_path', None)
        transformer_layer_module_path = kwargs.get('transformer_layer_module_path', None)
        loss_module_path = kwargs.get('loss_module_path', None)
        cache_path = kwargs.get('cache_path', None)
        data_module_path = kwargs.get('data_module_path', None)
        dataset_name = kwargs.get('dataset_name', None)
        num_train_datapoints = kwargs.get('num_train_datapoints', None)
        num_eval_datapoints = kwargs.get('num_eval_datapoints', None)
        train_batch_size_per_device = kwargs.get('train_batch_size_per_device', None)
        eval_batch_size_per_device = kwargs.get('eval_batch_size_per_device', None)
        total_batch_size = kwargs.get('total_batch_size', None)
        max_seq_length = kwargs.get('max_seq_length', None)
        wandb_project = kwargs.get('wandb_project', 'none')
        show_gpu_usage = kwargs.get('show_gpu_usage', False)
        kl_beta = kwargs.get('kl_beta', 0.0)

        assert run_name is not None, 'run_name must be specified'

        run_dir = Path(run_base_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        exp_manager = ExperimentManager(run_dir, world_size=world_size)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl', rank=local_rank, world_size=world_size)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        transformers.set_seed(seed)

        config = dict(
            run_name=run_name,
            run_base_dir=run_base_dir,
            seed=seed,
            model_name=model_name,
            dataset_name=dataset_name,
            wandb_project=wandb_project,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr=lr,
            num_epochs=num_epochs,
            warmup_ratio=warmup_ratio,
            scheduler_type=scheduler_type,
            gradient_clipping=gradient_clipping,
            eval_every_n_steps=eval_every_n_steps,
            save_every_n_steps=save_every_n_steps,
            model_module_path=model_module_path,
            transformer_layer_module_path=transformer_layer_module_path,
            loss_module_path=loss_module_path,
            cache_path=cache_path,
            data_module_path=data_module_path,
            num_train_datapoints=num_train_datapoints,
            num_eval_datapoints=num_eval_datapoints,
            train_batch_size_per_device=train_batch_size_per_device,
            eval_batch_size_per_device=eval_batch_size_per_device,
            max_seq_length=max_seq_length,
        )
        print_rank0(config)

        model_module = load_module(model_module_path)
        transformer_layer_module = load_module(transformer_layer_module_path)
        model = model_module.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
            use_cache=False,
            trust_remote_code=True,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side='right')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer = tokenizer_factory(tokenizer, dataset_split='train')

        fsdp_config = dict(
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={transformer_layer_module},
            ),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            backward_prefetch=None,
            param_init_fn=None,
            cpu_offload=None,
        )
        model = FSDP(model, **fsdp_config)

        if ref_model_name != 'none' and kl_beta > 0.0:
            # Gradients of ref_model will not be tracked
            ref_model_module = load_module(model_module_path)
            ref_model = ref_model_module.from_pretrained(
                ref_model_name,
                torch_dtype=torch.bfloat16,
                use_cache=False,
                trust_remote_code=True,
                local_files_only=True,
            )
            ref_model.eval()
            ref_model = FSDP(ref_model, **fsdp_config)
        else:
            ref_model = None
                
        optimizer = get_optimizer(model, lr)

        dataset_module = load_module(data_module_path, dataset_name)
        train_dataloader, train_sampler, train_dataset = dataset_module.construct_dataloader(
            num_datapoints=num_train_datapoints,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            batch_size=train_batch_size_per_device,
            shuffle=True,
            use_distributed=True,
            seed=seed,
            local_rank=local_rank,
            world_size=world_size,
            dataset_split='train',
            model_name=model_name,
        )
        eval_dataloader, eval_sampler, eval_dataset = dataset_module.construct_dataloader(
            num_datapoints=num_eval_datapoints,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            batch_size=eval_batch_size_per_device,
            shuffle=False,
            use_distributed=True,
            seed=seed,
            local_rank=local_rank,
            world_size=world_size,
            dataset_split='eval',
            model_name=model_name,
        )

        training_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        total_training_steps = training_steps_per_epoch * num_epochs
        scheduler = transformers.get_scheduler(
            name=scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=math.ceil(total_training_steps * warmup_ratio),
            num_training_steps=total_training_steps,
            scheduler_specific_kwargs={'min_lr': lr * 0.1} if scheduler_type == 'cosine_with_min_lr' else None,
        )

        loss_module = load_module(loss_module_path)(config=config)
        model.gradient_checkpointing_enable()
        model.train()
        dist.barrier()

        if local_rank == 0:
            with open(run_dir / 'config.yaml', 'w') as f:
                yaml.dump(config, f)

            if wandb_project != 'none':
                os.environ['WANDB_MODE'] = 'dryrun'
                os.environ['WANDB_PROJECT'] = wandb_project
                wandb.init(project=wandb_project, name=run_name, config=config, dir=run_base_dir)
    
        start_time = time.time()
        global_step = 0
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            for step, batch in enumerate(train_dataloader):
                output = loss_module.compute_loss(
                    model,
                    batch,
                    ref_model=ref_model,
                    kl_beta=kl_beta,
                    tokenizer=tokenizer,
                    local_rank=local_rank,
                )
                loss = output.loss

                # Save predictions ############################################
                datapoint_ids = batch['datapoint_ids'].tolist()
                datapoints = [train_dataset.id_to_datapoint[datapoint_id] for datapoint_id in datapoint_ids]
                exp_manager.write_prediction(dict(
                    datapoint_ids=datapoint_ids,
                    datapoints=datapoints,
                    probs=output.logprobs.squeeze(-1).exp().tolist() if output.logprobs is not None else None,
                    local_rank=local_rank,
                    global_step=global_step,
                ))
                ###############################################################

                (loss / gradient_accumulation_steps).backward()

                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1 or step == 0:
                    grad_norm = model.clip_grad_norm_(gradient_clipping).item()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    loss = get_all_reduce_mean(loss.detach())
                    if local_rank == 0:
                        log_stats(
                            global_step=global_step,
                            total_training_steps=total_training_steps,
                            epoch=epoch,
                            loss=loss,
                            grad_norm=grad_norm,
                            scheduler=scheduler,
                            wandb_project=wandb_project,
                            exp_manager=exp_manager,
                        )
                        if show_gpu_usage:
                            print_gpu_usage()
    
                    if global_step and global_step % eval_every_n_steps == 0:
                        eval_loss = run_evaluation(model, eval_dataloader, local_rank)
                        log_info = f'\[eval]  global_step={global_step} / {total_training_steps} | epoch={epoch} | eval_loss={eval_loss:.6f}'
                        if exp_manager is not None:
                            exp_manager.write(log_info)
                        model.train()
    
                    if global_step and global_step % save_every_n_steps == 0:
                        save_model_and_optimizer(local_rank, model, tokenizer, optimizer, run_dir, ckpt_name=f'checkpoint-{global_step}')
    
                    global_step += 1
        dist.barrier()
        save_model_and_optimizer(local_rank, model, tokenizer, optimizer, run_dir, ckpt_name='final')
        print_rank0(f'Training finished in {(time.time() - start_time) / 60:.2f} minutes')
        dist.destroy_process_group()


def main(**kwargs):
    trainer = FSDPTrainer()
    trainer.run(**kwargs)


if __name__ == '__main__':
    fire.Fire(main)
