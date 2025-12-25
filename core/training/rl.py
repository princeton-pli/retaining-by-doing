"""
Based on Ray and FSDP.
"""
import os
import re
import sys
import copy
import time
import yaml
import time
import math
import shutil
import random
import signal
import functools
from pathlib import Path
import fire
import wandb

import ray
import vllm
from vllm import LLM, SamplingParams
import transformers
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
)
from core.training.trainer import DistributedBaseTrainer
from core.utils import load_module, ExperimentManager, print_rank0
from core.data import tokenizer_factory

os.environ['RAY_DEDUP_LOGS'] = '0'


def print_gpu_usage(num_total_gpus):
    nvmlInit()

    print('========== GPU Utilization ==========')
    for gpu_id in range(num_total_gpus):
        h = nvmlDeviceGetHandleByIndex(gpu_id)
        info = nvmlDeviceGetMemoryInfo(h)
        print(f'GPU {gpu_id}')
        print(f'- Used:       {info.used / 1024 ** 3:>8.2f} B ({info.used / info.total * 100:.1f}%)')
        print(f'- Available:  {info.free / 1024 ** 3:>8.2f} B ({info.free / info.total * 100:.1f}%)')
        print(f'- Total:      {info.total / 1024 ** 3:>8.2f} B')
    print('=====================================')


@ray.remote(num_gpus=1)
class OnPolicyActor:
    def __init__(self, model_name, num_gen_gpus, max_seq_length, max_new_tokens):
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=num_gen_gpus,
            #max_model_len=max_seq_length + max_new_tokens,
        )
        self.model_update_group = dist.new_group(ranks=list(range(num_gen_gpus)))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(
        self,
        input_texts,
        max_new_tokens,
        temperature=1.0,
        num_generation_per_prompt=1,
    ):
        outputs = self.model.generate(
            input_texts,
            sampling_params=SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=1.0,
                n=num_generation_per_prompt,
            ),
            use_tqdm=False,
        )
        results = [
            dict(
                input_text=output.prompt,
                generated_text=[output.outputs[i].text for i in range(num_generation_per_prompt)],
                finish_reason=[output.outputs[i].finish_reason for i in range(num_generation_per_prompt)],
            ) for output in outputs
        ]
        return results

    def update_weight(self, name, param):
        dist.broadcast(param, src=0, group=self.model_update_group)
        self.model.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(weights=[(name, param)])


def generate_on_policy_data(
    on_policy_actor,
    tokenizer,
    datapoint_ids,
    dataset,
    max_new_tokens,
    temperature,
    num_generation_per_prompt=1,
):
    batch_for_gen = dataset.tokenize_for_inference_by_ids(datapoint_ids)
    input_texts = tokenizer.batch_decode(batch_for_gen['input_ids'], skip_special_tokens=False)
    results = ray.get(
        on_policy_actor.generate.remote(
            input_texts=input_texts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_generation_per_prompt=num_generation_per_prompt,
        )
    )
    on_policy_texts = [result['generated_text'] for result in results]
    parsed_on_policy_texts = [[dataset.parse_output_text(text) for text in on_policy_text] for on_policy_text in on_policy_texts]
    ends_with_eos = [[r == 'stop' for r in result['finish_reason']] for result in results]
    return on_policy_texts, parsed_on_policy_texts, ends_with_eos


def log_stats(
    run_name,
    global_step,
    total_training_steps,
    epoch,
    loss,
    reward,
    kl,
    grad_norm,
    scheduler,
    wandb_project='none',
    exp_manager=None,
    time_spent=None,
):
    lr = scheduler.get_last_lr()[0]
    if time_spent is not None:
        time_spent = time_spent / 60
    log_msg = f'[train] run_name={run_name} | global_step={global_step} / {total_training_steps} | epoch={epoch} | loss={loss:.6f} | reward={reward.item():.4f} | kl={kl.item():.4f} | lr={lr:.10f} | grad_norm={grad_norm:.6f} | time_spent={time_spent:.2f} min'

    if exp_manager is not None:
        exp_manager.write(log_msg)

    if wandb_project != 'none':
        wandb.log({
            'global_step': global_step,
            'total_training_steps': total_training_steps,
            'epoch': epoch,
            'loss': loss,
            'reward': reward,
            'kl': kl,
            'lr': lr,
            'grad_norm': grad_norm,
        })


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


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


class RewardModel:
    def __init__(
        self,
        reward_model_name,
        reward_fn,
        tokenizer,
        **kwargs,
    ):
        self.reward_model_name = reward_model_name
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.model = None  # TODO: initialize model
    
    def __call__(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def compute_outcome_reward(self, text, datapoint):
        return self.reward_fn(text, datapoint)


@ray.remote(num_gpus=1)
class RLRayFSDPTrainer(DistributedBaseTrainer):
    def __init__(
        self,
        backend,
        master_addr,
        master_port,
        local_rank,
        world_size,
        on_policy_actor=None,
    ):
        super().__init__()
        self.local_rank = local_rank
        self.world_size = world_size
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port
        self.on_policy_actor = on_policy_actor
    
    def run(self, **kwargs):
        run_name = kwargs.get('run_name', None)
        run_base_dir = kwargs.get('run_base_dir', None)
        seed = kwargs.get('seed', 0)
        model_name = kwargs.get('model_name', None)
        ref_model_name = kwargs.get('ref_model_name', 'none')
        reward_model_name = kwargs.get('reward_model_name', 'none')
        max_new_tokens = kwargs.get('max_new_tokens', 200)
        temperature = kwargs.get('temperature', 0.0)
        num_generation_per_prompt = kwargs.get('num_generation_per_prompt', 1)
        kl_beta = kwargs.get('kl_beta', 0.0)
        gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', None)
        lr = kwargs.get('lr', None)
        num_epochs = kwargs.get('num_epochs', None)
        warmup_ratio = kwargs.get('warmup_ratio', None)
        scheduler_type = kwargs.get('scheduler_type', None)
        gradient_clipping = kwargs.get('gradient_clipping', None)
        eval_every_n_steps = kwargs.get('eval_every_n_steps', None)
        save_every_n_steps = kwargs.get('save_every_n_steps', None)
        print_every_n_steps = kwargs.get('print_every_n_steps', None)
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
        num_gpus = kwargs.get('num_gpus', None)
        num_gen_gpus = kwargs.get('num_gen_gpus', None)

        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = str(self.master_port)
        os.environ['LOCAL_RANK'] = str(self.local_rank)
        os.environ['WORLD_SIZE'] = str(self.world_size)

        backend = self.backend
        master_addr = self.master_addr
        master_port = self.master_port
        local_rank = self.local_rank
        world_size = self.world_size

        exp_start_time = time.time()
        assert run_name is not None, 'run_name must be specified'

        run_dir = Path(run_base_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        exp_manager = ExperimentManager(run_dir, world_size=world_size)
        exp_manager.write(f'backend={backend} | master_addr={master_addr} | master_port={master_port} | num_gpus={num_gpus} | num_gen_gpus={num_gen_gpus} | world_size={world_size}')

        torch.cuda.set_device(0)
        dist.init_process_group(
            backend=backend,
            rank=local_rank,
            world_size=world_size,
        )

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        transformers.set_seed(seed)

        config = copy.deepcopy(kwargs)
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

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side='right')
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
            use_orig_params=True,  # Critical for generation with FSDP.summon_full_params()
        )
        model = FSDP(model, **fsdp_config)

        if ref_model_name != 'none' and kl_beta > 0.0:
            ref_model = model_module.from_pretrained(
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

        reward_model = RewardModel(
            reward_model_name=reward_model_name,
            reward_fn=train_dataset.reward_fn,
            tokenizer=tokenizer,
            model_module=model_module,
            fsdp_config=fsdp_config,
        )

        training_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

        total_training_steps = training_steps_per_epoch * num_epochs
        scheduler = transformers.get_scheduler(
            name=scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=math.ceil(total_training_steps * warmup_ratio),
            num_training_steps=total_training_steps,
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
        total_rewards = []
        total_kl = []
        best_model_reward = float('-inf')

        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.run_dir = run_dir
        self.local_rank = local_rank
        self.global_step = global_step
        self.epoch = 0
        self.train_sampler = train_sampler
        self._register_signal_handlers()

        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            for step, batch in enumerate(train_dataloader):
                datapoint_ids = batch['datapoint_ids'].tolist()
                datapoints = [train_dataset.id_to_datapoint[datapoint_id] for datapoint_id in datapoint_ids]

                ###############################################################
                # NOTE: generate on-policy data
                batch_on_policy_texts, batch_parsed_on_policy_texts, batch_ends_with_eos = generate_on_policy_data(
                    self.on_policy_actor,
                    tokenizer,
                    datapoint_ids,
                    train_dataset,
                    max_new_tokens,
                    temperature,
                    num_generation_per_prompt,
                )
                ###############################################################

                output = loss_module.compute_loss(
                    model,
                    ref_model,
                    reward_model,
                    batch_on_policy_texts,
                    batch_parsed_on_policy_texts,
                    batch_ends_with_eos,
                    datapoints,
                    tokenizer,
                    local_rank=local_rank,
                    **kwargs,
                )
                loss = output.loss
                rewards = output.rewards
                kl = output.kl
                probs = output.logprobs.exp()

                total_rewards.append(rewards)
                total_kl.append(kl)
                ###############################################################
                if print_every_n_steps is not None and step != 0 and step % print_every_n_steps == 0 and local_rank == 0:
                    try:
                        for batch_idx, (group_rewards, on_policy_texts, parsed_on_policy_texts, datapoint_id) in enumerate(zip(rewards, batch_on_policy_texts, batch_parsed_on_policy_texts, datapoint_ids)):
                            if batch_idx >= 3:
                                break
                            datapoint = train_dataset.id_to_datapoint[datapoint_id]
                            target_text = datapoint['targets']
                            group_rewards = group_rewards.tolist()
                            for group_idx, (reward, on_policy_text, parsed_on_policy_text) in enumerate(zip(group_rewards, on_policy_texts, parsed_on_policy_texts)):
                                if group_idx >= 1:
                                    break
                                _input_text = datapoint["messages"][0]["content"]
                                _on_policy_text = on_policy_text    
                                _parsed_on_policy_text = parsed_on_policy_text
                                print(f'{"#" * 100}\nlocal_rank={local_rank}\nid={datapoint_id}\n### input_text ###\n{_input_text}\n### on_policy_text ###\n{_on_policy_text}\npred_text={_parsed_on_policy_text}\ntarget_text={target_text}\nreward={reward}\n{"#" * 100}')
                    except Exception as e:
                        print(f'Error in printing: {e}')
                ###############################################################

                # Save predictions ############################################
                exp_manager.write_prediction(dict(
                    datapoint_ids=datapoint_ids,
                    datapoints=datapoints,
                    rewards=rewards.tolist(),
                    probs=probs.tolist(),
                    on_policy_texts=batch_on_policy_texts,
                    parsed_on_policy_texts=batch_parsed_on_policy_texts,
                    ends_with_eos=batch_ends_with_eos,
                    local_rank=local_rank,
                    global_step=global_step,
                ))
                ###############################################################

                (loss / gradient_accumulation_steps).backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    grad_norm = model.clip_grad_norm_(gradient_clipping).item()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    loss = get_all_reduce_mean(loss.detach())

                    total_rewards = torch.cat(total_rewards, dim=0)
                    reward = get_all_reduce_mean(total_rewards.mean())
                    total_rewards = []

                    total_kl = torch.cat(total_kl, dim=0)
                    kl = get_all_reduce_mean(total_kl.mean())
                    total_kl = []
                    if local_rank == 0:
                        current_time = time.time()
                        log_stats(
                            run_name=run_name,
                            global_step=global_step,
                            total_training_steps=total_training_steps,
                            epoch=epoch,
                            loss=loss,
                            reward=reward,
                            kl=kl,
                            grad_norm=grad_norm,
                            scheduler=scheduler,
                            wandb_project=wandb_project,
                            exp_manager=exp_manager,
                            time_spent=current_time - exp_start_time,
                        )
                        print_gpu_usage(num_gpus)

                    if reward > best_model_reward:
                        best_model_reward = reward
                        self.save_model(local_rank, model, tokenizer, optimizer, run_dir, ckpt_name=f'checkpoint-best')
                        exp_manager.write(f'Best model saved at global_step={global_step} | best_model_reward={best_model_reward}')

                    ###########################################################
                    # NOTE: update on-policy actor weights
                    def update_on_policy_actor_weights(model, on_policy_actor, local_rank):
                        full_state_dict = None
                        with FSDP.summon_full_params(model, writeback=False):
                            full_state_dict = model.state_dict()
    
                            for name, param in full_state_dict.items():
                                dist.broadcast(param, src=0)
                                ray.get(self.on_policy_actor.update_weight.remote(name, param))
                        dist.barrier()
                    update_on_policy_actor_weights(model, self.on_policy_actor, local_rank)
                    print_rank0('Weights updated!!!')
                    ###########################################################

                    if global_step and global_step % eval_every_n_steps == 0:
                        eval_loss = run_evaluation(model, eval_dataloader, local_rank)
                        log_info = f'\[eval]  global_step={global_step} / {total_training_steps} | epoch={epoch} | eval_loss={eval_loss:.6f}'
                        if exp_manager is not None:
                            exp_manager.write(log_info)
                        model.train()
    
                    if global_step and global_step % save_every_n_steps == 0:
                        self.save_model(local_rank, model, tokenizer, optimizer, run_dir, ckpt_name=f'checkpoint-{global_step}')
                    
                    self.model = model
                    self.tokenizer = tokenizer
                    self.optimizer = optimizer
                    self.scheduler = scheduler
                    self.run_dir = run_dir
                    self.local_rank = local_rank
                    self.global_step = global_step
                    self.epoch = 0
                    self.train_sampler = train_sampler
    
                    dist.barrier()
                    global_step += 1
        dist.barrier()
    
        self.save_model(local_rank, model, tokenizer, optimizer, run_dir, ckpt_name='final')
        print_rank0(f'Training finished in {(time.time() - start_time) / 60:.2f} minutes')
        dist.destroy_process_group()
        return True  # successful termination


def main(**kwargs):
    backend = kwargs.pop('backend', 'nccl')
    master_addr = kwargs.pop('master_addr', '127.0.0.1')
    master_port = kwargs.pop('master_port', '29500')
    num_gpus = kwargs.get('num_gpus')
    num_gen_gpus = kwargs.get('num_gen_gpus')
    world_size = num_gpus - num_gen_gpus
    print(f'run_name={kwargs.get("run_name")} | backend={backend} | master_addr={master_addr} | master_port={master_port} | num_gpus={num_gpus} | num_gen_gpus={num_gen_gpus} | world_size={world_size}')

    ray.init()

    on_policy_actors = [
        OnPolicyActor.remote(
            model_name=kwargs.get('model_name'),
            num_gen_gpus=1,
            max_seq_length=kwargs.get('max_seq_length'),
            max_new_tokens=kwargs.get('max_new_tokens'),
        ) for _ in range(num_gen_gpus)
    ]

    training_actors = [
        RLRayFSDPTrainer.remote(
            backend,
            master_addr,
            master_port,
            local_rank,
            world_size,
            on_policy_actors[local_rank % num_gen_gpus],
        ) for local_rank in range(world_size)
    ]

    status = ray.get([actor.run.remote(**kwargs) for actor in training_actors])
    print(status)


if __name__ == '__main__':
    fire.Fire(main)
