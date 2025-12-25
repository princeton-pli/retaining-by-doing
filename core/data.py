"""
python -m core.data
"""
import os
import re
import ast
import json
import math
import random
import operator
from pathlib import Path
from functools import partial
from collections import Counter
from tqdm import tqdm
from rich import print

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    Dataset,
    DataLoader,
    DistributedSampler,
)
from datasets import load_from_disk

random.seed(0)


llama_chat_template = '''{{ '<|begin_of_text|>' }}{% for message in messages %}{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>
{{ message['content'] }}<|eot_id|>{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>

{{ message['content'] }}<|eot_id|>{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>

{{ message['content'] }}<|eot_id|>{% else %}{{ raise_exception("Invalid role: " + message['role']) }}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}<|start_header_id|>assistant<|end_header_id|>
{% endif %}
'''

def tokenizer_factory(tokenizer, dataset_split='eval'):
    assert dataset_split in ('train', 'eval')
    _model_name = tokenizer.name_or_path.lower()

    if 'llama' in _model_name:
        #if dataset_split == 'train':
        tokenizer.chat_template = llama_chat_template
        tokenizer.assistant_start_text = '<|start_header_id|>assistant<|end_header_id|>'
        tokenizer.eos_token = '<|eot_id|>'
    elif 'gemma' in _model_name:
        tokenizer.assistant_start_text = '<start_of_turn>model\n'
    elif 'qwen' in _model_name:
        tokenizer.eos_token = '<|im_end|>'
        tokenizer.assistant_start_text = '<|im_start|>assistant\n'
    else:
        raise ValueError(f'Unsupported model: {tokenizer.name_or_path}')
    
    return tokenizer


def tokenize_messages(messages, tokenizer, max_seq_length, dataset_split='eval', **kwargs):
    """
    Code modified from: https://github.com/allenai/open-instruct/blob/main/open_instruct/finetune.py#L310

    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    apply_chat_template = partial(tokenizer.apply_chat_template, tokenize=False, add_generation_prompt=False)
    example_text = apply_chat_template(messages).strip()
    tokenized_example = tokenizer(example_text, return_tensors='pt', truncation=False, add_special_tokens=False)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    
    # NOTE: where the length selection happens; allow for 50 extra buffer tokens
    if input_ids.size(1) > max_seq_length + 200:
        return None

    # mask the non-assistant part to exclude from loss calculation
    for message_idx, message in enumerate(messages):
        if message['role'] != 'assistant':
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    apply_chat_template(messages[:message_idx]),
                    return_tensors='pt',
                    max_length=max_seq_length,
                    truncation=True,
                    add_special_tokens=False,
                ).input_ids.size(1)
            if message_idx < len(messages) - 1 and messages[message_idx + 1]['role'] == 'assistant':
                # here we also ignore the role of the assistant
                messages_so_far = apply_chat_template(messages[:message_idx + 1]) + tokenizer.assistant_start_text
            else:
                messages_so_far = apply_chat_template(messages[:message_idx + 1])
            message_end_idx = tokenizer(messages_so_far, return_tensors='pt', max_length=max_seq_length, truncation=True, add_special_tokens=False).input_ids.size(1)
            labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break
    attention_mask = torch.ones_like(input_ids)
    return dict(
        input_ids=input_ids.flatten(),
        attention_mask=attention_mask.flatten(),
        labels=labels.flatten(),
    )


class BaseDataset(Dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def collate_fn(self, batch):
        pass

    def get_dataloader(self, dataset, batch_size, shuffle, use_distributed=False, **kwargs):
        if use_distributed:
            dataloader, sampler = distributed_dataloader(
                dataset=dataset,
                batch_size=batch_size,
                collate_fn=self.collate_fn,
                shuffle=shuffle,
                seed=kwargs['seed'],
                local_rank=kwargs['local_rank'],
                world_size=kwargs['world_size'],
            )
        else:
            dataloader, sampler = default_dataloader(
                dataset=dataset,
                batch_size=batch_size,
                collate_fn=self.collate_fn,
                shuffle=shuffle,
            )
        return dataloader, sampler

    def __len__(self):
        return self.data['input_ids'].size(0)

    def __getitem__(self, idx):
        return dict(
            datapoint_ids=self.data['datapoint_ids'][idx],
            input_ids=self.data['input_ids'][idx],
            attention_mask=self.data['attention_mask'][idx],
            labels=self.data['labels'][idx],
        )

    def collate_fn(self, batch):
        return dict(
            datapoint_ids=torch.stack([b['datapoint_ids'] for b in batch]),
            input_ids=torch.stack([b['input_ids'] for b in batch]),
            attention_mask=torch.stack([b['attention_mask'] for b in batch]),
            labels=torch.stack([b['labels'] for b in batch]),
        )

    @classmethod
    def construct_dataloader(
        cls,
        num_datapoints,
        max_seq_length,
        tokenizer,
        batch_size,
        shuffle,
        use_distributed=False,
        dataset_split=None,
        **kwargs,
    ):
        dataset = cls(
            num_datapoints=num_datapoints,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            dataset_split=dataset_split,
            **kwargs,
        )
        dataloader, sampler = dataset.get_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            use_distributed=use_distributed,
            **kwargs,
        )
        return dataloader, sampler, dataset

    def get_eval_batcher(
        self,
        batch_size,
        chat_template_fn=None,
        **kwargs,
    ):
        tokenizer = tokenizer_factory(self.tokenizer, dataset_split='eval')
    
        if chat_template_fn is None:
            chat_template_fn = partial(
                tokenizer.apply_chat_template,
                tokenize=False,
                add_generation_prompt=True,
            )

        num_batches = math.ceil(len(self.datapoints) / batch_size)
        for batch_idx in range(num_batches):
            batch_datapoints = self.datapoints[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_input_texts = [chat_template_fn(datapoint['messages']) for datapoint in batch_datapoints]
            inputs = tokenizer.batch_encode_plus(
                batch_input_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                add_special_tokens=False,
            )
            yield inputs, batch_datapoints

    def get_eval_batch_input_texts(
        self,
        batch_size,
        chat_template_fn=None,
        **kwargs,
    ):
        tokenizer = tokenizer_factory(self.tokenizer, dataset_split='eval')

        if chat_template_fn is None:
            chat_template_fn = partial(
                tokenizer.apply_chat_template,
                tokenize=False,
                add_generation_prompt=True,
            )

        num_batches = math.ceil(len(self.datapoints) / batch_size)
        for batch_idx in range(num_batches):
            batch_datapoints = self.datapoints[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_input_texts = [chat_template_fn(datapoint['messages']) for datapoint in batch_datapoints]
            yield batch_input_texts, batch_datapoints
    
    def tokenize_for_inference_by_ids(self, datapoint_ids: list[int]):
        tokenizer = tokenizer_factory(self.tokenizer, dataset_split='eval')

        datapoints = [self.id_to_datapoint[idx] for idx in datapoint_ids]
        messages = [datapoint['messages'][:-1] for datapoint in datapoints]
        input_texts = [
            tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
            )
            for msg in messages
        ]
        encoded = tokenizer.batch_encode_plus(
            input_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=self.max_seq_length,
            padding_side='left',
        )
        return encoded


def default_dataloader(dataset, batch_size, collate_fn, shuffle):
    dataloader = DataLoader(
        dataset,
        pin_memory=False,
        drop_last=False,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return dataloader, None


def distributed_dataloader(dataset, batch_size, collate_fn, shuffle, seed, local_rank, world_size):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=shuffle,
        seed=seed,
    )
    dataloader = DataLoader(
        dataset,
        pin_memory=True,
        drop_last=True,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
    )
    return dataloader, sampler


class GSM8KDataset(BaseDataset):
    def __init__(self, num_datapoints, max_seq_length, tokenizer, dataset_split='eval', for_generation=False, **kwargs):
        super().__init__()
        self.num_datapoints = num_datapoints
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.dataset_split = dataset_split
        self.data = []
        self.id_to_datapoint = dict()
        self.model_name = kwargs.get('model_name', None)

        path = '/scratch/gpfs/DANQIC/howardchen/data/openai/gsm8k'
        dataset = load_from_disk(path)

        _dataset_split = 'test' if dataset_split == 'eval' else dataset_split
        dataset = dataset[_dataset_split]
        if num_datapoints == 'all':
            num_datapoints = len(dataset)
        self.num_datapoints = min(num_datapoints, len(dataset))

        all_tokenized = []
        datapoint_ids = []
        for datapoint_idx, datapoint in tqdm(enumerate(dataset), total=num_datapoints):
            if datapoint_idx >= num_datapoints:
                break

            user_content = datapoint['question']
            messages = [dict(role='user', content=user_content)]

            answer = datapoint['answer']
            if _dataset_split == 'train' and not for_generation:
                messages.append(dict(role='assistant', content=answer.split('####')[0].strip()))
            tokenized = tokenize_messages(messages, tokenizer, max_seq_length, dataset_split, **kwargs)

            parsed_output_text = self.parse_output_text(answer)
            datapoint['targets'] = [parsed_output_text]
            if tokenized is None:
                continue
            all_tokenized.append(tokenized)
            datapoint_ids.append(datapoint_idx)
            self.id_to_datapoint[datapoint_idx] = dict(
                messages=messages,
                **datapoint,
            )
        
        input_ids = [tokenized['input_ids'] for tokenized in all_tokenized]
        attention_mask = [tokenized['attention_mask'] for tokenized in all_tokenized]
        labels = [tokenized['labels'] for tokenized in all_tokenized]

        datapoint_ids = torch.tensor(datapoint_ids).long()
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        self.data = dict(
            datapoint_ids=datapoint_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        self.datapoints = [self.id_to_datapoint[datapoint_id] for datapoint_id in datapoint_ids.tolist()]

    def parse_output_text(self, output_text):
        if output_text is None:
            return None
        parsed = re.findall(r'\$?[\d,]+\.?\d*', output_text)
        if len(parsed) >= 1:
            parsed_output_text = parsed[-1]
            parsed_output_text = re.sub(r'[,$]', '', parsed_output_text)
            parsed_output_text = re.sub(r'\.$', '', parsed_output_text)
        else:
            parsed_output_text = None

        if parsed_output_text is not None:
            return parsed_output_text
        else:
            matches = re.findall(r'\\boxed{((?:[^{}]|{[^{}]*})*)}', output_text)
            if matches:
                return matches[-1].strip()
            
            matches = re.findall(r'\$([^$]+)\$', output_text)
            if matches:
                return matches[-1].strip()
                
            matches = re.findall(r'(?:^|[^\d])(\d+(?:\.\d+)?|\.\d+)(?:[^\d]|$)', output_text)
            if matches:
                return matches[-1].strip()

    def reward_fn(self, pred_text, datapoint):
        targets = datapoint['targets']
        reward = pred_text in targets
        return reward


class MATHDataset(BaseDataset):
    def __init__(self, num_datapoints, max_seq_length, tokenizer, dataset_split='eval', for_generation=False, **kwargs):
        super().__init__()
        from core.evaluation.math_utils import normalize_final_answer
        self.normalize_final_answer = normalize_final_answer
        self.num_datapoints = num_datapoints
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.dataset_split = dataset_split
        self.data = []
        self.id_to_datapoint = dict()
        self.model_name = kwargs.get('model_name', None)

        _dataset_split = 'test' if dataset_split == 'eval' else dataset_split
        dataset = []
        with open('data/eval/math.jsonl', 'r') as f:
            dataset = [json.loads(line.strip()) for line in f]
        if num_datapoints == 'all':
            num_datapoints = len(dataset)
        num_datapoints = min(num_datapoints, len(dataset))

        datapoint_ids = []
        all_tokenized = []
        for datapoint_idx, datapoint in tqdm(enumerate(dataset), total=num_datapoints):
            if len(all_tokenized) >= num_datapoints:
                break
            problem = datapoint['problem']
            solution = datapoint['solution']

            messages = [dict(role='user', content=f'{problem}\n\nReason about the problem, derive your answer, and wrap your final answer in $\\boxed{{ }}$.')]

            if _dataset_split == 'train' and not for_generation:
                messages.append(dict(role='assistant', content=solution))

            tokenized = tokenize_messages(messages, tokenizer, max_seq_length, dataset_split, **kwargs)
            if tokenized is None:
                continue
            
            parsed_output_text = self.parse_output_text(solution)
            datapoint['targets'] = [parsed_output_text]

            all_tokenized.append(tokenized)
            datapoint_ids.append(datapoint_idx)
            self.id_to_datapoint[datapoint_idx] = dict(
                messages=messages,
                **datapoint,
            )
        
        input_ids = [tokenized['input_ids'] for tokenized in all_tokenized]
        attention_mask = [tokenized['attention_mask'] for tokenized in all_tokenized]
        labels = [tokenized['labels'] for tokenized in all_tokenized]

        datapoint_ids = torch.tensor(datapoint_ids).long()
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        self.data = dict(
            datapoint_ids=datapoint_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        self.datapoints = [self.id_to_datapoint[datapoint_id] for datapoint_id in datapoint_ids.tolist()]

    def parse_output_text(self, output_text):
        if output_text is None:
            return None
        
        output_text = self._parse(output_text)
        
        matches = re.findall(r'\\boxed{((?:[^{}]|{[^{}]*})*)}', output_text)
        if matches:
            return matches[-1].strip()
        
        matches = re.findall(r'\$([^$]+)\$', output_text)
        if matches:
            return matches[-1].strip()
            
        matches = re.findall(r'(?:^|[^\d])(\d+(?:\.\d+)?|\.\d+)(?:[^\d]|$)', output_text)
        if matches:
            return matches[-1].strip()
        
        return output_text

    def _parse(self, text):
        text = text.replace('\\dfrac', '\\frac')
        text = text.replace('\pi', '\\pi')
        text = text.replace('\\left', '')
        text = text.replace('\\right', '')
        return text

    def reward_fn(self, pred_text, datapoint):
        pred_text = self._parse(self.normalize_final_answer(pred_text))
        targets = datapoint['targets']
        targets = [self._parse(self.normalize_final_answer(target)) for target in targets]
        return pred_text in targets


class WildJailbreakDataset(BaseDataset):
    def __init__(self, num_datapoints, max_seq_length, tokenizer, dataset_split='eval', for_generation=False, **kwargs):
        super().__init__()
        self.num_datapoints = num_datapoints
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.dataset_split = dataset_split
        self.data = []
        self.id_to_datapoint = dict()
        self.model_name = kwargs.get('model_name', None)

        dataset = []
        with open('data/eval/wildjailbreak.jsonl', 'r') as f:
            for line in f:
                dataset.append(json.loads(line.strip()))

        if num_datapoints == 'all':
            num_datapoints = len(dataset)
        num_datapoints = min(num_datapoints, len(dataset))

        datapoint_ids = []
        all_tokenized = []
        for datapoint_idx, datapoint in tqdm(enumerate(dataset), total=num_datapoints):
            if len(all_tokenized) >= num_datapoints:
                break
            
            query = datapoint['adversarial']
            messages = [dict(role='user', content=query)]

            if dataset_split == 'train' and not for_generation:
                messages.append(dict(role='assistant', content=datapoint['completion']))

            tokenized = tokenize_messages(messages, tokenizer, max_seq_length, dataset_split, **kwargs)
            if tokenized is None:
                continue
            
            if dataset_split == 'train':
                datapoint['targets'] = [None]
            else:
                datapoint['targets'] = datapoint['label']

            all_tokenized.append(tokenized)
            datapoint_ids.append(datapoint_idx)
            self.id_to_datapoint[datapoint_idx] = dict(
                messages=messages,
                **datapoint,
            )
        
        input_ids = [tokenized['input_ids'] for tokenized in all_tokenized]
        attention_mask = [tokenized['attention_mask'] for tokenized in all_tokenized]
        labels = [tokenized['labels'] for tokenized in all_tokenized]

        datapoint_ids = torch.tensor(datapoint_ids).long()
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        self.data = dict(
            datapoint_ids=datapoint_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        self.datapoints = [self.id_to_datapoint[datapoint_id] for datapoint_id in datapoint_ids.tolist()]
    
    def parse_output_text(self, output_text):
        return output_text
    
    def reward_fn(self, pred_text, datapoint):
        return None


class WildGuardTestDataset(BaseDataset):
    def __init__(self, num_datapoints, max_seq_length, tokenizer, dataset_split='eval', for_generation=False, **kwargs):
        super().__init__()
        self.num_datapoints = num_datapoints
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.dataset_split = dataset_split
        self.data = []
        self.id_to_datapoint = dict()
        self.model_name = kwargs.get('model_name', None)


        dataset = []
        with open('data/eval/wildguardtest.jsonl', 'r') as f:
            for line in f:
                dataset.append(json.loads(line.strip()))

        if num_datapoints == 'all':
            num_datapoints = len(dataset)
        num_datapoints = min(num_datapoints, len(dataset))

        datapoint_ids = []
        all_tokenized = []  
        for datapoint_idx, datapoint in tqdm(enumerate(dataset), total=num_datapoints):
            if len(all_tokenized) >= num_datapoints:
                break
            
            query = datapoint['prompt']
            messages = [dict(role='user', content=query)]

            if dataset_split == 'train' and not for_generation: 
                messages.append(dict(role='assistant', content='...'))

            tokenized = tokenize_messages(messages, tokenizer, max_seq_length, dataset_split, **kwargs)
            if tokenized is None:
                continue
            
            datapoint['targets'] = int(datapoint['label'] == 'harmful')

            all_tokenized.append(tokenized)
            datapoint_ids.append(datapoint_idx)
            self.id_to_datapoint[datapoint_idx] = dict(
                messages=messages,
                **datapoint,
            )
        
        input_ids = [tokenized['input_ids'] for tokenized in all_tokenized]
        attention_mask = [tokenized['attention_mask'] for tokenized in all_tokenized]
        labels = [tokenized['labels'] for tokenized in all_tokenized]

        datapoint_ids = torch.tensor(datapoint_ids).long()
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        self.data = dict(
            datapoint_ids=datapoint_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        self.datapoints = [self.id_to_datapoint[datapoint_id] for datapoint_id in datapoint_ids.tolist()]

    def parse_output_text(self, output_text):
        return output_text
    
    def reward_fn(self, pred_text, datapoint):
        return None


class MMLUOnPolicyDataset(BaseDataset):
    def __init__(self, num_datapoints='all', max_seq_length=4096, tokenizer=None, dataset_split='train', **kwargs):
        super().__init__()
        self.num_datapoints = num_datapoints
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.dataset_split = dataset_split
        self.data = []
        self.id_to_datapoint = dict()
        self.model_name = kwargs.get('model_name', None)

        data_dir = Path('data/self-sft')
        if 'Llama-3.2-1B-Instruct' in self.model_name:
            data_dir = data_dir / 'Llama-3.2-1B-Instruct'
        elif 'Llama-3.1-8B-Instruct' in self.model_name:
            data_dir = data_dir / 'Llama-3.1-8B-Instruct'
        elif 'Qwen2.5-1.5B-Instruct' in self.model_name:
            data_dir = data_dir / 'Qwen2.5-1.5B-Instruct'
        elif 'Qwen2.5-7B-Instruct' in self.model_name:
            data_dir = data_dir / 'Qwen2.5-7B-Instruct'
        else:
            raise ValueError(f'Model name {self.model_name} not supported')
        data_dir.mkdir(parents=True, exist_ok=True)

        data_path = data_dir / 'mmlu.jsonl'
        print(f'Loading data from: {data_path}')

        datapoints = []
        with open(data_path, 'r') as f:
            for line in f:
                datapoints.append(json.loads(line.strip()))

        if num_datapoints == 'all':
            num_datapoints = len(datapoints)
        num_datapoints = min(num_datapoints, len(datapoints))
        self.num_datapoints = num_datapoints

        self.tokenize_datapoints(datapoints, tokenizer, max_seq_length, self.model_name)
    
    def tokenize_datapoints(self, datapoints, tokenizer, max_seq_length, model_name):
        datapoint_ids = []
        all_tokenized = []
        for datapoint_id, prediction in tqdm(enumerate(datapoints), total=self.num_datapoints):
            if 'datapoint' not in prediction:
                datapoint = prediction
            else:
                datapoint = prediction['datapoint']

            if len(all_tokenized) >= self.num_datapoints:
                break

            if not prediction['correct']:
                continue

            on_policy_generated_text = prediction['output_text']
            messages = datapoint['messages']
            messages.append(dict(role='assistant', content=on_policy_generated_text))
            tokenized = tokenize_messages(messages, tokenizer, max_seq_length, model_name=model_name)
            if tokenized is None:
                continue

            all_tokenized.append(tokenized)
            datapoint_ids.append(datapoint_id)
            self.id_to_datapoint[datapoint_id] = datapoint

        input_ids = [tokenized['input_ids'] for tokenized in all_tokenized]
        attention_mask = [tokenized['attention_mask'] for tokenized in all_tokenized]
        labels = [tokenized['labels'] for tokenized in all_tokenized]

        datapoint_ids = torch.tensor(datapoint_ids).long()
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        self.data = dict(
            datapoint_ids=datapoint_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        self.datapoints = [self.id_to_datapoint[datapoint_id] for datapoint_id in datapoint_ids.tolist()]


class CountdownSFTDataset(BaseDataset):
    def __init__(self, num_datapoints, max_seq_length, tokenizer, dataset_split='eval', for_generation=False, **kwargs):
        super().__init__()

        self.num_datapoints = num_datapoints
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.dataset_split = dataset_split
        self.id_to_datapoint = dict()
        self.model_name = kwargs.get('model_name', None)

        data_dir = Path('data/sft')
        data_path = data_dir / 'countdown.jsonl'
        print(f'Loading data from: {data_path}')
        predictions = []
        with open(data_path, 'r') as f:
            for line in f:
                prediction = json.loads(line.strip())
                if not prediction['correct']:
                    continue
                predictions.append(prediction)

        if num_datapoints == 'all':
            num_datapoints = len(predictions)
        num_datapoints = min(num_datapoints, len(predictions))
        self.num_datapoints = num_datapoints

        datapoint_ids = []
        all_tokenized = []
        for datapoint_idx, prediction in tqdm(enumerate(predictions), total=num_datapoints):
            datapoint = prediction['datapoint']
            if len(all_tokenized) >= num_datapoints:
                break
            
            input_text = datapoint['input_text']
            target_text = datapoint['target_text']

            messages = [dict(role='user', content=input_text)]

            if _dataset_split == 'train' and not for_generation:
                messages.append(dict(role='assistant', content=target_text))

            tokenized = tokenize_messages(messages, tokenizer, max_seq_length, dataset_split, **kwargs)
            if tokenized is None:
                continue

            #parsed_output_expr = self.parse_output_text(target_text)
            expr = datapoint['expr']
            x = datapoint['x']
            y = datapoint['y']
            assert self.validate_expression(x, expr, y), f'{x} -> {expr} -> {y} failed'

            datapoint['targets'] = [(x, expr, y)]
            datapoint['messages'] = messages

            all_tokenized.append(tokenized)
            datapoint_ids.append(datapoint_idx)
            self.id_to_datapoint[datapoint_idx] = dict(
                **datapoint,
                prediction=prediction,
            )
        
        input_ids = [tokenized['input_ids'] for tokenized in all_tokenized]
        attention_mask = [tokenized['attention_mask'] for tokenized in all_tokenized]
        labels = [tokenized['labels'] for tokenized in all_tokenized]

        datapoint_ids = torch.tensor(datapoint_ids).long()
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        self.data = dict(
            datapoint_ids=datapoint_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        self.datapoints = [self.id_to_datapoint[datapoint_id] for datapoint_id in datapoint_ids.tolist()]

    def parse_output_text(self, output_text):
        if output_text is None:
            return None
        matches = re.findall(r'\\boxed{((?:[^{}]|{[^{}]*})*)}', output_text)
        if matches:
            return matches[-1].strip()
        return output_text

    def reward_fn(self, pred_text, datapoint):
        pred_expr = self.parse_output_text(pred_text)
        return all([self.validate_expression(x, pred_expr, y) for x, _, y in datapoint['targets']])

    def validate_expression(self, x_list, expr, target,
                            min_val=0, max_val=99,
                            check_intermediate=False):
        """
        Validate that `expr`:
          • uses exactly the numbers in `x_list`
          • employs only + - * /  (integer division)
          • lands on `target`
          • keeps results inside [min_val, max_val] if `check_intermediate` is True
        """
        OPS = {ast.Add: operator.add,
               ast.Sub: operator.sub,
               ast.Mult: operator.mul,
               ast.Div: operator.floordiv,
               ast.FloorDiv: operator.floordiv}
    
        try:
            tree = ast.parse(expr, mode='eval')
        except SyntaxError:
            return False
    
        used = Counter()
    
        def _eval(node):
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                op = OPS.get(type(node.op))
                if not op:
                    raise ValueError
                if op is operator.floordiv and (right == 0 or left % right):
                    raise ValueError
                result = op(left, right)
                if check_intermediate and not (min_val <= result <= max_val):
                    raise ValueError
                return result
            elif isinstance(node, ast.Constant) and isinstance(node.value, int):
                used[node.value] += 1
                return node.value
            else:
                raise ValueError
    
        try:
            final = _eval(tree.body)
        except (ValueError, AttributeError):
            return False

        return (final == target and
                min_val <= final <= max_val and
                used == Counter(x_list))


class CountdownDataset(BaseDataset):
    def __init__(self, num_datapoints, max_seq_length, tokenizer, dataset_split='eval', for_generation=False, **kwargs):
        super().__init__()

        self.num_datapoints = num_datapoints
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.dataset_split = dataset_split
        self.id_to_datapoint = dict()
        self.model_name = kwargs.get('model_name', None)

        if dataset_split == 'train':
            path = 'data/rl/countdown.jsonl'
        else:
            path = 'data/eval/countdown.jsonl'
        print(f'Loading data from: {path}')
        datapoints = []
        with open(path, 'r') as f:
            for line in f:
                datapoint = json.loads(line.strip())
                datapoints.append(datapoint)

        if num_datapoints == 'all':
            num_datapoints = len(datapoints)
        num_datapoints = min(num_datapoints, len(datapoints))
        self.num_datapoints = num_datapoints

        datapoint_ids = []
        all_tokenized = []
        for datapoint_idx, datapoint in tqdm(enumerate(datapoints), total=num_datapoints):
            if len(all_tokenized) >= num_datapoints:
                break
            
            input_text = datapoint['input_text']
            target_text = datapoint['target_text']

            messages = [dict(role='user', content=input_text)]

            if dataset_split == 'train' and not for_generation:
                messages.append(dict(role='assistant', content=target_text))

            tokenized = tokenize_messages(messages, tokenizer, max_seq_length, dataset_split, **kwargs)
            if tokenized is None:
                continue

            expr = datapoint['expr']
            x = datapoint['x']
            y = datapoint['y']
            assert self.validate_expression(x, expr, y), f'{x} -> {expr} -> {y} failed'

            datapoint['targets'] = [(x, expr, y)]

            all_tokenized.append(tokenized)
            datapoint_ids.append(datapoint_idx)
            self.id_to_datapoint[datapoint_idx] = dict(
                messages=messages,
                **datapoint,
            )
        
        input_ids = [tokenized['input_ids'] for tokenized in all_tokenized]
        attention_mask = [tokenized['attention_mask'] for tokenized in all_tokenized]
        labels = [tokenized['labels'] for tokenized in all_tokenized]

        datapoint_ids = torch.tensor(datapoint_ids).long()
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        self.data = dict(
            datapoint_ids=datapoint_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        self.datapoints = [self.id_to_datapoint[datapoint_id] for datapoint_id in datapoint_ids.tolist()]

    def parse_output_text(self, output_text):
        if output_text is None:
            return None
        matches = re.findall(r'\\boxed{((?:[^{}]|{[^{}]*})*)}', output_text)
        if matches:
            return matches[-1].strip()
        return output_text

    def reward_fn(self, pred_text, datapoint):
        pred_expr = self.parse_output_text(pred_text)
        return all([self.validate_expression(x, pred_expr, y) for x, _, y in datapoint['targets']])

    def validate_expression(self, x_list, expr, target,
                            min_val=0, max_val=99,
                            check_intermediate=False):
        """
        Validate that `expr`:
          • uses exactly the numbers in `x_list`
          • employs only + - * /  (integer division)
          • lands on `target`
          • keeps results inside [min_val, max_val] if `check_intermediate` is True
        """
        OPS = {ast.Add: operator.add,
               ast.Sub: operator.sub,
               ast.Mult: operator.mul,
               ast.Div: operator.floordiv,
               ast.FloorDiv: operator.floordiv}

        # Fast-fail on bad inputs
        if not isinstance(expr, str):
            return False
        if '\x00' in expr:
            return False
        expr = expr.strip()
        if not expr:
            return False

        try:
            tree = ast.parse(expr, mode='eval')
        except (SyntaxError, ValueError, TypeError):
            return False
    
        used = Counter()
    
        def _eval(node):
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                op = OPS.get(type(node.op))
                if not op:
                    raise ValueError
                if op is operator.floordiv and (right == 0 or left % right):
                    raise ValueError
                result = op(left, right)
                if check_intermediate and not (min_val <= result <= max_val):
                    raise ValueError
                return result
            elif isinstance(node, ast.Constant) and isinstance(node.value, int):
                used[node.value] += 1
                return node.value
            else:
                raise ValueError
    
        try:
            final = _eval(tree.body)
        except (ValueError, AttributeError):
            return False

        return (final == target and
                min_val <= final <= max_val and
                used == Counter(x_list))


class CountdownOnPolicyDataset(BaseDataset):
    def __init__(self, num_datapoints, max_seq_length, tokenizer, dataset_split='eval', for_generation=False, **kwargs):
        super().__init__()

        self.num_datapoints = num_datapoints
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.dataset_split = dataset_split
        self.id_to_datapoint = dict()
        self.model_name = kwargs.get('model_name', None)

        data_dir = Path('data/self-sft')
        if 'Llama-3.2-1B-Instruct' in self.model_name:
            data_dir = data_dir / 'Llama-3.2-1B-Instruct'
        elif 'Llama-3.1-8B-Instruct' in self.model_name:
            data_dir = data_dir / 'Llama-3.1-8B-Instruct'
        elif 'Qwen2.5-1.5B-Instruct' in self.model_name:
            data_dir = data_dir / 'Qwen2.5-1.5B-Instruct'
        elif 'Qwen2.5-7B-Instruct' in self.model_name:
            data_dir = data_dir / 'Qwen2.5-7B-Instruct'
        else:
            raise ValueError(f'Model name {self.model_name} not supported')
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / 'countdown.jsonl'
        print(f'Loading data from: {data_path}')

        _dataset_split = 'dev' if dataset_split == 'eval' else dataset_split

        predictions = []
        with open(data_path, 'r') as f:
            for line in f:
                prediction = json.loads(line.strip())
                if not prediction['correct']:
                    continue
                predictions.append(prediction)

        if num_datapoints == 'all':
            num_datapoints = len(predictions)
        num_datapoints = min(num_datapoints, len(predictions))
        self.num_datapoints = num_datapoints

        datapoint_ids = []
        all_tokenized = []
        for datapoint_idx, prediction in tqdm(enumerate(predictions), total=num_datapoints):
            datapoint = prediction['datapoint']
            if len(all_tokenized) >= num_datapoints:
                break
            
            input_text = datapoint['input_text']
            target_text = datapoint['target_text']

            messages = [dict(role='user', content=input_text)]

            if _dataset_split == 'train' and not for_generation:
                messages.append(dict(role='assistant', content=target_text))

            tokenized = tokenize_messages(messages, tokenizer, max_seq_length, dataset_split, **kwargs)
            if tokenized is None:
                continue

            #parsed_output_expr = self.parse_output_text(target_text)
            expr = datapoint['expr']
            x = datapoint['x']
            y = datapoint['y']
            assert self.validate_expression(x, expr, y), f'{x} -> {expr} -> {y} failed'

            datapoint['targets'] = [(x, expr, y)]
            datapoint['messages'] = messages

            all_tokenized.append(tokenized)
            datapoint_ids.append(datapoint_idx)
            self.id_to_datapoint[datapoint_idx] = dict(
                **datapoint,
                prediction=prediction,
            )
        
        input_ids = [tokenized['input_ids'] for tokenized in all_tokenized]
        attention_mask = [tokenized['attention_mask'] for tokenized in all_tokenized]
        labels = [tokenized['labels'] for tokenized in all_tokenized]

        datapoint_ids = torch.tensor(datapoint_ids).long()
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        self.data = dict(
            datapoint_ids=datapoint_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        self.datapoints = [self.id_to_datapoint[datapoint_id] for datapoint_id in datapoint_ids.tolist()]

    def parse_output_text(self, output_text):
        if output_text is None:
            return None
        matches = re.findall(r'\\boxed{((?:[^{}]|{[^{}]*})*)}', output_text)
        if matches:
            return matches[-1].strip()
        return output_text

    def reward_fn(self, pred_text, datapoint):
        pred_expr = self.parse_output_text(pred_text)
        return all([self.validate_expression(x, pred_expr, y) for x, _, y in datapoint['targets']])

    def validate_expression(self, x_list, expr, target,
                            min_val=0, max_val=99,
                            check_intermediate=False):
        """
        Validate that `expr`:
          • uses exactly the numbers in `x_list`
          • employs only + - * /  (integer division)
          • lands on `target`
          • keeps results inside [min_val, max_val] if `check_intermediate` is True
        """
        OPS = {ast.Add: operator.add,
               ast.Sub: operator.sub,
               ast.Mult: operator.mul,
               ast.Div: operator.floordiv,
               ast.FloorDiv: operator.floordiv}
    
        try:
            tree = ast.parse(expr, mode='eval')
        except SyntaxError:
            return False
    
        used = Counter()
    
        def _eval(node):
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                op = OPS.get(type(node.op))
                if not op:
                    raise ValueError
                if op is operator.floordiv and (right == 0 or left % right):
                    raise ValueError
                result = op(left, right)
                if check_intermediate and not (min_val <= result <= max_val):
                    raise ValueError
                return result
            elif isinstance(node, ast.Constant) and isinstance(node.value, int):
                used[node.value] += 1
                return node.value
            else:
                raise ValueError
    
        try:
            final = _eval(tree.body)
        except (ValueError, AttributeError):
            return False

        return (final == target and
                min_val <= final <= max_val and
                used == Counter(x_list))


class MMLUDataset(BaseDataset):
    def __init__(self, num_datapoints, max_seq_length, tokenizer, dataset_split='eval', for_generation=False, **kwargs):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.dataset_split = dataset_split
        self.data = []
        self.id_to_datapoint = dict()
        self.model_name = kwargs.get('model_name', None)

        dataset = []
        if dataset_split == 'train':
            data_path = data_dir / 'rl' / 'mmlu.jsonl'
            print(f'Loading data from: {data_path}')
            with open(data_path, 'r') as f:
                for line in f:
                    dataset.append(json.loads(line.strip()))
        else:
            data_path = data_dir / 'eval' / 'mmlu.jsonl'
            print(f'Loading data from: {data_path}')
            with open(data_path, 'r') as f:
                for line in f:
                    dataset.append(json.loads(line.strip()))

        if num_datapoints == 'all':
            num_datapoints = len(dataset)
        self.num_datapoints = num_datapoints

        datapoint_ids = []
        all_tokenized = []
        for datapoint_idx, datapoint in tqdm(enumerate(dataset), total=num_datapoints):
            if len(all_tokenized) >= self.num_datapoints:
                break
            subject = datapoint['subject']
            question = datapoint['question']
            choices = datapoint['choices']
            answer = datapoint['answer']
            answer_option = list('ABCD')[answer]
            choices_text = '\n'.join([f'{c}. {s}' for c, s in zip(list('ABCD'), choices)])
            user_content = f'{question}\n\nAnswer options:\n{choices_text}\n\nReason about it and answer with "The answer is: <option>"'
            messages = [dict(role='user', content=user_content)]
            if dataset_split == 'train' and not for_generation:
                messages.append(dict(role='assistant', content=answer_option))
            tokenized = tokenize_messages(messages, tokenizer, max_seq_length, dataset_split, **kwargs)
            if tokenized is None:
                continue
            all_tokenized.append(tokenized)
            datapoint_ids.append(datapoint_idx)
            self.id_to_datapoint[datapoint_idx] = dict(
                messages=messages,
                targets=[answer_option],
                **datapoint,
            )

        input_ids = [tokenized['input_ids'] for tokenized in all_tokenized]
        attention_mask = [tokenized['attention_mask'] for tokenized in all_tokenized]
        labels = [tokenized['labels'] for tokenized in all_tokenized]

        datapoint_ids = torch.tensor(datapoint_ids).long()
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        self.data = dict(
            datapoint_ids=datapoint_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        self.datapoints = [self.id_to_datapoint[datapoint_id] for datapoint_id in datapoint_ids.tolist()]

    def parse_output_text(self, output_text):
        if output_text is None:
            return None
        
        output_text = output_text.replace('*', '')
        match = re.search(r'\bThe(.*)answer is(?: option)?:?\s*(\w+)', output_text, re.IGNORECASE)
        if match:
            return match.group(2).upper().strip()
        else:
            return None

    def print_accuracy(self, corrects):
        print('### Accuracy ###')
        num_total_correct = 0
        num_total = 0
        for category, category_corrects in corrects.items():
            num_category_correct = sum(category_corrects)
            num_category_total = len(category_corrects)
            print(f'{category}: {num_category_correct / num_category_total:.2f}')
    
            num_total_correct += num_category_correct
            num_total += num_category_total
        print(f'Average: {num_total_correct / num_total:.2f}')
        print('################')

    def reward_fn(self, pred_text, datapoint):
        targets = datapoint['targets']
        return pred_text in targets


class MMLUSFTDataset(BaseDataset):
    def __init__(self, num_datapoints, max_seq_length, tokenizer, dataset_split='train', for_generation=False, **kwargs):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.dataset_split = dataset_split
        self.data = []
        self.id_to_datapoint = dict()
        self.model_name = kwargs.get('model_name', None)

        dataset = []
        path = 'data/sft/mmlu.jsonl'
        print(f'Loading data from: {path}')
        with open(path, 'r') as f:
            for line in f:
                datapoint = json.loads(line.strip())
                if not datapoint['correct']:
                    continue
                dataset.append(datapoint)

        if num_datapoints == 'all':
            num_datapoints = len(dataset)
        num_datapoints = min(num_datapoints, len(dataset))
        dataset = dataset[:num_datapoints]
        
        datapoint_ids = []
        all_tokenized = []
        for datapoint_idx, datapoint in tqdm(enumerate(dataset), total=num_datapoints):
            if len(all_tokenized) >= num_datapoints:
                break

            messages = datapoint['messages']
            messages.append(dict(role='assistant', content=datapoint['output_text'].strip()))
            tokenized = tokenize_messages(messages, tokenizer, max_seq_length, dataset_split, **kwargs)
            if tokenized is None:
                continue
            
            datapoint['targets'] = [None]
                
            all_tokenized.append(tokenized)
            datapoint_ids.append(datapoint_idx)
            self.id_to_datapoint[datapoint_idx] = dict(**datapoint)
        
        input_ids = [tokenized['input_ids'] for tokenized in all_tokenized]
        attention_mask = [tokenized['attention_mask'] for tokenized in all_tokenized]
        labels = [tokenized['labels'] for tokenized in all_tokenized]

        datapoint_ids = torch.tensor(datapoint_ids).long()
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        self.data = dict(
            datapoint_ids=datapoint_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,  
            labels=labels,
        )
        self.datapoints = [self.id_to_datapoint[datapoint_id] for datapoint_id in datapoint_ids.tolist()]


class IFEvalDataset(BaseDataset):
    def __init__(self, num_datapoints, max_seq_length, tokenizer, dataset_split='train', for_generation=False, **kwargs):
        super().__init__()
        from core.evaluation.ifeval_utils import IF_FUNCTIONS_MAP
        self.if_functions_map = IF_FUNCTIONS_MAP
        self.num_datapoints = num_datapoints
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.dataset_split = dataset_split
        self.data = []
        self.id_to_datapoint = dict()
        self.model_name = kwargs.get('model_name', None)

        dataset = []
        if dataset_split == 'train':
            print(f'Loading data from: data/rl/ifeval.jsonl')
            with open('data/rl/ifeval.jsonl', 'r') as f:
                for line in f:
                    dataset.append(json.loads(line.strip()))
        else:
            print(f'Loading data from: data/eval/ifeval.jsonl')
            with open('data/eval/ifeval.jsonl', 'r') as f:
                for line in f:
                    dataset.append(json.loads(line.strip()))

        if num_datapoints == 'all':
            num_datapoints = len(dataset)
        num_datapoints = min(num_datapoints, len(dataset))
        dataset = dataset[:num_datapoints]

        datapoint_ids = []
        all_tokenized = []
        for datapoint_idx, datapoint in tqdm(enumerate(dataset), total=num_datapoints):
            if len(all_tokenized) >= num_datapoints:
                break

            messages = datapoint['messages']
            if dataset_split == 'train' and not for_generation:
                # TODO: handle this
                messages.append(dict(role='assistant', content='...'))
            tokenized = tokenize_messages(messages, tokenizer, max_seq_length, dataset_split, **kwargs)
            if tokenized is None:
                continue
            
            datapoint['targets'] = json.loads(datapoint['ground_truth'])

            all_tokenized.append(tokenized)
            datapoint_ids.append(datapoint_idx)
            self.id_to_datapoint[datapoint_idx] = dict(**datapoint)
        
        input_ids = [tokenized['input_ids'] for tokenized in all_tokenized]
        attention_mask = [tokenized['attention_mask'] for tokenized in all_tokenized]
        labels = [tokenized['labels'] for tokenized in all_tokenized]

        datapoint_ids = torch.tensor(datapoint_ids).long()
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        self.data = dict(
            datapoint_ids=datapoint_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        self.datapoints = [self.id_to_datapoint[datapoint_id] for datapoint_id in datapoint_ids.tolist()]

    def parse_output_text(self, output_text):
        return output_text

    def reward_fn(self, pred_text, datapoint):
        func_name = datapoint['targets']['func_name']
        func = self.if_functions_map[func_name]
        try:
            reward = func(pred_text, **datapoint['targets'])
        except Exception as e:
            print(e)
            print(pred_text)
            print(datapoint['targets'])
            reward = False
        return reward


class IFEvalOnPolicyDataset(BaseDataset):
    def __init__(self, num_datapoints='all', max_seq_length=4096, tokenizer=None, dataset_split='train', **kwargs):
        super().__init__()
        self.num_datapoints = num_datapoints
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.dataset_split = dataset_split
        self.id_to_datapoint = dict()
        self.model_name = kwargs.get('model_name', None)

        data_dir = Path('data/self-sft')
        if 'Llama-3.2-1B-Instruct' in self.model_name:
            data_dir = data_dir / 'Llama-3.2-1B-Instruct'
        elif 'Llama-3.1-8B-Instruct' in self.model_name:
            data_dir = data_dir / 'Llama-3.1-8B-Instruct'
        elif 'Qwen2.5-1.5B-Instruct' in self.model_name:
            data_dir = data_dir / 'Qwen2.5-1.5B-Instruct'
        elif 'Qwen2.5-7B-Instruct' in self.model_name:
            data_dir = data_dir / 'Qwen2.5-7B-Instruct'
        else:
            raise ValueError(f'Model name {self.model_name} not supported')

        datapoints = []
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / 'ifeval.jsonl'
        print(f'Loading data from: {data_path}')
        with open(data_path, 'r') as f:
            for line in f:
                datapoints.append(json.loads(line.strip()))

        if num_datapoints == 'all':
            num_datapoints = len(datapoints)
        num_datapoints = min(num_datapoints, len(datapoints))
        self.num_datapoints = num_datapoints

        self.tokenize_datapoints(datapoints, tokenizer, max_seq_length, self.model_name)

    def tokenize_datapoints(self, datapoints, tokenizer, max_seq_length, model_name):
        datapoint_ids = []
        all_tokenized = []
        for datapoint_id, prediction in tqdm(enumerate(datapoints), total=self.num_datapoints):
            if 'datapoint' not in prediction:
                datapoint = prediction
            else:
                datapoint = prediction['datapoint']

            if len(all_tokenized) >= self.num_datapoints:
                break

            if not prediction['correct']:
                continue

            on_policy_generated_text = prediction['output_text']
            messages = datapoint['messages']
            messages.append(dict(role='assistant', content=on_policy_generated_text))
            tokenized = tokenize_messages(messages, tokenizer, max_seq_length, model_name=model_name)
            if tokenized is None:
                continue

            all_tokenized.append(tokenized)
            datapoint_ids.append(datapoint_id)
            self.id_to_datapoint[datapoint_id] = datapoint

        actual_num_datapoints = len(all_tokenized)        

        ratio = 0.1
        self.num_datapoints = int(actual_num_datapoints * ratio)
        all_tokenized = all_tokenized[:self.num_datapoints]
        datapoint_ids = datapoint_ids[:self.num_datapoints]

        input_ids = [tokenized['input_ids'] for tokenized in all_tokenized]
        attention_mask = [tokenized['attention_mask'] for tokenized in all_tokenized]
        labels = [tokenized['labels'] for tokenized in all_tokenized]

        datapoint_ids = torch.tensor(datapoint_ids).long()
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        self.data = dict(
            datapoint_ids=datapoint_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        self.datapoints = [self.id_to_datapoint[datapoint_id] for datapoint_id in datapoint_ids.tolist()]
    

class IFEvalSFTDataset(BaseDataset):
    def __init__(self, num_datapoints, max_seq_length, tokenizer, dataset_split='train', for_generation=False, **kwargs):
        super().__init__()
        self.num_datapoints = num_datapoints
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.dataset_split = dataset_split
        self.data = []
        self.id_to_datapoint = dict()
        self.model_name = kwargs.get('model_name', None)

        data_dir = Path('data/sft')
        dataset = []
        data_path = data_dir / 'ifeval.jsonl'
        print(f'Loading data from: {data_path}')
        with open(data_path, 'r') as f:
            for line in f:
                datapoint = json.loads(line.strip())
                if not datapoint['correct']:
                    continue
                dataset.append(datapoint)

        if num_datapoints == 'all':
            num_datapoints = len(dataset)
        num_datapoints = min(num_datapoints, len(dataset))
        dataset = dataset[:num_datapoints]
        
        datapoint_ids = []
        all_tokenized = []
        for datapoint_idx, datapoint in tqdm(enumerate(dataset), total=num_datapoints):
            if len(all_tokenized) >= num_datapoints:
                break

            messages = datapoint['messages']
            messages.append(dict(role='assistant', content=datapoint['output_text'].strip()))
            tokenized = tokenize_messages(messages, tokenizer, max_seq_length, dataset_split, **kwargs)
            if tokenized is None:
                continue
            
            datapoint['targets'] = [None]
                
            all_tokenized.append(tokenized)
            datapoint_ids.append(datapoint_idx)
            self.id_to_datapoint[datapoint_idx] = dict(**datapoint)
        
        input_ids = [tokenized['input_ids'] for tokenized in all_tokenized]
        attention_mask = [tokenized['attention_mask'] for tokenized in all_tokenized]
        labels = [tokenized['labels'] for tokenized in all_tokenized]

        datapoint_ids = torch.tensor(datapoint_ids).long()
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        self.data = dict(
            datapoint_ids=datapoint_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,  
            labels=labels,
        )
        self.datapoints = [self.id_to_datapoint[datapoint_id] for datapoint_id in datapoint_ids.tolist()]
