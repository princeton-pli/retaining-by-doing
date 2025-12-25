"""
python -m core.vllm_utils
"""
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from rich import print


class vLLMCausalLM:
    def __init__(self, model=None):
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model_name,
        device_map=None,
        local_files_only=True,
        max_seq_length=128000,
    ):
        num_gpus = torch.cuda.device_count() if device_map == 'auto' else 1
        #num_gpus = 1
        model = LLM(model_name, tensor_parallel_size=num_gpus)
        return cls(model)
    
    def generate(
        self,
        input_texts,
        max_new_tokens=1600,
        eos_token_id=None,
        pad_token_id=None,
        use_cache=True,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        num_return_sequences=1,
        stop_strings=['<|end_of_text|>', '<|eot_id|>'],
        tokenizer=None,
        **kwargs,
    ):
        assert tokenizer is not None

        temperature = 0.0 if not do_sample else temperature
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            n=num_return_sequences,
            stop=stop_strings + ['<|eot_id|>'],
            logprobs=1,
        )
        _outputs = self.model.generate(input_texts, sampling_params, use_tqdm=False)
        outputs = [
            dict(
                input_text=o.prompt,
                generated_text=[o.outputs[i].text for i in range(num_return_sequences)],
                finish_reason=[o.outputs[i].finish_reason for i in range(num_return_sequences)],
                generated_tokens=[[list(logprob_token_dict.values())[0].decoded_token for logprob_token_dict in o.outputs[i].logprobs] for i in range(num_return_sequences)],
                logprobs=[[list(logprob_token_dict.values())[0].logprob for logprob_token_dict in o.outputs[i].logprobs] for i in range(num_return_sequences)],
            ) for o in _outputs
        ]
        return outputs


if __name__ == '__main__':
    from core.utils import load_module
    model_name = 'meta-llama/Llama-3.2-1B-Instruct'
    model_module = load_module('core.vllm_utils', 'vLLMCausalLM')
    model = model_module.from_pretrained(
        model_name,
        #device_map='auto',
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, add_prefix_space=True, padding_side='left')
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    generation_config = dict(
        max_new_tokens=1500,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        num_return_sequences=1,
        stop_strings=['<|end_of_text|>', '<|eot_id|>'],
    )    

    prompts = [
        [dict(role='user', content='Hello, how are you?')],
        [dict(role='user', content='What is 2 + 2?')],
        [dict(role='user', content='Who is the president of the United States?')],
    ]
    input_texts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]
    inputs = tokenizer.batch_encode_plus(
        input_texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=8192,
        add_special_tokens=False,
    )
    output_ids = model.generate(**inputs, **generation_config, tokenizer=tokenizer)
    output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=False)

    for input_text, output_text in zip(input_texts, output_texts):
        output_text = output_text[len(input_text):].replace(tokenizer.eos_token, '').strip()
        input_text = input_text.replace(tokenizer.eos_token, '').strip()
        print(input_text)
        print('---')
        print(output_text)
        print('#' * 100)
