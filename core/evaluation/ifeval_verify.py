from pathlib import Path
from functools import partial

import fire
from tqdm import tqdm
from rich import print

import torch
from core.evaluation.utils import setup_model_for_inference
from core.evaluation.utils import GenerationManager
from core.data import apply_chat_template
from core.utils import load_module, print_gpu_usage


@torch.no_grad()
def run(
    model_name,
    run_dir='none',
    subdir_name='none',
    batch_size=32,
    num_return_sequences=1,
    temperature=0.0,
    dataset_split='eval',
    model_module_path='transformers.AutoModelForCausalLM',
):
    model, tokenizer = setup_model_for_inference(model_name, model_module_path=model_module_path)

    base_dir = Path(model_name) if run_dir == 'none' else Path(run_dir)
    if subdir_name == 'none':
        run_dir = base_dir / 'ifeval_verify' / dataset_split
    else:
        run_dir = base_dir / 'ifeval_verify' / dataset_split / subdir_name
    tokenizer.apply_chat_template = apply_chat_template

    print(f'model_name={model_name}')
    print(f'run_dir={run_dir}')

    dataset_module = load_module('core.data', 'Tulu3RLVRIFEvalDataset')
    dataset = dataset_module(
        num_datapoints='all',
        max_seq_length=8192,
        tokenizer=tokenizer,
        dataset_split=dataset_split,
        for_generation=True,
    )
    dataloader = dataset.get_eval_batcher(
        batch_size=batch_size,
        chat_template_fn=partial(
            tokenizer.apply_chat_template,
            tokenize=False,
            add_generation_prompt=True,
        ),
    )

    generation_config = dict(
        max_new_tokens=1500,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        do_sample=temperature > 0.0,
        temperature=temperature,
        top_p=1.0,
        num_return_sequences=num_return_sequences,
        stop_strings=['<|end_of_text|>', '<|eot_id|>'],
    )
    gen_manager = GenerationManager(run_dir, print_to_stdout=True, overwrite=True, dry_run=False)
    gen_manager.save_generation_config(generation_config)

    datapoint_idx = 0
    corrects = []
    for batch, batch_datapoints in dataloader:
        batch_input_texts = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False)
        inputs = {k: v.cuda() for k, v in batch.items()}

        # NOTE: batch_output_ids has shape (batch_size, num_return_sequences, seq_length)
        batch_output_ids = model.generate(**inputs, **generation_config, tokenizer=tokenizer)

        for input_text, output_ids, datapoint in zip(batch_input_texts, batch_output_ids, batch_datapoints):
            output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
            for output_text in output_texts:
                output_text = output_text[len(input_text):].replace(tokenizer.eos_token, '').strip()
                pred_text = dataset.parse_output_text(output_text)
                _input_text = input_text.replace(tokenizer.eos_token, '').strip()
                target_text = datapoint['targets']

                gen_manager.write_log(f'model_name={model_name}')
                gen_manager.write_log(f'run_dir={run_dir}')
                gen_manager.write_log(f'datapoint_idx={datapoint_idx} / {len(dataset.datapoints)}')
                gen_manager.write_log('### Input Text ###')
                gen_manager.write_log(_input_text)
                gen_manager.write_log('---')
                gen_manager.write_log('### Output Text ###')
                gen_manager.write_log(output_text)
                gen_manager.write_log('---')
                gen_manager.write_log('### Pred Text ###')
                gen_manager.write_log(pred_text)
                gen_manager.write_log('---')
                gen_manager.write_log('### Target ###')
                gen_manager.write_log(target_text)
                gen_manager.write_log('---')
    
                correct = dataset.reward_fn(pred_text, datapoint)
                corrects.append(correct)
                gen_manager.write_log(f'correct={correct}')
                gen_manager.write_log(f'Accuracy: {sum(corrects) / len(corrects):.4f}')
                gen_manager.write_log('#' * 100)
                generated = dict(
                    datapoint_idx=datapoint_idx,
                    **datapoint,
                    input_text=_input_text,
                    output_text=output_text,
                    pred_text=pred_text,
                    correct=correct,
                )
                gen_manager.write_prediction(generated)
            datapoint_idx += 1
    gen_manager.save_metrics(dict(accuracy=sum(corrects) / len(corrects)))


if __name__ == '__main__':
    fire.Fire(run)
