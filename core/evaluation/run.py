"""
./scripts/eval.sh
"""
import re
import math
from pathlib import Path
from functools import partial

import fire
from tqdm import tqdm
from rich import print

import torch
from core.evaluation.utils import setup_model_for_inference
from core.evaluation.utils import GenerationManager
from core.utils import load_module, print_gpu_usage

DATASET_NAME_TO_MODULE = {
    'ifeval_verify': 'core.data.IFEvalDataset',
    'mmlu': 'core.data.MMLUDataset',
    'countdown': 'core.data.CountdownDataset',
    'wildjailbreak': 'core.data.WildJailbreakDataset',
    'wildguardtest': 'core.data.WildGuardTestDataset',
    'math': 'core.data.MATHDataset',
}


@torch.no_grad()
def run(
    model_name,
    dataset_name,
    run_dir='none',
    subdir_name='none',
    batch_size=32,
    num_return_sequences=1,
    temperature=0.0,
    dataset_split='eval',
    model_module_path='transformers.AutoModelForCausalLM',
    print_to_stdout=False,
):
    model, tokenizer = setup_model_for_inference(model_name, model_module_path=model_module_path)

    base_dir = Path(model_name) if run_dir == 'none' else Path(run_dir)
    run_dir = base_dir / dataset_name / dataset_split
    if subdir_name != 'none':
        run_dir = run_dir / subdir_name

    print(f'model_name={model_name}')
    print(f'run_dir={run_dir}')

    dataset_module = load_module(DATASET_NAME_TO_MODULE[dataset_name])
    dataset = dataset_module(
        num_datapoints='all',
        max_seq_length=2000,
        tokenizer=tokenizer,
        dataset_split=dataset_split,
        for_generation=True,
        model_name=model_name,
    )
    dataloader = dataset.get_eval_batch_input_texts(batch_size=batch_size)

    generation_config = dict(
        max_new_tokens=4096,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        do_sample=temperature > 0.0,
        temperature=temperature,
        top_p=1.0,
        num_return_sequences=num_return_sequences,
        stop_strings=[tokenizer.eos_token],
    )
    gen_manager = GenerationManager(run_dir, print_to_stdout=print_to_stdout, overwrite=True, dry_run=False)
    gen_manager.save_generation_config(generation_config)

    datapoint_idx = 0
    corrects = []

    num_batches = math.ceil(len(dataset.datapoints) / batch_size)
    for batch_input_texts, batch_datapoints in tqdm(dataloader, total=num_batches):
        outputs = model.generate(batch_input_texts, **generation_config, tokenizer=tokenizer)
        for output, datapoint in zip(outputs, batch_datapoints):
            input_text = output['input_text']

            output_texts = output['generated_text']
            finish_reasons = output['finish_reason']

            for output_text, finish_reason in zip(output_texts, finish_reasons):
                generated = dict()
                pred_text = dataset.parse_output_text(output_text)
                target_text = datapoint['targets']
                correct = dataset.reward_fn(pred_text, datapoint)
    
                gen_manager.write_log(f'model_name={model_name}')
                gen_manager.write_log(f'run_dir={run_dir}')
                gen_manager.write_log(f'datapoint_idx={datapoint_idx} / {len(dataset.datapoints)}')
                gen_manager.write_log('### Input Text ###')
                gen_manager.write_log(input_text)
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
                gen_manager.write_log(f'finish_reason={finish_reason}')
                gen_manager.write_log('---')
    
                corrects.append(correct)
                gen_manager.write_log(f'correct={correct}')
                if correct is not None:
                    gen_manager.write_log(f'Accuracy: {sum(corrects) / len(corrects):.4f}')
                gen_manager.write_log('#' * 100)
                generated = dict(
                    generated=generated,
                    datapoint=datapoint,
                    datapoint_idx=datapoint_idx,
                    input_text=input_text,
                    output_text=output_text,
                    pred_text=pred_text,
                    correct=correct,
                    finish_reason=finish_reason,
                )
                gen_manager.write_prediction(generated)
            datapoint_idx += 1

    if all([correct is None for correct in corrects]):
        gen_manager.save_metrics(dict(accuracy='To Be Calculated'))
    else:
        gen_manager.save_metrics(dict(accuracy=sum(corrects) / len(corrects)))
        print(f'Accuracy: {sum(corrects) / len(corrects):.4f}')


if __name__ == '__main__':
    fire.Fire(run)
