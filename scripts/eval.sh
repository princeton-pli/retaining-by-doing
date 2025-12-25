#!/bin/bash -l

conda activate py310n
root_dir="/path/to/your/project/dir"
cd ${root_dir}


run_dir="/path/to/your/run/dir"
model_names=(
    "/path/to/your/model/path"
)


model_module_path="core.vllm_utils.vLLMCausalLM"
dataset_name_shorts=(
    "ifeval_verify"
    "mmlu"
    "countdown"
    "wildjailbreak"
    "wildguardtest"
    "math"
)

batch_size=128
dataset_split="eval"
print_to_stdout=True
temperature=0.0


for model_name in "${model_names[@]}"; do
    for dataset_name_short in "${dataset_name_shorts[@]}"; do
        echo "Model: ${model_name}"
        echo "Dataset: ${dataset_name_short}"
        echo "Dataset split: ${dataset_split}"
        echo "Run dir: ${run_dir}"
        echo "Model module path: ${model_module_path}"
        
        python -m core.evaluation.run \
            --model_name "${model_name}" \
            --dataset_name "${dataset_name_short}" \
            --run_dir "${run_dir}" \
            --batch_size "${batch_size}" \
            --model_module_path "${model_module_path}" \
            --dataset_split "${dataset_split}" \
            --temperature "${temperature}" 
    done 
done
