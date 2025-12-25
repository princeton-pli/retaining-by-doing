#!/bin/bash -l

conda activate py310n
root_dir="/path/to/your/project/dir"
cd ${root_dir}

# Set variables ###############################################################
run_dir="/path/to/run/dir"
model_names=(
    "/path/to/model/dir"
)
###############################################################################

model_module_path="core.vllm_utils.vLLMCausalLM"
dataset_name_shorts=("wildjailbreak" "wildguardtest")
dataset_split="eval"
judge_model_name="/path/to/allenai/wildguard"


# loop over
for model_name in "${model_names[@]}"; do
    for dataset_name_short in "${dataset_name_shorts[@]}"; do
        echo "Model: ${model_name}"
        echo "Judge model: ${judge_model_name}"
        echo "Dataset: ${dataset_name_short}"
        echo "Dataset split: ${dataset_split}"
        echo "Run dir: ${run_dir}"
        echo "Model module path: ${model_module_path}"
        
        python -m core.evaluation.run_judge \
            --model_name "${model_name}" \
            --judge_model_name "${judge_model_name}" \
            --dataset_name "${dataset_name_short}" \
            --run_dir "${run_dir}" \
            --model_module_path "${model_module_path}" \
            --dataset_split "${dataset_split}"
    done 
done
