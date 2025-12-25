#!/bin/bash -l

num_gpus=4
export OMP_NUM_THREADS=${num_gpus}
master_addr=localhost
master_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# TODO: set dir paths #########################################################
conda activate py310n
root_dir="/path/to/your/project/dir"
cd ${root_dir}

run_base_dir="/path/to/your/run/dir"
###############################################################################

# TODO: choose the model ######################################################
model_name_short="llama-3.2-1b-inst"
#model_name_short="llama-3.1-8b-inst"
#model_name_short="qwen-2.5-1.5b-inst"
#model_name_short="qwen-2.5-7b-inst"
###############################################################################

# TODO: choose the dataset ####################################################
dataset_name="IFEvalSFTDataset"
#dataset_name="MMLUSFTDataset"
#dataset_name="CountdownSFTDataset"
###############################################################################

# TODO: set your local model paths here #######################################
if [[ "${model_name_short}" == "llama-3.2-1b-inst" ]]; then
    model_name="/path/to/your/local_models/meta-llama/Llama-3.2-1B-Instruct"
elif [[ "${model_name_short}" == "llama-3.1-8b-inst" ]]; then
    model_name="/path/to/your/local_models/meta-llama/Meta-Llama-3.1-8B-Instruct"
elif [[ "${model_name_short}" == "qwen-2.5-1.5b-inst" ]]; then
    model_name="/path/to/your/local_models/Qwen/Qwen2.5-1.5B-Instruct"
elif [[ "${model_name_short}" == "qwen-2.5-7b-inst" ]]; then
    model_name="/path/to/your/local_models/Qwen/Qwen2.5-7B-Instruct"
fi
###############################################################################

ref_model_name="${model_name}"
ref_model_name_short="${model_name_short}"

seed=0
wandb_project="none"
data_module_path="core.data"

if [[ "${model_name_short}" == "llama-3.1-8b-inst" || "${model_name_short}" == "qwen-2.5-7b-inst" ]]; then 
    lr=5e-5
else
    lr=1e-4
fi

num_epochs=2
warmup_ratio=0.03
scheduler_type="cosine"
gradient_clipping=1.0
num_train_datapoints="all"
num_eval_datapoints=100
eval_batch_size_per_device=2
train_batch_size_per_device=2
gradient_accumulation_steps=32


eval_every_n_steps=5000
save_every_n_steps=10000000


max_seq_length=4096
kl_beta=0.0

show_gpu_usage=false

model_module_path="transformers.AutoModelForCausalLM"

if [[ "${model_name_short}" == *"llama"* ]]; then 
    transformer_layer_module_path="transformers.models.llama.modeling_llama.LlamaDecoderLayer"
elif [[ "${model_name_short}" == *"gemma"* ]]; then
    transformer_layer_module_path="transformers.models.gemma2.modeling_gemma2.Gemma2DecoderLayer"
elif [[ "${model_name_short}" == *"qwen"* ]]; then
    transformer_layer_module_path="transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer"
fi

loss_module_path="core.training.objectives.CrossEntropyLoss"

total_batch_size=$((train_batch_size_per_device * num_gpus * gradient_accumulation_steps))


run_name="algo=sft_data=${dataset_name}_m=${model_name_short}_n=${num_train_datapoints}_epoch=${num_epochs}_bsz=${total_batch_size}_lr=${lr}_klbeta=${kl_beta}_seed=${seed}"


command="torchrun --rdzv-backend=c10d --rdzv-endpoint=${master_addr}:${master_port} --nnodes=1 --nproc-per-node=${num_gpus} -m core.training.sft"
args=(
    --run_name ${run_name} \
    --run_base_dir ${run_base_dir} \
    --seed ${seed} \
    --model_name ${model_name} \
    --ref_model_name ${ref_model_name} \
    --kl_beta ${kl_beta} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --lr ${lr} \
    --num_epochs ${num_epochs} \
    --warmup_ratio ${warmup_ratio} \
    --scheduler_type ${scheduler_type} \
    --gradient_clipping ${gradient_clipping} \
    --eval_every_n_steps ${eval_every_n_steps} \
    --save_every_n_steps ${save_every_n_steps} \
    --model_module_path ${model_module_path} \
    --transformer_layer_module_path ${transformer_layer_module_path} \
    --loss_module_path ${loss_module_path} \
    --data_module_path ${data_module_path} \
    --dataset_name ${dataset_name} \
    --num_train_datapoints ${num_train_datapoints} \
    --num_eval_datapoints ${num_eval_datapoints} \
    --train_batch_size_per_device ${train_batch_size_per_device} \
    --eval_batch_size_per_device ${eval_batch_size_per_device} \
    --total_batch_size ${total_batch_size} \
    --max_seq_length ${max_seq_length} \
    --show_gpu_usage ${show_gpu_usage} \
    --wandb_project ${wandb_project}
)


${command} "${args[@]}"

