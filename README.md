# Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting

Repository for paper: [Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting](https://arxiv.org/abs/2510.18874)

<img src="assets/teaser.png" width="700">

## Data

Download and extract the data to the `data/` directory:

[Link to data](https://drive.google.com/file/d/1IBhBH5TdVp1gxeg3BT8kgZ1j8PyCGI1l/view?usp=sharing)

The data is organized as follows:
```
data/
├── sft/                # Expert SFT data
│   ├── ifeval.jsonl
│   ├── mmlu.jsonl
│   └── countdown.jsonl
├── self-sft/           # On-policy (Self-SFT) data per model
│   ├── Llama-3.2-1B-Instruct/
│   ├── Llama-3.1-8B-Instruct/
│   ├── Qwen2.5-1.5B-Instruct/
│   └── Qwen2.5-7B-Instruct/
├── rl/                 # RL training data
│   ├── ifeval.jsonl
│   ├── mmlu.jsonl
│   └── countdown.jsonl
└── eval/               # Evaluation data
    ├── ifeval.jsonl
    ├── mmlu.jsonl
    ├── countdown.jsonl
    ├── math.jsonl
    ├── wildjailbreak.jsonl
    └── wildguardtest.jsonl
```


## Training
`model_name_short`: Choose from 
  * `llama-3.2-1b-inst`
  * `llama-3.1-8b-inst`
  * `qwen-2.5-1.5b-inst`
  * `qwen-2.5-7b-inst`

### SFT
```bash
bash scripts/sft.sh
```
Configure the script by modifying:
Set `dataset_name` to one of the following:
- `IFEvalSFTDataset`
- `MMLUSFTDataset`
- `CountdownSFTDataset`

### Self-SFT
```bash
bash scripts/self_sft.sh
```
Configure the script by modifying:
Set `dataset_name` to one of the following:
- `IFEvalOnPolicyDataset`
- `MMLUOnPolicyDataset`
- `CountdownOnPolicyDataset`

### RL
```bash
bash scripts/rl.sh
```
Configure the script by modifying:
Set `dataset_name` to one of the following:
- `IFEvalDataset`
- `MMLUDataset`
- `CountdownDataset`


## Evaluation
```bash
bash scripts/eval.sh
```
Configure the script by modifying:
- `model_names`: List of model paths to evaluate
- `dataset_name_shorts`: Choose from `ifeval_verify`, `mmlu`, `countdown`, `wildjailbreak`, `wildguardtest`, `math`
- `dataset_split`: Dataset split to evaluate on (default: `eval`)

### Evaluation with Judge (Safety)
```bash
bash scripts/eval_with_judge.sh
```
Configure the script by modifying:
- `model_names`: List of model paths to evaluate
- `judge_model_name`: Path to the judge model (default: `allenai/wildguard`)
- `dataset_name_shorts`: Choose from `wildjailbreak`, `wildguardtest`
- `dataset_split`: Dataset split to evaluate on (default: `eval`)


## Simulation
```bash
# Uni-modal 
python -m simulation.run_single_mode --gain_thresholds "0.9,0.9" --save pdf

# Bi-modal
python -m simulation.run --gain_thresholds "0.9,0.9,0.9" --save pdf
```

## Citation
```
@article{chen2025retaining,
  title={Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting},
  author={Chen, Howard and Razin, Noam and Narasimhan, Karthik and Chen, Danqi},
  journal={arXiv preprint arXiv:2510.18874},
  year={2025}
}
```
