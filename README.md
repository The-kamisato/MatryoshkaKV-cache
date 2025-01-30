# MatryoshkaKV cache: Adaptive KV compression via Trainable orthogonal projection

![architecture](https://github.com/The-kamisato/MatryoshkaKV-cache/blob/main/figure/architecture.jpg)

Code for [MAtryoshkaKV-cache](https://arxiv.org/abs/2410.14731).

This project delivered LLaMA equipped with optimized orthogonal projections in `modeling_pacllama_trial.py`, and we conducted experiments by simply patching the base LLaMA implementation using this Python file.

# Usage
## Installation
1. Environment setup:
   ```
   conda create -n MatryoshkaKV python=3.10
   conda activate MatryoshkaKV
   ```
2. Clone this repository and build from source:
   ```
   git clone https://github.com/The-kamisato/MatryoshkaKV-cache.git
   cd MatryoshkaKV-cache
   ```
4. Install dependency:
   ```
   pip install -r requirements.txt
   ```
5. Install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory.git)
   ```
   git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
   cd LLaMA-Factory
   pip install -e ".[torch,metrics]"
   cd ..
   
   ```
6. Install [opencompass](https://github.com/open-compass/opencompass.git)
   ```
   git clone https://github.com/open-compass/opencompass opencompass
   cd opencompass
   pip install -e .
   cd ..
   ```
## Initialization
We first initialize our orthogonal projections by PCA(Principal Component Analysis) running `cal_pcallama_init.py`. 
The dataset used for initialization is downloaded according to [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca.git).
After downloading all of them, organize the data as follows:
```
├──LLaMA-Factory
│   └──data
│       └── alpaca_data_en_52k.json
```

You can calculate the initial parameters using PCA by executing the following command：

```
python cal_pcallama_init.py --model_path [your llama2 checkpoint path] --data_path [your alpaca_en_52k json file path]
```

## Training
During training, our patches are applied to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory.git) at:
- `LLaMA-Factory/src/llamafactory/model/custom_model/modeling_pcallama_trial.py`

Furthermore, due to the use of a distillation objective, we deliver our custom trainer `PcaLlamaDistillationTrainer` and  `PcaLlamaTrainer` at:
- `LLaMA-Factory/src/llamafactory/train/pt/trainer.py`
- `LLaMA-Factory/src/llamafactory/train/sft/trainer.py`

Our training scripts are under `LLaMA-Factory/scripts`. 

And our dataset for continual pre-training is downloaded from [RedPajama-Sample](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample). 

After downloading all of them, organize the data as follows:
```
├──LLaMA-Factory
│   └──data
│       └── alpaca_data_en_52k.json
│       └── RedPajama_Sample.json
```
## Evaluation
For evaluation, our patches are applied to [opencompass](https://github.com/open-compass/opencompass.git) at:
- `opencompass/opencompass/models/custom_model`
  
Additionally, modifications are made for loading Hugging Face models in:
- `opencompass/opencompass/models/huggingface_above_v4_33.py`

First, you can refer to [opencompass](https://github.com/open-compass/opencompass.git) and download the dataset of six benchmarks: HellaSwag, ARC-c, Arc-e, PIQA, WinoGrande, and CommonsenseQA.
Then, organize the data as follows:
```
├──opencompass
│   └──configs
│       └── datasets
│             └── ARC_c
│             └── ARC_e
│             └── commonsenseqa
│             └── hellaswag
│             └── piqa
│             └── winogrande
│       └── ...
│   └──data
│       └── ARC
│             └── ARC_c
│             └── ARC_e
│       └── commonsenseqa
│       └── hellaswag
│       └── piqa
│       └── winogrande
│   └──opencompass
│       └── ...
```

You can evaluate model performance on six benchmarks: HellaSwag, ARC-c, Arc-e, PIQA, WinoGrande, and CommonsenseQA by executing the following command (Referenced to [opencompass](https://github.com/open-compass/opencompass.git))：
```
cd opencompass
python run.py --datasets hellaswag_ppl_a6e128 ARC_c_ppl ARC_e_ppl piqa_ppl_1cf9f0 winogrande_ppl_55a66e commonsenseqa_ppl_e51e32 --hf-type base --hf-path [your llama2 checkpoint path]  
```


# Performance

![result table](https://github.com/The-kamisato/MatryoshkaKV-cache/blob/main/figure/result_table.jpg)

# Visualization

![compression rate](https://github.com/The-kamisato/MatryoshkaKV-cache/blob/main/figure/compression_rate.jpg)

# TODO

In the future, we will release the complete training process.
