# MatryoshkaKV cache: Adaptive KV compression via Trainable orthogonal projection

![architecture](https://github.com/The-kamisato/MatryoshkaKV-cache/blob/main/figure/architecture.jpg)

Code for [MAtryoshkaKV-cache](https://arxiv.org/abs/2410.14731).

This project delivered LLaMA equipped with optimized orthogonal projections in `modeling_pacllama_trial.py`, and we conducted experiments by simply patching the base LLaMA implementation using this Python file.

During training, our patches are applied to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory.git) at:
- `LLaMA-Factory/src/llamafactory/model/custom_model/modeling_pcallama_trial.py`

Furthermore, due to the use of a distillation objective, we deliver our custom trainer `PcaLlamaDistillationTrainer` and  `PcaLlamaTrainer` at:
- `LLaMA-Factory/src/llamafactory/train/pt/trainer.py`
- `LLaMA-Factory/src/llamafactory/train/sft/trainer.py`

Our training scripts are under `LLaMA-Factory/scripts`. 

And our dataset for continual pre-training is downloaded from [RedPajama-Sample](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample). 

For evaluation, our patches are applied to [opencompass](https://github.com/open-compass/opencompass.git) at:
- `opencompass/opencompass/models/custom_model`
  
Additionally, modifications are made for loading Hugging Face models in:
- `opencompass/opencompass/models/huggingface_above_v4_33.py`

# Performance

![result table](https://github.com/The-kamisato/MatryoshkaKV-cache/blob/main/figure/result_table.jpg)

# Visualization

![compression rate](https://github.com/The-kamisato/MatryoshkaKV-cache/blob/main/figure/compression_rate.jpg)

