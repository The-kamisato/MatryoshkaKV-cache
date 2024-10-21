VERSION="non_lora_key_value"
export CUDA_VISIBLE_DEVICES=2,3
export WANDB_MODE=offline

accelerate launch \
    --config_file /liymai24/sjtu/bokai/LLaMA-Factory/examples/accelerate/single_config.yaml \
    --main_process_port 29501 \
    /liymai24/sjtu/bokai/LLaMA-Factory/src/train.py \
    --stage sft \
    --do_train True \
    --resume_from_checkpoint False \
    --model_name_or_path /liymai24/sjtu/bokai/PCA_kvcache/checkpoint/Llama-2-7b-hf \
    --low_cpu_mem_usage False \
    --preprocessing_num_workers 16 \
    --ddp_find_unused_parameters False \
    --finetuning_type freeze \
    --freeze_trainable_layers 0 \
    --template llama2 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset obqa_train,obqa_train2,piqa_train,siqa_train,hellaswag_train \
    --cutoff_len 256 \
    --learning_rate 1e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --optim adamw_torch \
    --packing False \
    --report_to wandb \
    --output_dir saves/LLaMA-7B/non_lora/train_non_lora_key_value_freeze_mean_64 \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --additional_target key_unitary_transform,value_unitary_transform