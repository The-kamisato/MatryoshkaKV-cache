###  device, port, batch_size, dataset, output_dir, model.py(idx, load)
export CUDA_VISIBLE_DEVICES=2
export WANDB_DISABLED=true

accelerate launch \
    --config_file /liymai24/sjtu/bokai/LLaMA-Factory/examples/accelerate/single_config.yaml \
    --main_process_port 29502 \
    /liymai24/sjtu/bokai/LLaMA-Factory/src/train.py \
    --stage sft \
    --do_train True \
    --model_name_or_path /liymai24/sjtu/bokai/PCA_kvcache/checkpoint/Llama-2-7b-hf \
    --low_cpu_mem_usage False \
    --preprocessing_num_workers 16 \
    --ddp_find_unused_parameters False \
    --finetuning_type lora \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset alpaca_en \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 2 \
    --max_samples 1000000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 500 \
    --val_size 10 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --warmup_steps 10 \
    --optim adamw_torch \
    --packing False \
    --output_dir saves/LLaMA-7B/lora/alpaca_baseline/train_only_lora_5e-05 \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0 \
    --lora_target up_proj,o_proj,k_proj,gate_proj,q_proj,down_proj,v_proj \
    # --resume_from_checkpoint /liymai24/sjtu/bokai/LLaMA-Factory/saves/LLaMA-7B/lora/alpaca_baseline/train_lora_key_value_mean_128_128_5e-05 \
    # ,mean_key,mean_value