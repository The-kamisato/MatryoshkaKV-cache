###  device, port, batch_size, dataset, output_dir, train_mean, model.py(idx, load)
VERSION="lora_key_value"
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=offline

accelerate launch \
    --config_file /liymai24/sjtu/bokai/LLaMA-Factory/examples/accelerate/single_config.yaml \
    --main_process_port 29502 \
    /liymai24/sjtu/bokai/LLaMA-Factory/src/train.py \
    --stage sft \
    --do_train True \
    --resume_from_checkpoint False \
    --model_name_or_path /liymai24/sjtu/bokai/PCA_kvcache/checkpoint/Llama-2-7b-chat-hf \
    --low_cpu_mem_usage False \
    --preprocessing_num_workers 16 \
    --ddp_find_unused_parameters False \
    --finetuning_type lora \
    --template llama2 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset siqa_train_detail \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 1 \
    --max_samples 1000000 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --val_size 10 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy steps \
    --eval_steps 20 \
    --warmup_steps 0 \
    --optim adamw_torch \
    --packing False \
    --output_dir saves/LLaMA-7B/lora/siqa_baseline/train_original_llama2-7b-chat_4 \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0 \
    --lora_target up_proj,o_proj,k_proj,gate_proj,q_proj,down_proj,v_proj \