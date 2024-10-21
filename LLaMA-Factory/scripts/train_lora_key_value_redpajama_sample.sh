###  device, port, batch_size, dataset, output_dir, model.py(idx, load)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_DISABLED=true

accelerate launch \
    --config_file /liymai24/sjtu/bokai/LLaMA-Factory/examples/accelerate/single_config.yaml \
    --main_process_port 29501 \
    /liymai24/sjtu/bokai/LLaMA-Factory/src/train.py \
    --stage pt \
    --do_train True \
    --model_name_or_path /liymai24/sjtu/bokai/PCA_kvcache/checkpoint/Llama-2-7b-hf \
    --low_cpu_mem_usage False \
    --preprocessing_num_workers 16 \
    --ddp_find_unused_parameters False \
    --finetuning_type lora \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset redpajama_sample \
    --key_unitary_transform_lr 1e-03 \
    --value_unitary_transform_lr 1e-03 \
    --cutoff_len 1800 \
    --learning_rate 5e-05 \
    --num_train_epochs 1 \
    --max_samples 100000000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 500 \
    --val_size 10 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --warmup_steps 100 \
    --optim adamw_torch \
    --packing False \
    --output_dir saves/LLaMA-7B/lora/redpajama/train_all_heads_16_64_64_eval_5e-05 \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0 \
    --lora_target up_proj,o_proj,k_proj,gate_proj,q_proj,down_proj,v_proj \
    --additional_target key_unitary_transform,value_unitary_transform,mean_key,mean_value \
