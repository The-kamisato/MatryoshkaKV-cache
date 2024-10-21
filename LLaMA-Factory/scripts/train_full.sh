VERSION="lora_key_value"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline
export NCCL_DEBUG=INFO
# export FSDP_TRANSFORMER_CLS_TO_WRAP='PcaLlamaDecoderLayer'


accelerate launch \
    --config_file /liymai24/sjtu/bokai/LLaMA-Factory/examples/accelerate/fsdp_config.yaml \
    --main_process_port 29506 \
    /liymai24/sjtu/bokai/LLaMA-Factory/src/train.py \
    --stage sft \
    --do_train True \
    --resume_from_checkpoint False \
    --model_name_or_path /liymai24/sjtu/bokai/PCA_kvcache/checkpoint/Llama-2-7b-hf \
    --low_cpu_mem_usage False \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset obqa_train,obqa_train2,piqa_train,siqa_train,hellaswag_train,gsm8k_train \
    --cutoff_len 1024 \
    --learning_rate 1e-05 \
    --num_train_epochs 10.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --optim adamw_torch \
    --packing False \
    --output_dir saves/LLaMA-7B/lora/train_full \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
