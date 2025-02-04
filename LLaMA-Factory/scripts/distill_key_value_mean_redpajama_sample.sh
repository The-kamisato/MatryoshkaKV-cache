###  device, port, batch_size, dataset, output_dir, model.py(idx, load)
export CUDA_VISIBLE_DEVICES=0,1,2,3


accelerate launch \
    --config_file LLaMA-Factory/examples/accelerate/single_config.yaml \
    --main_process_port 29506 \
    LLaMA-Factory/src/train.py \
    --stage pt \
    --do_train True \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --low_cpu_mem_usage False \
    --preprocessing_num_workers 16 \
    --ddp_find_unused_parameters False \
    --finetuning_type freeze \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset redpajama_sample \
    --key_unitary_transform_lr 1e-03 \
    --value_unitary_transform_lr 1e-03 \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 1 \
    --max_samples 100000000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing True \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 300 \
    --val_size 10 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --warmup_steps 100 \
    --optim adamw_torch \
    --packing False \
    --output_dir saves/LLaMA-7B/distillation/redpajama/ditillation_ckpt \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --additional_target key_unitary_transform,value_unitary_transform,mean_key,mean_value \
