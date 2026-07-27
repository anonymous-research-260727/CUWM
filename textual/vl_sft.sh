# 22GB
WANDB_PROJECT='cuwm-vl-sft'
nproc_per_node=4
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model "${MODEL_BASE:-models/Qwen}/Qwen2.5-VL-7B-Instruct" \
    --train_type lora \
    --dataset "${DATA_ROOT:-./data}/train_data_1231_2867.json" \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --save_strategy epoch \
    --dataset_shuffle true \
    --train_dataloader_shuffle true \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --max_length 8192 \
    --output_dir output/prompt_1228 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
    --report_to wandb \
