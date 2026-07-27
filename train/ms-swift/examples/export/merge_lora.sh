# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
swift export \
    --adapters /path/to/cuwm/office-wm/output/prompt_1228/v3-20251231-143109/checkpoint-450 \
    --merge_lora true
