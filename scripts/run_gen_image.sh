export CUDA_VISIBLE_DEVICES=0 && conda activate base && python -m eval.image_gen --model_key base --data_folder new_prompt/test/
export CUDA_VISIBLE_DEVICES=1 && conda activate diffsynth && python -m eval.image_gen --model_key epoch-24 --data_folder new_prompt/test/word
export CUDA_VISIBLE_DEVICES=2 && conda activate diffsynth && python -m eval.image_gen --model_key epoch-24 --data_folder new_prompt/test/ppt
export CUDA_VISIBLE_DEVICES=3 && conda activate diffsynth && python -m eval.image_gen --model_key epoch-24 --data_folder new_prompt/test/excel