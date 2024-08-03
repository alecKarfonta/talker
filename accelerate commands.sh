# Fine-tune sequence-to-sequence models
accelerate launch run_seq2seq_no_trainer.py     --dataset_name "smangrul/MuDoConv"     --max_source_length 128     --source_prefix "chatbot: "     --max_target_length 64     --val_max_target_length 64     --val_min_target_length 20     --n_val_batch_generations 5     --n_train 10000     --n_val 1000     --pad_to_max_length     --num_beams 10     --model_name_or_path "PygmalionAI/pygmalion-6b"     --per_device_train_batch_size 200     --per_device_eval_batch_size 100     --learning_rate 1e-6     --weight_decay 0.0     --num_train_epochs 1     --gradient_accumulation_steps 1     --num_warmup_steps 100     --output_dir "/tmp/deepspeed_zero_stage2_accelerate_test"     --seed 25     --logging_steps 100     --with_tracking     --report_to "wandb"     --report_name "pygamalion_finetuning"


# Fine-tune casual language models 
accelerate launch run_clm_no_trainer.py     --dataset_name "smangrul/MuDoConv" --model_name_or_path "PygmalionAI/pygmalion-350m" --num_warmup_steps 100 --gradient_accumulation_steps 5 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --seed=25  --output_dir "/py/Source/deepspeed_zero_stage2_accelerate_test"


# Deepspeed - for distributed model training
# https://huggingface.co/docs/transformers/main_classes/deepspeed

# Configure deepspeed
deepspeed Configure
# Config file path
/py/Source/accelerate-deepspeed-test/zero3_offload_config_accelerate.json


# GPT-Q for model quatization
# https://github.com/qwopqwop200/GPTQ-for-LLaMa
python3 gptj.py models/pygmalion-6b_dev c4 --wbits 4 --groupsize 128 --save_safetensors models/pygmalion-6b_dev-4bit-128g.safetensors
