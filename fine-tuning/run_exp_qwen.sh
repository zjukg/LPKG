MASTER_PORT=${MASTER_PORT:-12346}

NUM_PROCESSES=`echo ${CUDA_VISIBLE_DEVICES} | awk -F ',' '{print NF}'`
if [ ${NUM_PROCESSES} -eq 0 ]; then
NUM_PROCESSES=`echo ${NVIDIA_VISIBLE_DEVICES} | awk -F ',' '{print NF}'`
fi




BASE_MODEL_DIR=
# DATASET_DIR=
DATASET_NAME=kg_supervise

OUTPUT_DIR=


CUTOFF_LEN=8192

pip install -r requirements.txt

# adhoc change
pip uninstall flash_attn -y
pip install accelerate==0.27.2

deepspeed_cmd="deepspeed --num_gpus ${NUM_PROCESSES} --master_port=${MASTER_PORT}"

args=(
    src/train_bash.py
    --deepspeed ds_config.json
    --stage sft
    --model_name_or_path ${BASE_MODEL_DIR}
    --do_train
    --dataset ${DATASET_NAME}
    # --dataset_dir ${DATASET_DIR}
    --template qwen 
    --cutoff_len ${CUTOFF_LEN}
    --max_new_tokens 8192 #8192
    # --max_samples 50
    --finetuning_type lora #full,lora
    --lora_target q_proj,v_proj #q_proj,v_proj c_attn
    --lora_rank 8 
    --output_dir ${OUTPUT_DIR}
    --overwrite_cache
    --per_device_train_batch_size 2 #2
    --gradient_accumulation_steps 1
    --lr_scheduler_type cosine
    --logging_steps 10
    --save_strategy "epoch"
    --learning_rate 5e-5
    --num_train_epochs 2 #5
    --plot_loss
    --report_to tensorboard
    --bf16
    --overwrite_output_dir
    --save_only_model
)

$deepspeed_cmd "${args[@]}"

