
pip install -r requirements.txt

pip uninstall flash_attn -y 
#pip uninstall flash_attn
pip install accelerate==0.27.2
# pip install "unsloth[cu118_ampere] @ git+https://github.com/unslothai/unsloth.git" 

BASE_MODEL_DIR=
# DATASET_DIR=
CKPT_PATH=

DATASET_NAME=hotpotqa_test_500_planning 
PRED_PATH=

mkdir -p ${PRED_PATH}


args=(
    src/train_bash.py
    --stage sft
    --model_name_or_path ${BASE_MODEL_DIR}
    --adapter_name_or_path  ${CKPT_PATH} 
    --do_predict
    --do_sample False
    --num_beams 3
    --dataset ${DATASET_NAME}
    # --dataset_dir  ${DATASET_DIR}
    --template qwen 
    --cutoff_len 8192 
    --finetuning_type lora
    --lora_target q_proj,v_proj 
    --lora_rank 8
    --output_dir ${PRED_PATH}
    --per_device_eval_batch_size 1 #2
    --predict_with_generate
)

accelerate launch "${args[@]}" 

