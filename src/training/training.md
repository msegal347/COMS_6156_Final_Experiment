%%bash
export SQUAD_DIR='COMS_6156_Final_Experiment/data/raw/SQuAD_1.1_dev.json'  
torchrun --nproc_per_node=8 src/training/run_squad.py \
    --model_type bert \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir models/wwm_uncased_finetuned_squad/ \
    --per_gpu_eval_batch_size=3 \
    --per_gpu_train_batch_size=3