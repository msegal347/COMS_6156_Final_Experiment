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

    %run /content/COMS_6156_Final_Experiment/src/training/run_squad.py \
  --train_file /content/COMS_6156_Final_Experiment/data/raw/SQuAD_1.1_train.json \
  --predict_file /content/COMS_6156_Final_Experiment/data/raw/SQuAD_1.1_dev.json \
  --model_type bert \  # Assuming you are using BERT, replace with 'xlnet' or 'xlm' if using others
  --model_name_or_path bert-base-uncased \  # This should be the model identifier or the path to a pre-trained model
  --output_dir /content/COMS_6156_Final_Experiment/data/output \  # Specify where you want to save outputs
  --max_seq_length 384 \
  --doc_stride 128 \
  --max_query_length 64 \
  --do_train \
  --do_eval \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --max_steps -1 \
  --warmup_steps 0 \
  --n_best_size 20 \
  --max_answer_length 30 \
  --logging_steps 50 \
  --save_steps 50 \
  --seed 42 \
  --overwrite_output_dir