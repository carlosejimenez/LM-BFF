# If we want to run K-shot, we need to use the right data_dir (with k-shot data generated from tools/generate_k_shot_data.py) and num_k, num_sample is only used for prompt-demo
K=16
SEED=42
LR=1e-5

python run.py \
    --task_name CSTS \
    --data_dir data/k-shot/CSTS/$K-$SEED \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluate_during_training \
    --model_name_or_path roberta-large \
    --few_shot_type prompt-demo \
    --num_k $K \
    --max_steps 1000 \
    --eval_steps 100 \
    --per_device_train_batch_size 2 \
    --learning_rate $LR \
    --num_train_epochs 0 \
    --output_dir result/tmp \
    --seed $SEED \
    --template "*cls*_Sentences_*sent_0*._similarity_with_respect_to*+sent_1*_is_*mask**sep+*" \
    --mapping "{'0':'low','1':'high'}" \
    --num_sample 16 
