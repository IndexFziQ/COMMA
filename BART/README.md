
## Run script

CUDA_VISIBLE_DEVICES=7 nohup python finetune_trainer.py --model_name_or_path facebook/bart-large --data_dir ./test_data/diag_intention/ --output_dir ./output_my_intention --learning_rate=3e-5 --do_train --do_eval --do_predict --evaluation_strategy steps --predict_with_generate --n_val 1000 --overwrite_output_dir --per_device_train_batch_size 8 --gradient_accumulation_steps 4 > my_bart_intention.out 2>&1 &
