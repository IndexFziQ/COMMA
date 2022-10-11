# Code for COMMA (COLING 2022)

The Code of COMMA (COgnitive fraMework of huMan Activity).

## Setup

Download the pre-trained language models from https://huggingface.co/datasets.
- BERT-LARGE
- ROBERTA-LARGE
- GPT-2-LARGE
- BART-LARGE

## Running the COMMA experiment

Please go into src file.
### Emotion Understanding
```bash
# BERT
bash ./scripts/comma_m2e/run_comma_m2e_bert.sh
# RoBERTa
bash ./scripts/comma_m2e/run_comma_m2e_roberta.sh
```

### Motivation Understanding
```bash
# BERT
bash ./scripts/comma_e2m/run_comma_e2m_bert.sh
# RoBERTa
bash ./scripts/comma_e2m/run_comma_e2m_roberta.sh
```

### Conditioned Action Generation

#### GPT-2
```bash
# train w/ individual and needs
bash ./scripts/comma_x2b/run_gpt2_lm_m2b.single.sh
# train w/ individual and emotions
bash ./scripts/comma_x2b/run_gpt2_lm_e2b.single.sh
# train w/ all
bash ./scripts/comma_x2b/run_gpt2_lm_me2b.single.sh
```
#### BART-Large

Please return into BART file.
```bash
CUDA_VISIBLE_DEVICES=0 python finetune_trainer.py --model_name_or_path ./model/bart-large --data_dir ./bart_data/mber/ --output_dir ./output --learning_rate=3e-5 --do_train --do_eval --do_predict --evaluation_strategy steps --predict_with_generate --n_val 1000 --overwrite_output_dir --per_device_train_batch_size 8 --gradient_accumulation_steps 4
```

### Acknowledgement
Thanks for the following github projects:
- https://github.com/huggingface/transformers
- https://github.com/allenai/allennlp
- https://github.com/debjitpaul/Multi-Hop-Knowledge-Paths-Human-Needs

