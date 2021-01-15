#!/bin/bash
source activate interp

# python RobustRepresentationLearning.py \
# --databunch_method DataBunch_e_snli_marked \
# --dataset e-snli \
# --load_few \
# --model_name_or_path prajjwal1/bert-medium \
# --train_process train_cause_word \
# --epoch 3 \
# --causal_ratio 0 \
# --learning_rate 2e-5 \
# --batch_size 32

python RobustRepresentationLearning.py \
--databunch_method DataBunch_e_snli_marked \
--dataset e-snli \
--load_few \
--model_name_or_path bert-base-uncased \
--train_process train_cause_word \
--epoch 2 \
--causal_ratio 0 \
--learning_rate 2e-5 \
--batch_size 16

# --model_name_or_path prajjwal1/bert-tiny \
# --model_name_or_path prajjwal1/bert-mini \
# --model_name_or_path prajjwal1/bert-small \
# --model_name_or_path prajjwal1/bert-medium \
# --model_name_or_path bert-base-uncased \

