#!/bin/bash
source activate interp

# python RobustRepresentationLearning.py \
# --databunch_method DataBunch_e_snli_marked \
# --dataset e-snli \
# --load_few \
# --model_name_or_path prajjwal1/bert-medium \
# --train_process train_cause_word \
# --epoch 2 \
# --causal_ratio 0 \
# --learning_rate 3e-5 \
# --batch_size 32

python RobustRepresentationLearning.py \
--databunch_method DataBunch_e_snli_marked \
--dataset e-snli \
--load_few \
--model_name_or_path bert-base-uncased \
--train_process train_cause_word \
--epoch 10 \
--causal_ratio 0 \
--learning_rate 5e-5 \
--batch_size 32

python RobustRepresentationLearning.py \
--databunch_method DataBunch_e_snli_marked \
--dataset e-snli \
--model_name_or_path bert-base-uncased \
--train_process train_cause_word \
--epoch 15 \
--causal_ratio 0.0 \
--batch_size 32 \
--learning_rate 5e-5 \
--load_few \
--grad_loss_func argmax_one_loss

python RobustRepresentationLearning.py \
--databunch_method DataBunch_e_snli_marked \
--dataset e-snli \
--model_name_or_path prajjwal1/bert-medium \
--train_process train_cause_word \
--epoch 15 \
--causal_ratio 0.0 \
--batch_size 32 \
--learning_rate 5e-5 \
--load_few \
--grad_loss_func argmax_one_loss
# --model_name_or_path prajjwal1/bert-tiny \
# --model_name_or_path prajjwal1/bert-mini \
# --model_name_or_path prajjwal1/bert-small \
# --model_name_or_path prajjwal1/bert-medium \
# --model_name_or_path bert-base-uncased \

