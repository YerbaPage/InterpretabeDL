#!/bin/bash
source activate interp

python RobustRepresentationLearning.py \
--databunch_method DataBunch_e_snli_marked \
--dataset e-snli \
--load_few \
--model_name_or_path prajjwal1/bert-tiny \
--train_process train_cause_word \
--saliancy_method compute_saliancy_batch_hess \
--epoch 10 \
--causal_ratio 0 \
--learning_rate 5e-4 \
--batch_size 32 \
--batch_size_test 1 \

# python RobustRepresentationLearning.py \
# --databunch_method DataBunch_e_snli_marked \
# --dataset e-snli \
# --model_name_or_path bert-large-uncased \
# --train_process train_cause_word \
# --epoch 1 \
# --causal_ratio 0.0 \
# --batch_size 32 \
# --learning_rate 2e-5 \
# --load_few 


# --model_name_or_path prajjwal1/bert-tiny \ (32, 3e-4, 0.67)
# --model_name_or_path prajjwal1/bert-mini \
# --model_name_or_path prajjwal1/bert-small \
# --model_name_or_path prajjwal1/bert-medium \
# --model_name_or_path bert-base-uncased \

