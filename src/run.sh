#!/bin/bash
source activate interp

python RobustRepresentationLearning.py \
--databunch_method DataBunch_e_snli_marked \
--dataset e-snli \
--load_few \
--model_name_or_path bert-base-uncased \
--train_process train_cause_word \
--epoch 2 \
--causal_ratio 0.1 \
--batch_size 32

# --model_name_or_path prajjwal1/bert-mini \
