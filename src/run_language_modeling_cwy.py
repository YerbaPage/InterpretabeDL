# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)

from torch.utils.data.dataset import Dataset
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import random

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput, TrainOutput
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm, trange
from transformers.training_args import TrainingArguments, is_torch_tpu_available

from module import Lwf_model
import copy
from torch import nn

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


class LineByLineTextDataset_pair(LineByLineTextDataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('\t') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size,padding=block_size,truncation=True)
        self.examples = batch_encoding["input_ids"]
    def get_original_sample(self,index):
        return self.examples[index]

def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size) #cwy_debug
    else:
        return TextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, overwrite_cache=args.overwrite_cache
        )


stop_words = set(['a', 'about', 'after', 'against', 'all', 'along', 'also', 'although', 'am', 'among', 'an', 'and', 'are',
     'as', 'at', 'be', 'became', 'because', 'been', 'being', 'between', 'but', 'by', 'can', 'caution', 'come',
     'could', 'details', 'did', 'do', 'does', 'doing', 'done', 'during', 'each', 'early', 'far', 'finally',
     'for', 'found', 'from', 'had', 'has', 'have', 'he', 'her', 'here', 'his', 'how', 'however', 'I', 'if',
     'in', 'include', 'including', 'information', 'into', 'is', 'it', 'its', 'late', 'later', 'made', 'many',
     'may', 'me', 'med', 'might', 'more', 'most', 'near', 'no', 'non', 'none', 'nor', 'not', 'note', 'of',
     'off', 'on', 'only', 'or', 'other', 'over', 'saw', 'see', 'seen', 'several', 'she', 'since', 'so', 'some',
     'such', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this', 'those',
     'through', 'tip', 'to', 'too', 'under', 'until', 'up', 'us', 'use', 'very', 'warning', 'was', 'we',
     'were', 'when', 'where', 'which', 'while', 'who', 'with', 'you', '.', ',', ':'])

#overlap_count = {}

@dataclass
class DataCollatorForLanguageModeling_cwy(DataCollatorForLanguageModeling):
    def compute_equal_mask(self,inputs):
        #writer = open('temp.txt','w',encoding='utf-8')

        ret=torch.zeros(inputs.size())
        for i, batch in enumerate(inputs):
            sep_index=0
            for j, v in enumerate(batch):
                if v==102:
                    sep_index=j
                    break
            for j1 in range(sep_index):
                for j2 in range(sep_index+1,len(batch)):
                    if batch[j1]==batch[j2]:
                        overlap=self.tokenizer.convert_ids_to_tokens([batch[j1]])[0]
                        if overlap in stop_words:
                            continue
                        if len(overlap)==1:
                            continue
                        #overlap_count[overlap]=overlap_count.get(overlap,0)+1
                        if random.random()>0.5:
                            ret[i][j1]=0.5
                        else:
                            ret[i][j2]=0.5
        '''for overlap in overlap_count:
            writer.write('{}\t{}\n'.format(overlap,overlap_count[overlap]))
        writer.close()'''
        return ret

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        #probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix = self.compute_equal_mask(inputs)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens


        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        '''# 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]'''

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class Trainer_lwf(Trainer):
    def __init__(
            self,
            anchor_model =None,
            **kwargs,
    ):
        super(Trainer_lwf, self).__init__(**kwargs)
        self.anchor_model = anchor_model.cuda()
        for param in self.anchor_model.parameters():
            param.requires_grad = False
        self.loss =nn.KLDivLoss()
        self.logsoftmax=nn.LogSoftmax()
        self.softmax=nn.Softmax()

    def compute_loss(self, model, inputs):
        outputs = model(**inputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        t_outputs = self.anchor_model(**inputs)
        dim=outputs[1].size()[-1]
        d1=self.logsoftmax(outputs[1].reshape(-1,dim))
        d2=self.softmax(t_outputs[1].reshape(-1,dim))
        lwf_loss=self.loss(d1,d2)
        return outputs[0]+lwf_loss

class Trainer_lwf_hidden(Trainer):
    def __init__(
            self,
            anchor_model =None,
            **kwargs,
    ):
        super(Trainer_lwf_hidden, self).__init__(**kwargs)
        self.anchor_model = anchor_model.cuda()
        for param in self.anchor_model.parameters():
            param.requires_grad = False
        self.loss =nn.MSELoss()

    def compute_loss(self, model, inputs):
        outputs = model(**inputs,output_hidden_states=True)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        t_outputs = self.anchor_model(**inputs,output_hidden_states=True)
        dim=outputs[2][-1].size()[-1]
        d1=outputs[2][-1].reshape(-1,dim)
        d2=t_outputs[2][-1].reshape(-1,dim)
        lwf_loss=self.loss(d1,d2)
        return outputs[0]+lwf_loss

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args,remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    '''if '--lwf' in remaining_args:
        global anchor_model
        anchor_model = copy.deepcopy(model)
        model = Lwf_model(model,anchor_model)'''

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets

    train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    data_collator = DataCollatorForLanguageModeling(  #debug cwy
        tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
    )

    if '--lwf' in remaining_args:
        trainer = Trainer_lwf(  #debug cwy
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            prediction_loss_only=True,
            anchor_model=copy.deepcopy(model)
        )
    elif '--lwf_hidden' in remaining_args:
        trainer = Trainer_lwf_hidden(  # debug cwy
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            prediction_loss_only=True,
            anchor_model=copy.deepcopy(model)
        )
    else:
        trainer = Trainer(  #debug cwy
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            prediction_loss_only=True,
        )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
