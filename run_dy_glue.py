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
""" Modify file from transformers/examples/text-classification/run_glue.py.
    Finetuning the library models for sequence classification on GLUE.
"""


import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List

import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

from models import modeling_ebert, modeling_eroberta
from custom_utils import cutils, custom_trainer

MODELS = {
    'bert': modeling_ebert.BertForSequenceClassification,
    'roberta':modeling_eroberta.RobertaForSequenceClassification
}

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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
class CustomArguments:
    """
    Arguments to 
    """
    fill_mode: Optional[str] = field(
        default="zero", metadata={"help": "Choose fill mode when head is pruned: zero, residual, mul"}
    )
    head_mask_mode: Optional[str] = field(
        default="gumbel", metadata={"help": "Head mask generate mode: topk, gumbel"}
    )
    ffn_mask_mode: Optional[str] = field(
        default="gumbel", metadata={"help": "FFN mask generate mode: topk, gumbel"}
    )
    loss_lambda: Optional[float] = field(
        default=0.1, metadata={"help": "coefficient to control loss of sparsity"}
    )
    target_flops_ratio: Optional[float] = field(
        default=1.0, metadata={"help": "expected flops"}
    )
    predictor_lr: Optional[float] = field(
        default=0.1, metadata={"help": "learning rate for predictor branch"}
    )
    freeze_predictor: Optional[bool] = field(
        default=False, metadata={"help": "freeze the weights of predictors"}
    )
    
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, custom_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    training_args.predictor_lr = custom_args.predictor_lr
    training_args.mask_mode = custom_args.head_mask_mode
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

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if training_args.do_train:
        config.fill_mode = custom_args.fill_mode
        config.head_mask_mode = custom_args.head_mask_mode
        config.ffn_mask_mode = custom_args.ffn_mask_mode
        config.loss_lambda = custom_args.loss_lambda
        config.sparsity_target = 1 - custom_args.target_flops_ratio

    model = MODELS[config.model_type].from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if custom_args.freeze_predictor:
        for name, para in model.named_parameters():
            if 'predictor' in name:
                para.requires_grad = False
        logger.info("Freeze the weights of predictors")
    
    freeze_backone = False
    if freeze_backone:
        for name, para in model.named_parameters():
            if not 'predictor' in name:
                para.requires_grad = False

    # Get datasets
    train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    test_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
        if training_args.do_predict
        else None
    )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            if output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            else:  # regression
                preds = np.squeeze(preds)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    # Initialize our Trainer
    trainer = custom_trainer.Custom_Trainer(
        max_seq_length=data_args.max_seq_length,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )

    # Train
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
            )

        for eval_dataset in eval_datasets:
            # set flag to clear sparsity buffer and re-compute sparsity
            for name, module in model.named_modules():
                if hasattr(module, 'clear_sparsity'):
                    setattr(module, 'clear_sparsity', True)
                if hasattr(module, 'get_sparsity'):
                    setattr(module, 'get_sparsity', True)

            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            for name, module in model.named_modules():
                if hasattr(module, 'get_sparsity'):
                    setattr(module, 'get_sparsity', False)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                    cutils.compute_model_sparsity(data_args.max_seq_length, len(eval_dataset), 
                                                            model, logger, writer, trainer.log)

            eval_results.update(eval_result)

    # Test
    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
            )

        for test_dataset in test_datasets:
            for name, module in model.named_modules():
                if hasattr(module, 'clear_sparsity'):
                    setattr(module, 'clear_sparsity', True)
                if hasattr(module, 'get_sparsity'):
                    setattr(module, 'get_sparsity', True)

            predictions = trainer.predict(test_dataset=test_dataset).predictions
            for name, module in model.named_modules():
                if hasattr(module, 'get_sparsity'):
                    setattr(module, 'get_sparsity', False)

            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    cutils.compute_model_sparsity(data_args.max_seq_length, len(test_dataset), 
                                        model, logger, None, None)
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))
    return eval_results

if __name__ == "__main__":
    main()
