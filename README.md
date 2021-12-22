# EBERT
This repository serves as the official code release of the paper [EBERT: Efficient BERT Inference with Dynamic Structured Pruning](https://aclanthology.org/2021.findings-acl.425/) (pubilished at Findings of ACL 2021).

<div align=center>
<img src=EBERT.png>
</div>

EBERT is a dynamic structured pruning algorithm for efficient BERT inference. Unlike previous methods that randomly prune the model weights for static inference, EBERT dynamically determines and prunes the unimportant heads in multi-head self-attention layers and the unimportant structured computations in feed-forward network for each input sample at run-time.

## Prerequisites

The code has the following dependencies:
* python >= 3.8.5
* pytorch >= 1.4.0
* transformers = 3.3.1
As transformers v3.3.1 has a bug when the evaluation strategy is `epoch`, you need to make the following changes in the transformers library:
```
--- a/src/transformers/training_args.py
+++ b/src/transformers/training_args.py
@@ -323,7 +323,7 @@ class TrainingArguments:
     def __post_init__(self):
         if self.disable_tqdm is None:
             self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN
-        if self.evaluate_during_training is not None:
+        if self.evaluate_during_training:
             self.evaluation_strategy = (
                 EvaluationStrategy.STEPS if self.evaluate_during_training else EvaluationStrategy.NO
             )
```

## Usages
We provide script files for training and validation in the `scripts` folder, and users can run these script from the repo root, e.g. `bash scripts/eval_glue.sh`.
In each scripts, there are several arguments to modify before running:
* `--data_dir`: path to dataset：[GLUE](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e), [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/). 
* `MODEL_PATH` or `--model_name_or_path`: path to trained model folder
* `TASK_NAME`: task name in GLUE (SST-2, MNLI, ...)
* `RUN_NAME`: name of the current experiment, which influence the save path and log name for wandb.
* other hyper-parameters, e.g., `head_mask_mode`

You can download the original pretrained model of [BERT](https://huggingface.co/bert-base-uncased) and [RoBERTa](https://huggingface.co/roberta-base) from HuggingFace. 

## Citation
If you found the library useful for your work, please kindly cite our work:
```
@inproceedings{liu-etal-2021-ebert,
    title = "{EBERT}: Efficient {BERT} Inference with Dynamic Structured Pruning",
    author = "Liu, Zejian  and
              Li, Fanrong  and
              Li, Gang  and
              Cheng, Jian",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.425",
    doi = "10.18653/v1/2021.findings-acl.425",
    pages = "4814--4823",
}
```
