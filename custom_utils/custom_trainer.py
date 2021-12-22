import inspect
import json
import math
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange

from transformers.file_utils import WEIGHTS_NAME

from transformers import modeling_bert
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvaluationStrategy,
    HPSearchBackend,
    TrainerState,
    TrainOutput,
    set_seed,
)

from transformers.utils import logging

from transformers import Trainer

from custom_utils import cutils

PT_LR_SCHEDULER_WARNING = "Please also save or load the state of the optimzer when saving or loading the scheduler."


logger = logging.get_logger(__name__)


def reissue_pt_warnings(caught_warnings):
    # Reissue warnings that are not the PT_LR_SCHEDULER_WARNING
    if len(caught_warnings) > 1:
        for w in caught_warnings:
            if w.category != UserWarning or w.message != PT_LR_SCHEDULER_WARNING:
                warnings.warn(w.message, w.category)

def get_hook_fn_0(out, idx, name):
    def hook_forward_fn(module, input, output):
        out[idx] = output[0]
        # print(name, type(out[idx]))
    return hook_forward_fn

def get_hook_fn_1(out, idx, name):
    def hook_forward_fn(module, input, output):
        out[idx] = output[0][0]
        # print(name, type(out[idx]))
    return hook_forward_fn

def get_hook_fn_2(idx, name):
    def hook_forward_fn(module, input, output):
        print(idx, 'forward', input[0].mean(), output[0].mean())
    return hook_forward_fn

def get_hook_fn_3(idx, name):
    def hook_backward_fn(module, input, output):
        print(idx, 'backward', output[0].mean())
    return hook_backward_fn 

class Custom_Trainer(Trainer):

    def __init__(self, *args, max_seq_length=None, **kwargs):
        super(Custom_Trainer, self).__init__(*args, **kwargs)
        self.max_seq_length = max_seq_length

    def create_double_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            backbone_optimizer_grouped_parameters = [
                {"params":[], "weight_decay": self.args.weight_decay},
                {"params":[], "weight_decay": 0.0}
            ]
            predictor_optimizer_grouped_parameters = [{"params":[]}]
            for n, p in self.model.named_parameters():
                if p.requires_grad == False:
                    continue
                if "predictor" in n:
                    predictor_optimizer_grouped_parameters[0]["params"].append(p)
                elif not any(nd in n for nd in no_decay):
                    backbone_optimizer_grouped_parameters[0]["params"].append(p)
                else:
                    backbone_optimizer_grouped_parameters[1]["params"].append(p)
            self.optimizer = AdamW(
                backbone_optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,                
            )
            self.predictor_optimizer = AdamW(
                predictor_optimizer_grouped_parameters,
                lr=self.args.predictor_lr,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,                
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )
    
    def create_single_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
                                and p.requires_grad == True],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                                and p.requires_grad == True],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def soft_cross_entropy(self, predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).mean()

    def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(num_update_steps_per_epoch * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = t_total

        if self.args.mask_mode == 'gumbel':
            self.create_double_optimizer_and_scheduler(num_training_steps=t_total)
        else:
            self.create_single_optimizer_and_scheduler(num_training_steps=t_total)

        self.state = TrainerState()

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            with warnings.catch_warnings(record=True) as caught_warnings:
                self.lr_scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))
            reissue_pt_warnings(caught_warnings)

        # Check if a saved Trainer state exist
        if model_path is not None and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
            self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))

        model = self.model

        # multi-gpu training 
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training 
        if self.args.local_rank != -1:
            logging.ERROR("Don't support DistributedTrain!")
        # find_unused_parameters breaks checkpointing as per
        # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        total_train_batch_size = (
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        self.total_flos = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split(os.path.sep)[0])
                self.total_flos = getattr(self._actual_model(model).config, "total_flos", 0)

                epochs_trained = self.global_step // num_update_steps_per_epoch
                steps_trained_in_current_epoch = self.global_step % (num_update_steps_per_epoch)

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Continuing training from %d non-embedding floating-point operations", self.total_flos)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                self.total_flos = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        loss_mse = torch.nn.MSELoss()
        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        train_pbar = trange(epochs_trained, int(np.ceil(num_train_epochs)), desc="Epoch", disable=disable_tqdm)
        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)


            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            epoch_pbar = tqdm(epoch_iterator, desc="Iteration", disable=disable_tqdm)
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    epoch_pbar.update(1)
                    continue

                self.total_flos += self.floating_point_ops(inputs)
                inputs = self._prepare_inputs(inputs)
                model.train()
                cls_loss, student_logits  = model(**inputs)[0:2]

                
                loss = cls_loss

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                
                loss.backward()

                tr_loss += loss.detach()

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):

                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)


                    self.optimizer.step()
                    if self.args.mask_mode == 'gumbel':
                        self.predictor_optimizer.step()

                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            self.lr_scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else self.lr_scheduler.get_lr()[0]
                        )
                        logging_loss_scalar = tr_loss_scalar

                        self.log(logs)

                    if (
                        self.args.evaluation_strategy == EvaluationStrategy.STEPS
                        and self.global_step % self.args.eval_steps == 0
                    ):
                        metrics = self.evaluate()
                        self._report_to_hp_search(trial, epoch, metrics)
                        if self.args.load_best_model_at_end:
                            self._save_training(model, trial, metrics=metrics)

                    if (
                        not self.args.load_best_model_at_end
                        and self.args.save_steps > 0
                        and self.global_step % self.args.save_steps == 0
                    ):
                        self._save_training(model, trial)

                epoch_pbar.update(1)
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break
            epoch_pbar.close()
            train_pbar.update(1)
            # self.predictor_lr_scheduler.step()

            if self.args.evaluation_strategy == EvaluationStrategy.EPOCH:
                for name, module in model.named_modules():
                    if hasattr(module, 'clear_sparsity'):
                        setattr(module, 'clear_sparsity', True)
                    if hasattr(module, 'get_sparsity'):
                        setattr(module, 'get_sparsity', True)

                metrics = self.evaluate()

                for name, module in model.named_modules():
                    if hasattr(module, 'get_sparsity'):
                        setattr(module, 'get_sparsity', False)
                cutils.compute_model_sparsity(self.max_seq_length, len(self.eval_dataset), 
                                                            model, logger, None, self.log)        
                self._report_to_hp_search(trial, epoch, metrics)
                if self.args.load_best_model_at_end:
                    self._save_training(model, trial, metrics=metrics)

            if self.args.tpu_metrics_debug or self.args.debug:
                logger.warning(
                    "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                    "configured. Check your training configuration if this is unexpected."
                )
            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break

        train_pbar.close()
        if self.tb_writer:
            self.tb_writer.close()
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")

        return TrainOutput(self.global_step, tr_loss.item() / self.global_step)

    def _hp_search_setup(self, trial: Union["optuna.Trial", Dict[str, Any]]):
        """ HP search setup code """
        if self.hp_search_backend is None or trial is None:
            return
        params = self.hp_space(trial) if self.hp_search_backend == HPSearchBackend.OPTUNA else trial
        for key, value in params.items():
            if not hasattr(self.args, key):
                raise AttributeError(
                    f"Trying to set {key} in the hyperparameter search but there is no corresponding field in `TrainingArguments`."
                )
            old_attr = getattr(self.args, key, None)
            # Casting value to the proper type
            if old_attr is not None:
                value = type(old_attr)(value)
            setattr(self.args, key, value)
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            logger.info("Trial:", trial.params)

    def _report_to_hp_search(
        self, trial: Union["optuna.Trial", Dict[str, Any]], epoch: int, metrics: Dict[str, float]
    ):
        if self.hp_search_backend is None or trial is None:
            return
        self.objective = self.compute_objective(metrics)
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            trial.report(self.objective, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        elif self.hp_search_backend == HPSearchBackend.RAY:
            if self.global_step % self.args.save_steps == 0:
                self._tune_save_checkpoint()
            tune.report(objective=self.objective, **metrics)

    def _tune_save_checkpoint(self):
        if not self.use_tune_checkpoints:
            return
        with tune.checkpoint_dir(step=self.global_step) as checkpoint_dir:
            self.args.output_dir = checkpoint_dir
            output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")
            self.save_model(output_dir)
            if self.is_world_master():
                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))


    def _save_training(self, model, trial, metrics=None):
        # In all cases (even distributed/parallel), self.model is always a reference
        # to the model we want to save.
        if hasattr(model, "module"):
            assert model.module is self.model, f"Module {model.module} should be a reference to self.model"
        else:
            assert model is self.model, f"Model {model} should be a reference to self.model"
        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}"
        if self.hp_search_backend is not None and trial is not None:
            run_id = trial.number if self.hp_search_backend == HPSearchBackend.OPTUNA else tune.get_trial_id()
            checkpoint_folder += f"-run-{run_id}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

        self.store_flos()
        self.save_model(output_dir)

        # Save optimizer and scheduler
        if self.is_world_process_zero():
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            reissue_pt_warnings(caught_warnings)

        # Determine the new best metric / best model checkpoint
        if metrics is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.is_world_process_zero():
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        # Maybe delete some older checkpoints.
        if self.is_world_process_zero():
            self._rotate_checkpoints(use_mtime=True)



    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if self.state.best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(self.state.best_model_checkpoint)
            checkpoints_sorted[best_model_index], checkpoints_sorted[best_model_index][-1] = (
                checkpoints_sorted[-1],
                checkpoints_sorted[best_model_index],
            )
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)


    @staticmethod
    def _actual_model(
        model: Union[torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel, torch.nn.modules.Module]
    ) -> torch.nn.modules.Module:
        """

        Args:
            model: (:obj:`Union[torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel, torch.nn.modules.Module]`):
                Model object used during training

        Returns:
            :obj:`torch.nn.modules.Module`: unwrapped module
        """
        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        else:
            model = model
        return model
