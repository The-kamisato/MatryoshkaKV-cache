# Copyright 2024 the LlamaFactory team.
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

from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from transformers import Trainer
from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers import LlamaModel, LlamaForCausalLM, AutoConfig
from transformers import GemmaModel, GemmaForCausalLM
from transformers.trainer_pt_utils import nested_detach

import torch.nn.functional as F
from ...extras.logging import get_logger
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimzer, create_custom_scheduler

import os
import random
import torch
from torch import nn
import torch.optim as optim
import safetensors.torch
import optuna
import warnings
from peft import PeftModel
import huggingface_hub.utils as hf_hub_utils

from transformers import Trainer, Seq2SeqTrainer
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint
from transformers.trainer_utils import (
    PredictionOutput,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    set_seed,
)
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    is_peft_available,
    is_sagemaker_mp_enabled,
)
from transformers.trainer_callback import TrainerState


TRAINER_STATE_NAME = "trainer_state.json"

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"

def check_tensor(tensor, step_name):
    if not torch.is_tensor(tensor):
        return
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    inf_mask = torch.isinf(tensor)
    inf_indices = torch.nonzero(inf_mask)


    if has_nan or has_inf:
        print(f"Step {step_name}:")
        if has_nan:
            print("  Contains NaN")
        if has_inf:
            print("  Contains Inf", inf_indices)

def get_matching_params(named_params, keys_to_match):
    if keys_to_match is not None:
        to_return = {k: t for k, t in named_params if (any(key_match in k for key_match in keys_to_match))}
    else:
        to_return = {k: t for k, t in named_params}
    return to_return

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def split_arg(arg):
    if isinstance(arg, str):
        return [item.strip() for item in arg.split(",")]
    return arg

def print_trainable_params(model):  
    total_trainable_params = 0  # 初始化可训练参数总数为0  
    for name, param in model.named_parameters():  
        if param.requires_grad:  
            print(f"para_name: {name}, para_num: {param.numel()}")  
            total_trainable_params += param.numel()  # 累加当前参数的元素数量  
    print(f"Total number of trainable parameters: {total_trainable_params}")

logger = get_logger(__name__)


class CustomTrainer(Trainer):
    r"""
    Inherits Trainer for custom optimizer.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

class PcaLlamaTrainer(CustomTrainer):
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        """

        if resume_from_checkpoint is False or resume_from_checkpoint == "False":
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:        # change this
                self._load_from_checkpoint(resume_from_checkpoint)
                if self.finetuning_args.additional_target is not None:
                    strings_to_match = split_arg(self.finetuning_args.additional_target)
                    for name, param in self.model.named_parameters():
                        if any(string in name for string in strings_to_match):
                            param.requires_grad = True
            resume_from_checkpoint = None
        
        if self.finetuning_args.finetuning_type == "freeze":
            print("only train proj!")
            strings_to_match = split_arg(self.finetuning_args.additional_target)
            for name, param in self.model.named_parameters():
                if any(string in name for string in strings_to_match):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
        print_trainable_params(self.model)
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly

            # state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            # if state.train_batch_size is not None:
            #     self._train_batch_size = state.train_batch_size
            

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
        if args.push_to_hub:
            try:
                # Disable progress bars when uploading models during checkpoints to avoid polluting stdout
                hf_hub_utils.disable_progress_bars()
                return inner_training_loop(
                    args=args,
                    resume_from_checkpoint=resume_from_checkpoint,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                )
            finally:
                hf_hub_utils.enable_progress_bars()
        else:
            self.model.config.all_layers_mean_key_states = None
            self.model.config.all_layers_mean_value_states = None
            self.model.config.all_layers_key_unitary_transform_matrix = None
            self.model.config.all_layers_value_unitary_transform_matrix = None
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )
        
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        
        ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            if self.finetuning_args.key_unitary_transform_lr and self.finetuning_args.value_unitary_transform_lr is not None:
                unitary_transform_parameters = [name for name, _ in opt_model.named_parameters() 
                                                if "key_unitary_transform_weights" in name or "value_unitary_transform_weights" in name
                                                or "mean_key_weights" in name or "mean_value_weights" in name]
                key_unitary_transform_parameters = [name for name, _ in opt_model.named_parameters() if "key_unitary_transform_weights" in name or "mean_key_weights" in name]
                value_unitary_transform_parameters = [name for name, _ in opt_model.named_parameters() if "value_unitary_transform_weights" in name or "mean_value_weights" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in unitary_transform_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in unitary_transform_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in key_unitary_transform_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.finetuning_args.key_unitary_transform_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in key_unitary_transform_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.finetuning_args.key_unitary_transform_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in value_unitary_transform_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.finetuning_args.value_unitary_transform_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in value_unitary_transform_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.finetuning_args.value_unitary_transform_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer
    
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=False,
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=False
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        keys_to_match = ["key_unitary_transform", "mean_key", "value_unitary_transform", "mean_value"]
        weight_to_save = get_matching_params(self.model.named_parameters(), keys_to_match)
        torch.save(weight_to_save, os.path.join(output_dir, f'unitary_transform_weight.bin'))
        
        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

class PcaLlamaDistillationTrainer(PcaLlamaTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Loading source model:")
        self.source_model_config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=False)
        self.source_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16).to(self.model.device)
        print("Load over")
    
    def compute_loss(self, model, inputs, return_outputs=False, T=1.0, soft_target_loss_weight=0.25, ce_loss_weight=0.75):
        with torch.no_grad():
            teacher_logits = self.source_model(**inputs)["logits"]

        if model.training:
            inputs["key_truncate_index"] = 16 * torch.randint(1, 9, (32, 32), dtype=torch.long)
            inputs["value_truncate_index"] = 16 * torch.randint(1, 9, (32, 32), dtype=torch.long)
        else:
            inputs["key_truncate_index"] = torch.full((32, 32), 64, dtype=torch.long)
            inputs["value_truncate_index"] = torch.full((32, 32), 64, dtype=torch.long)
            
        outputs = model(**inputs)

        soft_targets = F.softmax(teacher_logits / T, dim=-1)
        soft_prob = F.log_softmax(outputs["logits"] / T, dim=-1)

        # loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / (soft_prob.size()[0] * soft_prob.size()[1]) * (T**2)
        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / (soft_prob.size()[0] * soft_prob.size()[1]) * (T**2)
        label_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

        return (loss, outputs) if return_outputs else loss

    def compute_eval_loss(self, model, inputs, return_outputs=False, T=1.0):
        with torch.no_grad():
            teacher_logits = self.source_model(**inputs)["logits"]
        
        inputs["key_truncate_index"] = torch.full((32, 32), 64, dtype=torch.long)
        inputs["value_truncate_index"] = torch.full((32, 32), 64, dtype=torch.long)

        outputs = model(**inputs)

        soft_targets = F.softmax(teacher_logits / T, dim=-1)
        soft_prob = F.log_softmax(outputs["logits"] / T, dim=-1)

        loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / (soft_prob.size()[0] * soft_prob.size()[1]) * (T**2)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_eval_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
    
