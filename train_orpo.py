import transformers
from torch.optim import Optimizer
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
import math
import torch
import datasets
import trl
from peft import LoraModel, LoraConfig
import json
    
# Modified from HuggingFace LR schedulers code
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1,
    max_lr_decrease_percentage: float = 0.9
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        max_lr_decrease_percentage = max_lr_decrease_percentage
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float,
    # is set in get_cosine_schedule_with_warmup as default to 0.9
    max_lr_decrease_percentage: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))*max_lr_decrease_percentage+(1-max_lr_decrease_percentage)

class CustomTrainer(trl.ORPOTrainer):
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

def read_jsonl(path):
    data = []
    with open(path, "r") as file:
        for line in file:
            data+=[json.loads(line)]
    return data

def train(
        train_data_path: str,
        test_data_path: str,
        pretrained_model_path: str,
        batch_size: int = 2,
        lr: float = 1e-4,
        cache_dir: str|None = None,
        token: str|None = None,
        out_folder: str = "output"
        ) -> None:
    dataset_train = datasets.Dataset.from_list(read_jsonl(train_data_path))
    eval_dataset = datasets.Dataset.from_list(read_jsonl(test_data_path))

    model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_path, cache_dir = cache_dir, use_cache = False, torch_dtype=torch.bfloat16, token = token)
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_path, cache_dir = cache_dir, token = token)
    tokenizer.pad_token = tokenizer.eos_token
    
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.01,
    )

    # Do ORPO 'DPO-style' training, afterwards do RLOO with the reward model we have
    training_configs = trl.ORPOConfig(beta=0.1, output_dir=out_folder, num_train_epochs=3, report_to="wandb", per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size, logging_steps=1, max_grad_norm=1.0, gradient_accumulation_steps=1, do_eval=True, evaluation_strategy='epoch', save_strategy='epoch', eval_accumulation_steps=1, gradient_checkpointing=True, optim="adamw_torch", bf16=True, max_length=2048, max_prompt_length=1900, learning_rate=lr)
    
    training_configs.remove_unused_columns = False
    training_configs.label_names = ['labels']
    warmup = 0.0
    training_configs.warmup_steps = int(len(dataset_train)*warmup/(training_configs.per_device_train_batch_size*training_configs.gradient_accumulation_steps*training_configs.world_size))
    training_configs.warmup_steps = 0
    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_configs, train_dataset=dataset_train, eval_dataset=eval_dataset, peft_config=lora) 
    trainer.train()
    trainer.save_model(training_configs.output_dir)
    
if __name__ == '__main__':
    train_data_path = "orpo_train.jsonl"
    test_data_path = "orpo_test.jsonl"
    pretrained_model_path = "LLM-PBE/Llama3.1-8b-instruct-LLMPC-Blue-Team"
    # I was using RunPod machine, so stored in the workspace default folder
    cache_dir = "/workspace"
    token = "..." # HF token
    batch_size = 2
    lr = 1e-4
    train(train_data_path, test_data_path, pretrained_model_path, batch_size, lr, cache_dir, token)
