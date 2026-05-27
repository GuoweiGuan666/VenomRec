#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import random
import numpy as np
import torch
import pprint
import yaml
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

def get_optimizer(optim, verbose=False):
    if optim == 'rms':
        if verbose:
            print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        if verbose:
            print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        if verbose:
            print("Optimizer: Using AdamW")
        optimizer = torch.optim.AdamW
    elif optim == 'adamax':
        if verbose:
            print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        if verbose:
            print("Optimizer: SGD")
        optimizer = torch.optim.SGD
    else:
        assert False, f"Please add your optimizer {optim} in the list."
    return optimizer

def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')
    parser.add_argument("--test", default=None)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--submit', action='store_true')
    # Checkpoint
    parser.add_argument('--output', type=str, default='snap/pretrain')
    parser.add_argument('--load', type=str, default=None, help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--from_scratch', action='store_true')
    # CPU/GPU
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument('--local_rank', type=int, default=-1)
    # Model Config
    parser.add_argument('--backbone', type=str, default='t5-base')
    parser.add_argument('--tokenizer', type=str, default='p5')
    parser.add_argument('--whole_word_embed', action='store_true')
    parser.add_argument('--category_embed', action='store_true')
    parser.add_argument('--max_text_length', type=int, default=128)
    parser.add_argument('--use_adapter', action="store_true")
    parser.add_argument('--reduction_factor', type=int, default=16)
    parser.add_argument('--add_adapter_cross_attn', type=str2bool, default=True)
    parser.add_argument('--use_lm_head_adapter', action="store_true")
    parser.add_argument('--use_single_adapter', action="store_true")
    parser.add_argument("--track_z", action="store_true")
    parser.add_argument('--unfreeze_layer_norms', action="store_true")
    parser.add_argument('--unfreeze_language_model', action="store_true")
    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--valid_batch_size', type=int, default=None)
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm', type=float, default=-1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--eval_start_epoch', type=int, default=4,
                        help='Run validation/monitoring beginning from this epoch index (default: 4).')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--losses", default='sequential,direct,explanation', type=str)
    parser.add_argument('--log_train_accuracy', action='store_true')
    parser.add_argument('--freeze_ln_statistics', action="store_true")
    parser.add_argument('--freeze_bn_statistics', action="store_true")
    # Training heuristics
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='Early stop training if valid loss does not improve for this many epochs (>0 enables, default 0 disables).')
    parser.add_argument('--monitor_hits', type=str2bool, default=False,
                        help='Enable validation Hit/NDCG monitoring for early stopping (default: False).')
    parser.add_argument('--monitor_hit_topk', type=int, default=10,
                        help='Top-K value used when computing validation Hit/NDCG metrics.')
    parser.add_argument('--monitor_hit_prompt', type=str, default='B-5',
                        help='Direct-task prompt ID (e.g., B-5) used for validation Hit/NDCG monitoring.')
    parser.add_argument('--monitor_hit_extra_topk', type=str, default='',
                        help='Comma-separated additional Top-K values (e.g. "50,100") to log alongside --monitor_hit_topk.')
    parser.add_argument('--monitor_hit_patience', type=int, default=-1,
                        help='Epoch patience for Hit/NDCG early stop (<=0 disables hit-based stopping).')
    parser.add_argument('--monitor_hit_batches', type=int, default=100,
                        help='Maximum validation batches sampled when computing Hit/NDCG (<=0 for all).')
    parser.add_argument('--monitor_hit_beams', type=int, default=10,
                        help='Beam width used for validation generation during Hit/NDCG monitoring.')
    parser.add_argument('--monitor_hit_tolerance', type=float, default=1e-5,
                        help='Required improvement threshold to treat validation Hit metric as better.')
    parser.add_argument('--monitor_hit_start_epoch', type=int, default=0,
                        help='Number of initial epochs to skip Hit/NDCG monitoring (warm-up).')
    parser.add_argument('--monitor_debug_samples', type=int, default=0,
                        help='When monitoring fails to parse items, log up to N raw targets/predictions (default 0 = off).')
    parser.add_argument('--monitor_target_rank', type=str2bool, default=False,
                        help='Enable per-epoch validation monitoring of a fixed target item rank on the direct task.')
    parser.add_argument('--monitor_target_rank_item', type=str, default='',
                        help='Target item id forced into every validation candidate list when --monitor_target_rank is enabled.')
    parser.add_argument('--monitor_target_rank_prompt', type=str, default='B-5',
                        help='Direct-task prompt used for target-rank monitoring (default: B-5).')
    parser.add_argument('--monitor_target_rank_batches', type=int, default=0,
                        help='Maximum validation batches for target-rank monitoring (<=0 for all).')
    parser.add_argument('--monitor_target_rank_chunk_size', type=int, default=32,
                        help='Number of candidate items scored per forward chunk during target-rank monitoring.')
    parser.add_argument('--monitor_target_rank_seed', type=int, default=2022,
                        help='Seed used to keep target-rank monitoring reproducible across epochs.')
    # Fake sample gradient attenuation to protect ranking quality
    parser.add_argument('--fake_sample_weight', type=float, default=1.0,
                        help='Per-sample loss multiplier for fake users (1.0 = no attenuation).')
    # Data sampling multipliers per task (train only)
    parser.add_argument('--train_seq_repeat', type=int, default=1,
                        help='Repeat factor for sequential task sampling in training (default 1).')
    parser.add_argument('--train_dir_repeat', type=int, default=1,
                        help='Repeat factor for direct task sampling in training (default 1).')
    parser.add_argument('--train_expl_repeat', type=int, default=1,
                        help='Repeat factor for explanation task sampling in training (default 1).')
    parser.add_argument('--candidate_target_repeat', type=int, default=10,
                        help='When training with fake users, repeat the target item in direct-task candidate lists this many times (default: 10).')
    # Poisoning safety knobs
    parser.add_argument('--direct_exclude_fake', type=str2bool, default=False,
                        help='When training with poisoned data (mr>0), exclude fake users from direct-task sampling to protect HR/NDCG (default: False for backwards compatibility).')
    # Adaptive task weighting
    parser.add_argument('--adaptive_task_weight', type=str2bool, default=True,
                        help='Enable adaptive task weighting (default: True).')
    parser.add_argument('--adaptive_method', type=str, default='dwa',
                        help='Adaptive scheme: dwa (Dynamic Weight Averaging) or none.')
    parser.add_argument('--adaptive_T', type=float, default=2.0,
                        help='Temperature for DWA (higher=flatter).')
    parser.add_argument('--adaptive_min_w', type=float, default=0.2,
                        help='Lower bound for task weight after normalization.')
    parser.add_argument('--adaptive_max_w', type=float, default=5.0,
                        help='Upper bound for task weight after normalization.')
    # Inference
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--gen_max_length', type=int, default=64)
    # Data & Configurations for Data Loading
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--data_root', type=str, default='data', help='Root directory for dataset')
    parser.add_argument('--original_file', type=str, default='sequential_data.txt', help='Original sequential data file name')
    parser.add_argument('--poisoned_file', type=str, default='sequential_data_poisoned.txt', help='Poisoned sequential data file name')
    parser.add_argument('--poison_subdir', type=str, default=None, help='Relative subdirectory under data/<split>/poisoned housing poisoned artefacts')
    # Visual features
    parser.add_argument('--image_feature_type', type=str, default='vitb32')
    parser.add_argument('--image_feature_size_ratio', type=int, default=2)
    parser.add_argument('--use_vis_layer_norm', default=True, type=str2bool)
    parser.add_argument('--train_visual_embedding', type=str2bool, default=True,
                        help='Whether to update visual_embedding parameters (default: True).')
    # Attack configuration
    parser.add_argument('--attack_mode', type=str, default="none",
                        help='Attack mode, e.g. "NoAttack" or "DirectBoostingAttack"')
    parser.add_argument('--mr', type=float, default=0.0,
                        help='Malicious user ratio (0.0 to 1.0)')
    # Others
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument("--dry", action='store_true')

    if parse:
        args = parser.parse_args()
    else:
        args = parser.parse_known_args()[0]

    # Load YAML config if exists
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config_from_yaml = yaml.safe_load(f) or {}
        print("Loaded YAML configuration:", config_from_yaml)

        # Overwrite attack_mode from suffix
        suffix = config_from_yaml.get('experiment', {}).get('suffix', 'NoAttack')
        args.attack_mode = suffix

        # Overwrite mr if specified
        mr_yaml = config_from_yaml.get('experiment', {}).get('mr', None)
        if mr_yaml is not None:
            args.mr = float(mr_yaml)

    # Turn into Config object
    kwargs = vars(args)
    kwargs.update(optional_kwargs)
    args = Config(**kwargs)

    # Optimizer
    args.optimizer = get_optimizer(args.optim, verbose=False)

    # Fix random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    return args

class Config(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        return 'Configurations\n' + self.config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f, Loader=yaml.FullLoader)
        return Config(**kwargs)

if __name__ == '__main__':
    args = parse_args(True)
    print("Parsed Arguments:")
    print(args)
    print("Configuration Details:")
    print(args.config_str)
