# src/train.py

#!/usr/bin/env python
import collections
import json
from datetime import datetime
import os
import random
from pathlib import Path
import logging
import re
import shutil
import math
import copy
import time
from packaging import version

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from param import parse_args
from data import get_loader
from utils import LossMeter, parse_item_id, load_target_items
from dist_utils import reduce_dict

_use_native_amp = False
_use_apex = False

# Check if PyTorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from trainer_base import TrainerBase, proj_dir

class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        assert args.whole_word_embed
        assert args.category_embed
        from model import VIP5Tuning

        model_kwargs = {}
        model_class = VIP5Tuning

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        self.model = self.create_model(model_class, config, **model_kwargs)

        if 'p5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)

        self.model.tokenizer = self.tokenizer

        # --- Adaptive task weighting state ---
        self.adaptive_on = bool(getattr(self.args, 'adaptive_task_weight', True))
        self.adaptive_method = str(getattr(self.args, 'adaptive_method', 'dwa')).lower()
        self.adaptive_T = float(getattr(self.args, 'adaptive_T', 2.0))
        self.adaptive_min_w = float(getattr(self.args, 'adaptive_min_w', 0.2))
        self.adaptive_max_w = float(getattr(self.args, 'adaptive_max_w', 5.0))
        self.task_names = [t.strip() for t in self.args.losses.split(',') if t.strip()]
        self.task_loss_history = {t: [] for t in self.task_names}
        self.task_weights = {t: 1.0 for t in self.task_names}
        self._adaptive_state_path = None  # filled after output is set externally (in __main__)

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            if args.load.endswith('.pth'):
                ckpt_path = args.load
            else:
                ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)
            # Only set start_epoch when args.load clearly encodes an Epoch number
            # Accept forms like .../Epoch12 or .../Epoch-12
            base = os.path.basename(args.load)
            m = re.search(r"Epoch-?(\d+)$", base)
            if m:
                try:
                    self.start_epoch = int(m.group(1))
                except Exception:
                    self.start_epoch = None

        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if args.gpu < 0:
            raise RuntimeError(f"Invalid GPU index {args.gpu}. Check CUDA_VISIBLE_DEVICES and GPU allocation.")
        
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)
        
        # Set which part of parameters as trainable
        self.freeze_whole_model()  # freeze whole parameters first
        self.unfreeze_parameters()  # unfreeze selected parameters

        # Calculate the percentage of trainable parameters (%)
        self.percent_updated_parameters = self.print_trainable_params_percentage(self.model)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()
            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif self.args.fp16 and _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True)
        if self.verbose:
            print(f'It took {time() - start:.1f}s')
        
        # 定义一个全局变量保存最佳验证 loss，确保跨 epoch 保存
        self.best_eval_loss = 1e6
        # Early stopping based on validation loss (patience <=0 disables)
        self.early_stop_patience = max(0, int(getattr(self.args, 'early_stop_patience', 0)))
        self._epochs_no_improve = 0

        # Hit/NDCG monitoring configuration
        self.hit_monitor_enabled = bool(getattr(self.args, 'monitor_hits', False))
        self.monitor_hit_topk = int(getattr(self.args, 'monitor_hit_topk', 10))
        self.monitor_hit_prompt = str(getattr(self.args, 'monitor_hit_prompt', 'B-5'))
        self.monitor_hit_patience = int(getattr(self.args, 'monitor_hit_patience', -1))
        self.monitor_hit_batches = int(getattr(self.args, 'monitor_hit_batches', 100))
        self.monitor_hit_beams = int(getattr(self.args, 'monitor_hit_beams', 10))
        self.monitor_hit_tolerance = float(getattr(self.args, 'monitor_hit_tolerance', 1e-5))
        self.monitor_hit_start_epoch = int(getattr(self.args, 'monitor_hit_start_epoch', 0))
        self.target_rank_monitor_enabled = bool(getattr(self.args, 'monitor_target_rank', False))
        self.monitor_target_rank_prompt = str(
            getattr(self.args, 'monitor_target_rank_prompt', self.monitor_hit_prompt or 'B-5')
        )
        self.monitor_target_rank_batches = int(getattr(self.args, 'monitor_target_rank_batches', 0))
        self.monitor_target_rank_chunk_size = max(
            1, int(getattr(self.args, 'monitor_target_rank_chunk_size', 32))
        )
        seed_fallback = int(getattr(self.args, 'seed', 2022) or 2022)
        self.monitor_target_rank_seed = int(
            getattr(self.args, 'monitor_target_rank_seed', seed_fallback) or seed_fallback
        )
        extra_topk_raw = str(getattr(self.args, 'monitor_hit_extra_topk', '') or '').strip()
        extra_values: set[int] = set()
        if extra_topk_raw:
            for chunk in extra_topk_raw.split(','):
                chunk = chunk.strip()
                if not chunk:
                    continue
                try:
                    val = int(chunk)
                except ValueError:
                    if self.verbose:
                        print(f"[monitor] Ignoring invalid extra top-k '{chunk}'")
                    continue
                if val > 0 and val != self.monitor_hit_topk:
                    extra_values.add(val)
        self.monitor_hit_extra_topk = sorted(extra_values)
        self.monitor_target_items = None
        if self.hit_monitor_enabled or self.target_rank_monitor_enabled:
            self.monitor_target_items = self._load_target_items(self.args.train)
            if self.hit_monitor_enabled and not self.monitor_target_items:
                if self.verbose:
                    print("[monitor] Unable to load target items; disabling Hit/NDCG monitoring.")
                self.hit_monitor_enabled = False
        raw_target_rank_item = str(getattr(self.args, 'monitor_target_rank_item', '') or '').strip()
        if not raw_target_rank_item and self.monitor_target_items:
            raw_target_rank_item = str(self.monitor_target_items[0])
        self.monitor_target_rank_item = raw_target_rank_item or None
        if self.target_rank_monitor_enabled and not self.monitor_target_rank_item:
            if self.verbose:
                print("[target-rank] No target item configured; disabling target-rank monitoring.")
            self.target_rank_monitor_enabled = False
        self.best_hit_metric = -1.0
        self.best_hit_epoch = None
        self.best_hit_ndcg = 0.0
        self.hit_no_improve = 0
        self.best_target_rank = float("inf")
        self.best_target_rank_epoch = None

        # Initialize adaptive state/logs after output dir exists
        try:
            out_dir = self.args.output
            os.makedirs(out_dir, exist_ok=True)
            self._adaptive_state_path = os.path.join(out_dir, 'adaptive_state.json')
            self._adaptive_tsv_path = os.path.join(out_dir, 'ADAPTIVE_TASK_WEIGHTS.tsv')
            # try resume adaptive state either from load dir or from output dir
            if self.adaptive_on:
                self._maybe_load_adaptive_state(args.load)
                # push weights to model
                if hasattr(self.model, 'set_task_weights'):
                    self.model.set_task_weights(self.task_weights)
                    # ensure model flag aligns
                    if hasattr(self.model, 'adaptive_task_weight'):
                        self.model.adaptive_task_weight = True
        except Exception:
            pass

    # ---------------- Adaptive Task Weighting utilities -----------------
    def _maybe_load_adaptive_state(self, load_tag: str = None) -> None:
        # first, try from output/adaptive_state.json
        def _load(path):
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    st = json.load(fh)
                hist = st.get('task_loss_history', {})
                for t in self.task_names:
                    if t in hist and isinstance(hist[t], list):
                        self.task_loss_history[t] = [float(x) for x in hist[t]]
                ws = st.get('task_weights', {})
                for t in self.task_names:
                    if t in ws:
                        self.task_weights[t] = float(ws[t])
                return True
            except Exception:
                return False

        if self._adaptive_state_path and os.path.exists(self._adaptive_state_path):
            _load(self._adaptive_state_path)
            return
        # next, try from directory of args.load if provided
        if load_tag:
            try:
                base = os.path.dirname(load_tag)
                cand = os.path.join(base, 'adaptive_state.json')
                if os.path.exists(cand):
                    _load(cand)
            except Exception:
                pass

    def _save_adaptive_state(self, epoch: int, per_task_avg: dict) -> None:
        if not self._adaptive_state_path:
            return
        try:
            state = {
                'epoch': int(epoch),
                'method': self.adaptive_method,
                'T': self.adaptive_T,
                'task_weights': {k: float(v) for k, v in self.task_weights.items()},
                'task_loss_history': {k: list(map(float, v)) for k, v in self.task_loss_history.items()},
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            with open(self._adaptive_state_path, 'w', encoding='utf-8') as fh:
                json.dump(state, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass
        # TSV log (append)
        try:
            header = 'epoch' + ''.join([f"\t{t}_avg" for t in self.task_names]) + ''.join([f"\t{t}_w" for t in self.task_names]) + '\n'
            line = f"{epoch}" + ''.join([f"\t{per_task_avg.get(t, 0.0):.6f}" for t in self.task_names]) + ''.join([f"\t{self.task_weights.get(t, 1.0):.6f}" for t in self.task_names]) + '\n'
            if not os.path.exists(self._adaptive_tsv_path):
                with open(self._adaptive_tsv_path, 'w', encoding='utf-8') as tf:
                    tf.write(header)
                    tf.write(line)
            else:
                with open(self._adaptive_tsv_path, 'a', encoding='utf-8') as tf:
                    tf.write(line)
        except Exception:
            pass

    def _update_adaptive_weights(self, epoch: int, per_task_avg: dict, save: bool = True) -> None:
        # Accumulate history
        for t in self.task_names:
            if t in per_task_avg and per_task_avg[t] is not None:
                self.task_loss_history[t].append(float(per_task_avg[t]))
        if not self.adaptive_on:
            return
        if self.adaptive_method not in ['dwa', 'dynamic', 'auto', 'autodwa']:
            # unsupported -> keep weights 1.0
            self.task_weights = {t: 1.0 for t in self.task_names}
            if hasattr(self.model, 'set_task_weights'):
                self.model.set_task_weights(self.task_weights)
            return
        # DWA: require at least 2 historical points
        ratios = []
        for t in self.task_names:
            hist = self.task_loss_history.get(t, [])
            if len(hist) < 2:
                ratios.append(1.0)
            else:
                prev, prev2 = hist[-1], hist[-2]
                r = prev / (prev2 + 1e-8)
                ratios.append(r)
        ratios = np.array(ratios, dtype=np.float64)
        # temperature scaling and softmax
        logits = ratios / max(self.adaptive_T, 1e-6)
        # numerical stability
        logits = logits - logits.max()
        exps = np.exp(logits)
        soft = exps / np.clip(exps.sum(), 1e-8, None)
        # scale to keep the average around 1.0
        weights = soft * len(self.task_names)
        # clip bounds
        weights = np.clip(weights, self.adaptive_min_w, self.adaptive_max_w)
        # renormalize to mean 1
        weights = weights / np.clip(weights.mean(), 1e-8, None)
        self.task_weights = {t: float(w) for t, w in zip(self.task_names, weights.tolist())}
        # apply to model
        if hasattr(self.model, 'set_task_weights'):
            self.model.set_task_weights(self.task_weights)
        # persist (rank0 only)
        if save:
            self._save_adaptive_state(epoch, per_task_avg)

    # 保存模型权重到指定目录，路径由 --output 参数传入
    def save(self, name):
        output_dir = self.args.output
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(output_dir, f"{name}.pth"))

    def train(self):
        LOSSES_NAME = self.args.LOSSES_NAME

        if self.args.dry:
            results = self.evaluate_epoch(epoch=0)

        if self.verbose:
            loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)

        if self.args.distributed:
            dist.barrier()

        global_step = 0
        eval_start_epoch = int(getattr(self.args, 'eval_start_epoch', 4))
        stop_training = False
        for epoch in range(self.args.epoch):
            if self.start_epoch is not None:
                epoch += self.start_epoch

            # Training阶段
            self.model.train()
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=275)

            epoch_results = {}
            for loss_name in LOSSES_NAME:
                epoch_results[loss_name] = 0.
                epoch_results[f'{loss_name}_count'] = 0

            for step_i, batch in enumerate(self.train_loader):
                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        results = self.model.module.train_step(batch) if self.args.distributed else self.model.train_step(batch)
                else:
                    results = self.model.module.train_step(batch) if self.args.distributed else self.model.train_step(batch)

                loss = results['loss']
                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

                if self.args.fp16 and _use_native_amp:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()

                for param in self.model.parameters():
                    param.grad = None

                global_step += 1

                if self.lr_scheduler:
                    lr = (self.lr_scheduler.get_last_lr()[0] 
                          if version.parse(torch.__version__) >= version.parse("1.4")
                          else self.lr_scheduler.get_lr()[0])
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                for k, v in results.items():
                    if k in epoch_results:
                        epoch_results[k] += v.item() if isinstance(v, torch.Tensor) else v

                if self.verbose and step_i % 200:
                    desc_str = f"Epoch {epoch} | LR {lr:.6f} |"
                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):
                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'
                    pbar.set_description(desc_str)
                    pbar.update(1)

                if self.args.distributed:
                    dist.barrier()

            if self.verbose:
                pbar.close()

            results = reduce_dict(epoch_results, average=False)
            if self.verbose:
                train_loss = results['total_loss']
                train_loss_count = results['total_loss_count']
                avg_train_loss = train_loss / train_loss_count
                losses_str = f"Train Loss: {avg_train_loss:.3f}\n"
                for name, loss in results.items():
                    if name.endswith('loss'):
                        loss_count = int(results[name+'_count'])
                        if loss_count > 0:
                            avg_loss = loss / loss_count
                            losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "
                losses_str += '\n'
                print(losses_str)

            # Update adaptive task weights from TRAIN averages (rank0 only)
            try:
                is_rank0 = (getattr(self.args, "local_rank", 0) in [0, -1]) or not self.args.distributed
                per_task_avg = {}
                for t in self.task_names:
                    key = f"{t}_loss"
                    cnt_key = f"{t}_loss_count"
                    if key in results and cnt_key in results and results[cnt_key] > 0:
                        per_task_avg[t] = float(results[key] / results[cnt_key])
                    else:
                        per_task_avg[t] = None
                # Update weights on all ranks to keep behavior consistent
                self._update_adaptive_weights(epoch, per_task_avg, save=is_rank0)
            except Exception:
                pass

            # Write per-epoch TRAIN loss breakdown and ratios (sidecar TSV)
            try:
                is_rank0 = (getattr(self.args, "local_rank", 0) in [0, -1]) or not self.args.distributed
                if is_rank0:
                    comp_names = [n for n in self.args.LOSSES_NAME if n.endswith('_loss') and n != 'total_loss']
                    comp_avgs = {}
                    for n in comp_names:
                        cnt_key = f"{n}_count"
                        if n in results and cnt_key in results and results[cnt_key] > 0:
                            comp_avgs[n] = float(results[n] / results[cnt_key])
                    total_avg = float(avg_train_loss)
                    denom = sum(comp_avgs.values()) or 1.0
                    ratios = {n: (v/denom) for n, v in comp_avgs.items()}
                    tsv = os.path.join(self.args.output, 'TRAIN_LOSS_BREAKDOWN.tsv')
                    header = 'epoch\t' + '\t'.join([f'{n}_avg' for n in comp_names]) + '\ttotal_avg\t' + '\t'.join([f'{n}_ratio' for n in comp_names]) + '\n'
                    line = f"{epoch}\t" + '\t'.join([f"{comp_avgs.get(n, 0.0):.6f}" for n in comp_names]) + f"\t{total_avg:.6f}\t" + '\t'.join([f"{ratios.get(n, 0.0):.6f}" for n in comp_names]) + '\n'
                    if not os.path.exists(tsv):
                        with open(tsv, 'w', encoding='utf-8') as fh:
                            fh.write(header)
                            fh.write(line)
                    else:
                        with open(tsv, 'a', encoding='utf-8') as fh:
                            fh.write(line)
            except Exception:
                pass

            if self.args.distributed:
                dist.barrier()
            if stop_training:
                break

            # 验证阶段：从 eval_start_epoch 开始触发验证 / 指标监控
            if epoch >= eval_start_epoch:
                valid_results = self.evaluate_epoch(epoch=epoch)
                valid_results = reduce_dict(valid_results, average=False)
                valid_loss = valid_results['total_loss']
                valid_loss_count = valid_results['total_loss_count']
                avg_valid_loss = valid_loss / valid_loss_count
                losses_str = f"Epoch {epoch}: Valid Loss: {avg_valid_loss:.3f}\n"
                for name, loss in valid_results.items():
                    if name.endswith('loss'):
                        loss_count = int(valid_results[name+'_count'])
                        if loss_count > 0:
                            avg_loss = loss / loss_count
                            losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "
                losses_str += '\n'
                print(losses_str)
                # 保存当前 epoch 模型（仅 rank0）
                is_rank0 = (getattr(self.args, "local_rank", 0) in [0, -1]) or not self.args.distributed
                if is_rank0:
                    self.save("Epoch%02d" % (epoch))
                # 如果当前验证 loss 优于历史最佳，则更新最佳验证 loss，并保存最佳模型 checkpoint
                prev_best = self.best_eval_loss
                improved = avg_valid_loss < (prev_best - 1e-9)

                if improved:
                    # Only rank 0 updates BEST to avoid multi-process overwrite
                    is_rank0 = (getattr(self.args, "local_rank", 0) in [0, -1]) or not self.args.distributed
                    if is_rank0:
                        try:
                            meta_path = os.path.join(self.args.output, "BEST_EVAL_LOSS.meta.json")
                            persisted_best = None
                            if os.path.exists(meta_path):
                                with open(meta_path, "r", encoding="utf-8") as fh:
                                    prev_meta = json.load(fh)
                                persisted_best = float(prev_meta.get("avg_valid_loss", 1e9))
                            if persisted_best is None or avg_valid_loss < (persisted_best - 1e-9):
                                self.best_eval_loss = avg_valid_loss
                                self.save("BEST_EVAL_LOSS")
                                print("Current Best Epoch: ", epoch)
                                self._write_best_sidecars(epoch, avg_valid_loss)
                            else:
                                # keep track of the best observed across restarts
                                self.best_eval_loss = min(self.best_eval_loss, persisted_best)
                        except Exception:
                            # Fallback to in-memory guard
                            self.best_eval_loss = avg_valid_loss
                            self.save("BEST_EVAL_LOSS")
                            print("Current Best Epoch: ", epoch)
                            self._write_best_sidecars(epoch, avg_valid_loss)
                    else:
                        self.best_eval_loss = avg_valid_loss

                # Early stopping book-keeping
                if self.early_stop_patience > 0:
                    if improved:
                        self._epochs_no_improve = 0
                    else:
                        self._epochs_no_improve += 1
                        if self._epochs_no_improve >= self.early_stop_patience:
                            print(f"Early stopping triggered at epoch {epoch} (no improvement for {self.early_stop_patience} epochs).")
                            stop_training = True
                            break
                if not stop_training and self.target_rank_monitor_enabled:
                    self._monitor_target_rank_after_validation_with_sync(epoch)
                if not stop_training and self.hit_monitor_enabled:
                    stop_training = self._monitor_hits_after_validation(epoch)
                if stop_training:
                    break
            else:
                print(f"Epoch {epoch}: Skip validation (eval_start_epoch={eval_start_epoch})")
                # 仅 rank0 保存
                is_rank0 = (getattr(self.args, "local_rank", 0) in [0, -1]) or not self.args.distributed
                if is_rank0:
                    self.save("Epoch%02d" % (epoch))
                if not stop_training and self.target_rank_monitor_enabled:
                    self._monitor_target_rank_after_validation_with_sync(epoch)
                if self.args.distributed:
                    dist.barrier()
            if stop_training:
                break

            # Write per-epoch VALID loss breakdown and ratios (sidecar TSV)
            try:
                is_rank0 = (getattr(self.args, "local_rank", 0) in [0, -1]) or not self.args.distributed
                if is_rank0:
                    comp_names = [n for n in self.args.LOSSES_NAME if n.endswith('_loss') and n != 'total_loss']
                    comp_avgs = {}
                    for n in comp_names:
                        cnt_key = f"{n}_count"
                        if n in valid_results and cnt_key in valid_results and valid_results[cnt_key] > 0:
                            comp_avgs[n] = float(valid_results[n] / valid_results[cnt_key])
                    total_avg = float(avg_valid_loss)
                    denom = sum(comp_avgs.values()) or 1.0
                    ratios = {n: (v/denom) for n, v in comp_avgs.items()}
                    tsv = os.path.join(self.args.output, 'VALID_LOSS_BREAKDOWN.tsv')
                    header = 'epoch\t' + '\t'.join([f'{n}_avg' for n in comp_names]) + '\ttotal_avg\t' + '\t'.join([f'{n}_ratio' for n in comp_names]) + '\n'
                    line = f"{epoch}\t" + '\t'.join([f"{comp_avgs.get(n, 0.0):.6f}" for n in comp_names]) + f"\t{total_avg:.6f}\t" + '\t'.join([f"{ratios.get(n, 0.0):.6f}" for n in comp_names]) + '\n'
                    if not os.path.exists(tsv):
                        with open(tsv, 'w', encoding='utf-8') as fh:
                            fh.write(header)
                            fh.write(line)
                    else:
                        with open(tsv, 'a', encoding='utf-8') as fh:
                            fh.write(line)
            except Exception:
                pass
        # Finalize BEST: ensure the on-disk BEST matches the true minimum in history
        try:
            is_rank0 = (getattr(self.args, "local_rank", 0) in [0, -1]) or not self.args.distributed
            if is_rank0:
                self._finalize_best_from_history()
        except Exception:
            pass

    def evaluate_epoch(self, epoch):
        LOSSES_NAME = self.args.LOSSES_NAME
        epoch_results = {}
        for loss_name in LOSSES_NAME:
            epoch_results[loss_name] = 0.
            epoch_results[f'{loss_name}_count'] = 0

        self.model.eval()
        with torch.no_grad():
            if self.verbose:
                loss_meter = LossMeter()
                loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
                pbar = tqdm(total=len(self.val_loader), ncols=275)
            for step_i, batch in enumerate(self.val_loader):
                results = (self.model.module.valid_step(batch)
                           if self.args.distributed else self.model.valid_step(batch))
                for k, v in results.items():
                    if k in epoch_results:
                        epoch_results[k] += v.item() if isinstance(v, torch.Tensor) else v
                if self.verbose and step_i % 200:
                    desc_str = f"Valid Epoch {epoch} |"
                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):
                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'
                    pbar.set_description(desc_str)
                    pbar.update(1)
                if self.args.distributed:
                    dist.barrier()
            if self.verbose:
                pbar.close()
            if self.args.distributed:
                dist.barrier()
            return epoch_results

    # --- hit@k monitoring helpers -------------------------------------------
    def _load_target_items(self, split: str):
        items = []
        if not split:
            return items
        try:
            items = load_target_items(split, proj_dir)
        except Exception as exc:
            if self.verbose:
                print(f"[monitor] Failed to load target item list for {split}: {exc}")
            items = []
        # stash in args for evaluation/monitoring routines that expect data_target
        if items:
            mapping = getattr(self.args, 'data_target', {})
            mapping[split] = items
            self.args.data_target = mapping
        return items

    def _encode_candidate_targets(self, candidates: list[str]) -> torch.Tensor:
        encoded = [
            self.tokenizer.encode(
                str(item),
                padding=True,
                truncation=True,
                max_length=self.args.gen_max_length,
            )
            for item in candidates
        ]
        max_len = max((len(ids) for ids in encoded), default=1)
        labels = torch.ones(len(encoded), max_len, dtype=torch.long) * self.tokenizer.pad_token_id
        for idx, ids in enumerate(encoded):
            if ids:
                labels[idx, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask = labels != self.tokenizer.pad_token_id
        labels[~mask] = -100
        return labels

    @torch.no_grad()
    def _score_candidate_losses(
        self,
        model,
        sample_inputs: dict[str, torch.Tensor],
        candidates: list[str],
        device: torch.device,
    ) -> list[float]:
        labels = self._encode_candidate_targets(candidates).to(device)
        losses: list[float] = []
        chunk_size = self.monitor_target_rank_chunk_size
        for start in range(0, len(candidates), chunk_size):
            end = min(len(candidates), start + chunk_size)
            label_chunk = labels[start:end]
            reps = end - start
            output = model(
                input_ids=sample_inputs['input_ids'].repeat(reps, 1),
                whole_word_ids=sample_inputs['whole_word_ids'].repeat(reps, 1),
                category_ids=sample_inputs['category_ids'].repeat(reps, 1),
                vis_feats=sample_inputs['vis_feats'].repeat(reps, 1, 1),
                labels=label_chunk,
                return_dict=True,
                task='direct',
            )
            lm_mask = (label_chunk != -100).float()
            B, L = label_chunk.size()
            chunk_loss = output['loss'].view(B, L) * lm_mask
            chunk_loss = chunk_loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)
            losses.extend(chunk_loss.detach().cpu().tolist())
        return losses

    @torch.no_grad()
    def _compute_val_target_rank_metrics(self):
        if not self.target_rank_monitor_enabled or not self.monitor_target_rank_item:
            return None

        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        try:
            random.seed(self.monitor_target_rank_seed)
            np.random.seed(self.monitor_target_rank_seed)
            torch.manual_seed(self.monitor_target_rank_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.monitor_target_rank_seed)

            eval_args = copy.deepcopy(self.args)
            eval_args.distributed = False
            eval_args.multiGPU = False
            eval_args.local_rank = 0
            eval_args.world_size = 1
            eval_args.gpu = self.args.gpu
            eval_args.batch_size = getattr(self.args, 'valid_batch_size', None) or self.args.batch_size
            eval_args.num_workers = 0
            eval_args.eval_target_item = str(self.monitor_target_rank_item)

            try:
                val_loader = get_loader(
                    eval_args,
                    {'direct': [self.monitor_target_rank_prompt]},
                    {'direct': (1, 1)},
                    split=eval_args.valid,
                    mode='val',
                    batch_size=eval_args.batch_size,
                    workers=0,
                    distributed=False
                )
            except Exception as exc:
                if self.verbose:
                    print(
                        f"[target-rank] Failed to build validation loader for prompt "
                        f"{self.monitor_target_rank_prompt}: {exc}"
                    )
                return None

            model = self.model.module if self.args.distributed else self.model
            model.eval()
            device = torch.device(f'cuda:{self.args.gpu}')
            target_item = str(self.monitor_target_rank_item)
            world_size = dist.get_world_size() if self.args.distributed and dist.is_available() and dist.is_initialized() else 1
            dist_rank = dist.get_rank() if self.args.distributed and dist.is_available() and dist.is_initialized() else 0
            local_ranks: list[int] = []
            local_candidate_sizes: list[int] = []
            local_target_missing = 0
            global_sample_idx = 0

            for step, batch in enumerate(val_loader):
                if self.monitor_target_rank_batches > 0 and step >= self.monitor_target_rank_batches:
                    break
                if batch['task'][0] != 'direct':
                    continue
                candidate_batches = batch.get('candidate_items') or []
                if not candidate_batches:
                    continue
                for idx, candidates in enumerate(candidate_batches):
                    assigned_rank = global_sample_idx % world_size
                    global_sample_idx += 1
                    if assigned_rank != dist_rank:
                        continue
                    candidate_list = [str(item) for item in candidates]
                    if target_item not in candidate_list:
                        local_target_missing += 1
                        continue
                    sample_inputs = {
                        'input_ids': batch['input_ids'][idx:idx+1].to(device),
                        'whole_word_ids': batch['whole_word_ids'][idx:idx+1].to(device),
                        'category_ids': batch['category_ids'][idx:idx+1].to(device),
                        'vis_feats': batch['vis_feats'][idx:idx+1].to(device),
                    }
                    losses = self._score_candidate_losses(model, sample_inputs, candidate_list, device)
                    order = sorted(range(len(losses)), key=lambda cand_idx: (losses[cand_idx], cand_idx))
                    target_idx = candidate_list.index(target_item)
                    item_rank = order.index(target_idx) + 1
                    local_ranks.append(item_rank)
                    local_candidate_sizes.append(len(candidate_list))

            if self.args.distributed and dist.is_available() and dist.is_initialized():
                gathered = [None for _ in range(world_size)]
                payload = {
                    'ranks': local_ranks,
                    'candidate_sizes': local_candidate_sizes,
                    'target_missing': int(local_target_missing),
                }
                dist.all_gather_object(gathered, payload)
                if dist_rank != 0:
                    return None
                ranks: list[int] = []
                candidate_sizes: list[int] = []
                target_missing = 0
                for item in gathered:
                    if not item:
                        continue
                    ranks.extend(item.get('ranks') or [])
                    candidate_sizes.extend(item.get('candidate_sizes') or [])
                    target_missing += int(item.get('target_missing', 0))
            else:
                ranks = local_ranks
                candidate_sizes = local_candidate_sizes
                target_missing = local_target_missing

            if not ranks:
                return None

            return {
                'target_item': target_item,
                'prompt': self.monitor_target_rank_prompt,
                'avg_rank': float(sum(ranks) / len(ranks)),
                'median_rank': float(np.median(ranks)),
                'min_rank': int(min(ranks)),
                'max_rank': int(max(ranks)),
                'count': int(len(ranks)),
                'mean_candidate_count': float(np.mean(candidate_sizes)) if candidate_sizes else 0.0,
                'target_missing': int(target_missing),
            }
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _append_target_rank_history(self, epoch: int, metrics: dict[str, float]) -> None:
        try:
            out_dir = self.args.output
            os.makedirs(out_dir, exist_ok=True)
            history_path = os.path.join(out_dir, "TARGET_RANK_HISTORY.tsv")
            header = (
                "timestamp\tepoch\ttarget_item\tprompt\tavg_rank\tmedian_rank\t"
                "min_rank\tmax_rank\tcount\tmean_candidate_count\ttarget_missing\n"
            )
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = (
                f"{ts}\t{int(epoch)}\t{metrics['target_item']}\t{metrics['prompt']}\t"
                f"{metrics['avg_rank']:.6f}\t{metrics['median_rank']:.6f}\t"
                f"{int(metrics['min_rank'])}\t{int(metrics['max_rank'])}\t"
                f"{int(metrics['count'])}\t{metrics['mean_candidate_count']:.6f}\t"
                f"{int(metrics.get('target_missing', 0))}\n"
            )
            if not os.path.exists(history_path):
                with open(history_path, "w", encoding="utf-8") as fh:
                    fh.write(header)
                    fh.write(line)
            else:
                with open(history_path, "a", encoding="utf-8") as fh:
                    fh.write(line)
        except Exception:
            pass

    def _rank0_sync_marker_path(self, tag: str, epoch: int) -> str:
        return os.path.join(self.args.output, f".rank0_sync_{tag}_epoch{int(epoch):02d}.json")

    def _run_rank0_sidework_with_file_sync(self, tag: str, epoch: int, fn):
        if not self.args.distributed:
            return fn()

        is_rank0 = getattr(self.args, 'local_rank', 0) in [0, -1]
        sync_path = self._rank0_sync_marker_path(tag, epoch)

        if is_rank0:
            os.makedirs(self.args.output, exist_ok=True)
            try:
                os.remove(sync_path)
            except FileNotFoundError:
                pass

            status = {
                "ok": True,
                "tag": str(tag),
                "epoch": int(epoch),
                "timestamp": None,
            }
            try:
                return fn()
            except Exception as exc:
                status["ok"] = False
                status["error"] = repr(exc)
                raise
            finally:
                status["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(sync_path, "w", encoding="utf-8") as fh:
                    json.dump(status, fh, ensure_ascii=False, indent=2)

        timeout_seconds = max(int(os.environ.get("NCCL_TIMEOUT", "7200")), 600)
        deadline = time.monotonic() + float(timeout_seconds)
        while not os.path.exists(sync_path):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for rank0 sidework '{tag}' at epoch {epoch}. "
                    f"Expected sync marker: {sync_path}"
                )
            time.sleep(5)

        try:
            with open(sync_path, "r", encoding="utf-8") as fh:
                status = json.load(fh)
        except Exception:
            status = {"ok": True}

        if not status.get("ok", True):
            raise RuntimeError(
                f"rank0 sidework '{tag}' failed at epoch {epoch}: "
                f"{status.get('error', 'unknown error')}"
            )

        return None

    def _write_target_rank_sidecars(self, epoch: int, metrics: dict[str, float]) -> None:
        try:
            out_dir = self.args.output
            os.makedirs(out_dir, exist_ok=True)
            meta = {
                "epoch": int(epoch),
                "target_item": str(metrics['target_item']),
                "prompt": str(metrics['prompt']),
                "avg_rank": float(metrics['avg_rank']),
                "median_rank": float(metrics['median_rank']),
                "min_rank": int(metrics['min_rank']),
                "max_rank": int(metrics['max_rank']),
                "count": int(metrics['count']),
                "mean_candidate_count": float(metrics['mean_candidate_count']),
                "target_missing": int(metrics.get('target_missing', 0)),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "checkpoint": f"Epoch{int(epoch):02d}.pth",
                "run_name": getattr(self.args, "run_name", ""),
            }
            with open(os.path.join(out_dir, "BEST_TARGET_RANK.meta.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _monitor_target_rank_after_validation_with_sync(self, epoch: int) -> None:
        if not self.target_rank_monitor_enabled:
            return
        self._monitor_target_rank_after_validation(epoch)

    def _monitor_target_rank_after_validation(self, epoch: int) -> None:
        if not self.target_rank_monitor_enabled:
            return
        metrics = self._compute_val_target_rank_metrics()
        if not metrics:
            return
        print(
            f"[target-rank] Epoch {epoch} target={metrics['target_item']} "
            f"prompt={metrics['prompt']} avg_rank={metrics['avg_rank']:.4f} "
            f"median={metrics['median_rank']:.4f} count={metrics['count']} "
            f"candidates={metrics['mean_candidate_count']:.2f}"
        )
        self._append_target_rank_history(epoch, metrics)
        if metrics['avg_rank'] < self.best_target_rank:
            self.best_target_rank = float(metrics['avg_rank'])
            self.best_target_rank_epoch = int(epoch)
            self._write_target_rank_sidecars(epoch, metrics)

    @torch.no_grad()
    def _compute_val_hit_metrics(self):
        if not self.monitor_target_items:
            return None
        if self.args.distributed and getattr(self.args, 'local_rank', 0) not in [0, -1]:
            return None
        eval_args = copy.deepcopy(self.args)
        eval_args.distributed = False
        eval_args.multiGPU = False
        eval_args.local_rank = 0
        eval_args.world_size = 1
        eval_args.gpu = self.args.gpu
        eval_args.batch_size = getattr(self.args, 'valid_batch_size', None) or self.args.batch_size
        eval_args.eval_target_item = None
        try:
            val_loader = get_loader(
                eval_args,
                {'direct': [self.monitor_hit_prompt]},
                {'direct': (1, 1)},
                split=eval_args.valid,
                mode='val',
                batch_size=eval_args.batch_size,
                workers=eval_args.num_workers,
                distributed=False
            )
        except Exception as exc:
            if self.verbose:
                print(f"[monitor] Failed to build validation loader for prompt {self.monitor_hit_prompt}: {exc}")
            return None

        model = self.model.module if self.args.distributed else self.model
        model.eval()
        device = torch.device(f'cuda:{self.args.gpu}')
        requested_topk = [self.monitor_hit_topk] + list(self.monitor_hit_extra_topk)
        requested_topk = [k for k in requested_topk if k and k > 0]
        if not requested_topk:
            requested_topk = [10]
        requested_topk = sorted(set(requested_topk))
        primary_topk = requested_topk[0] if self.monitor_hit_topk <= 0 else self.monitor_hit_topk
        if primary_topk not in requested_topk:
            requested_topk.insert(0, primary_topk)
        max_topk = max(requested_topk)

        num_beams = max(max_topk, max(1, self.monitor_hit_beams))
        total_processed = 0
        metric_sums = {
            k: {
                'hit': 0.0,
                'ndcg': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'mrr': 0.0,
                'map': 0.0,
                'er': 0.0,
            }
            for k in requested_topk
        }
        observation_counts = {k: 0 for k in requested_topk}
        parse_fail_targets = 0
        parse_fail_preds = 0
        debug_cap = int(getattr(self.args, 'monitor_debug_samples', 0) or 0)
        debug_records = []
        split_key = getattr(self.args, 'split', None) or getattr(self.args, 'train', None)
        target_items = [str(t) for t in getattr(self.args, 'data_target', {}).get(split_key, [])]
        target_id_set = set()
        for t in target_items:
            tid = parse_item_id(t)
            if tid is not None:
                target_id_set.add(tid)

        probe_logs: list[dict[str, object]] = []
        for step, batch in enumerate(val_loader):
            if self.monitor_hit_batches > 0 and step >= self.monitor_hit_batches:
                break
            if batch['task'][0] != 'direct':
                continue
            input_ids = batch['input_ids'].to(device)
            whole_word_ids = batch['whole_word_ids'].to(device)
            category_ids = batch['category_ids'].to(device)
            vis_feats = batch['vis_feats'].to(device)

            beam_outputs = model.generate(
                input_ids=input_ids,
                whole_word_ids=whole_word_ids,
                category_ids=category_ids,
                vis_feats=vis_feats,
                task=batch['task'][0],
                max_length=50,
                num_beams=num_beams,
                num_return_sequences=max_topk,
                no_repeat_ngram_size=0,
                early_stopping=True
            )
            generated = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)

            for j, tgt_text in enumerate(batch['target_text']):
                target = parse_item_id(tgt_text)
                if target is None:
                    parse_fail_targets += 1
                    if debug_cap and len(debug_records) < debug_cap:
                        debug_records.append({
                            'type': 'target',
                            'raw': str(tgt_text)
                        })
                    continue
                preds = []
                slice_start = j * max_topk
                slice_end = slice_start + max_topk
                for cand in generated[slice_start:slice_end]:
                    item_id = parse_item_id(cand)
                    if item_id is None:
                        parse_fail_preds += 1
                        if debug_cap and len(debug_records) < debug_cap:
                            debug_records.append({
                                'type': 'pred',
                                'raw': str(cand)
                            })
                        continue
                    preds.append(item_id)
                if not preds:
                    continue
                total_processed += 1
                try:
                    target_pos = preds.index(target)
                except ValueError:
                    target_pos = None

                for k in requested_topk:
                    actual_k = min(len(preds), k)
                    if actual_k <= 0:
                        continue
                    subset_pos = target_pos if target_pos is None else target_pos
                    hit_flag = subset_pos is not None and subset_pos < actual_k
                    observation_counts[k] += 1
                    if hit_flag:
                        ndcg_val = 1.0 / math.log2(subset_pos + 2.0)
                        mrr_val = 1.0 / (subset_pos + 1.0)
                        map_val = mrr_val
                    else:
                        ndcg_val = 0.0
                        mrr_val = 0.0
                        map_val = 0.0
                    metric_sums[k]['hit'] += 1.0 if hit_flag else 0.0
                    metric_sums[k]['ndcg'] += ndcg_val
                    metric_sums[k]['precision'] += (1.0 if hit_flag else 0.0) / max(1, actual_k)
                    metric_sums[k]['recall'] += 1.0 if hit_flag else 0.0
                    metric_sums[k]['mrr'] += mrr_val
                    metric_sums[k]['map'] += map_val

        if total_processed == 0:
            return None
        er_metric_sums = {k: 0.0 for k in requested_topk}
        er_counts = {k: 0 for k in requested_topk}
        if target_items:
            for tgt in target_items:
                tgt_id = parse_item_id(tgt)
                if tgt_id is None:
                    continue
                eval_args.eval_target_item = tgt
                try:
                    er_loader = get_loader(
                        eval_args,
                        {'direct': [self.monitor_hit_prompt]},
                        {'direct': (1, 1)},
                        split=eval_args.valid,
                        mode='val',
                        batch_size=eval_args.batch_size,
                        workers=eval_args.num_workers,
                        distributed=False
                    )
                except Exception:
                    continue
                for step, batch in enumerate(er_loader):
                    if self.monitor_hit_batches > 0 and step >= self.monitor_hit_batches:
                        break
                    if batch['task'][0] != 'direct':
                        continue
                    input_ids = batch['input_ids'].to(device)
                    whole_word_ids = batch['whole_word_ids'].to(device)
                    category_ids = batch['category_ids'].to(device)
                    vis_feats = batch['vis_feats'].to(device)
                    beam_outputs = model.generate(
                        input_ids=input_ids,
                        whole_word_ids=whole_word_ids,
                        category_ids=category_ids,
                        vis_feats=vis_feats,
                        task=batch['task'][0],
                        max_length=50,
                        num_beams=num_beams,
                        num_return_sequences=max_topk,
                        no_repeat_ngram_size=0,
                        early_stopping=True
                    )
                    generated = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
                    for j in range(len(batch['target_text'])):
                        preds = []
                        slice_start = j * max_topk
                        slice_end = slice_start + max_topk
                        for cand in generated[slice_start:slice_end]:
                            item_id = parse_item_id(cand)
                            if item_id is not None:
                                preds.append(item_id)
                        if not preds:
                            continue
                        if debug_cap and len(probe_logs) < debug_cap:
                            rank = None
                            if tgt_id in preds:
                                rank = preds.index(tgt_id) + 1
                            probe_logs.append({
                                "context": "er",
                                "target": tgt,
                                "rank": rank,
                                "topk": preds[:10],
                            })
                        for k in requested_topk:
                            actual_k = min(len(preds), k)
                            if actual_k <= 0:
                                continue
                            er_counts[k] += 1
                            if tgt_id in preds[:actual_k]:
                                er_metric_sums[k] += 1.0
            eval_args.eval_target_item = None

        snapshot_dir = Path(self.args.output)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        probe_path = snapshot_dir / "probe_recommendations.tsv"
        if (parse_fail_targets or parse_fail_preds) and self.args.local_rank in (0, -1):
            msg = (f"[monitor] parse failures: targets={parse_fail_targets}, preds={parse_fail_preds}")
            if debug_records:
                msg += " | samples=" + "; ".join(
                    f"{rec['type']}='{rec['raw']}'" for rec in debug_records)
            print(msg)
        if probe_logs and self.args.local_rank in (0, -1):
            print("[monitor] sample recommendation probes:")
            for rec in probe_logs[:debug_cap]:
                ctx = rec.get("context")
                info = f"ctx={ctx}"
                if ctx == "er":
                    info += f", target={rec.get('target')}, rank={rec.get('rank')}"
                if rec.get("hits"):
                    info += f", hits={rec.get('hits')}"
                info += f", topk={rec.get('topk')}"
                print("    ", info)
            with probe_path.open("a", encoding="utf-8") as pf:
                for rec in probe_logs[:debug_cap]:
                    ctx = rec.get("context")
                    target = rec.get("target", "")
                    rank = rec.get("rank", "")
                    hits = rec.get("hits", "")
                    topk = rec.get("topk", [])
                    pf.write(
                        f"{datetime.now().isoformat()}	{ctx}	{target}	{rank}	{hits}	{','.join(map(str, topk))}\n"
                    )
        def _average_metrics(k: int) -> dict[str, float]:
            count = observation_counts.get(k, 0)
            if count == 0:
                return {
                    'hit': 0.0,
                    'ndcg': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'mrr': 0.0,
                    'map': 0.0,
                    'count': 0,
                }
            result = {
                'hit': metric_sums[k]['hit'] / count,
                'ndcg': metric_sums[k]['ndcg'] / count,
                'precision': metric_sums[k]['precision'] / count,
                'recall': metric_sums[k]['recall'] / count,
                'mrr': metric_sums[k]['mrr'] / count,
                'map': metric_sums[k]['map'] / count,
                'count': count,
            }
            if er_counts.get(k, 0) > 0:
                result['er'] = er_metric_sums[k] / er_counts[k]
            else:
                result['er'] = 0.0
            return result

        primary_k = self.monitor_hit_topk if self.monitor_hit_topk > 0 else requested_topk[0]
        primary_metrics = _average_metrics(primary_k)
        extras = {
            k: _average_metrics(k)
            for k in requested_topk
            if k != primary_k
        }
        result = dict(primary_metrics)
        result['count'] = total_processed
        if extras:
            result['multi'] = extras
        return result

    def _append_hit_history(self, epoch: int, metrics: dict[str, float]) -> None:
        try:
            out_dir = self.args.output
            os.makedirs(out_dir, exist_ok=True)
            history_path = os.path.join(out_dir, "HIT_HISTORY.tsv")
            k = self.monitor_hit_topk
            extras = metrics.get('multi') if isinstance(metrics.get('multi'), dict) else None
            extra_cols = ''
            if extras:
                for extra_k in sorted(extras):
                    extra_cols += f"\ter@{extra_k}"
            header = (
                "timestamp\tepoch\thit@{k}\tndcg@{k}\tprecision@{k}\trecall@{k}\t"
                "mrr@{k}\tmap@{k}\ter@{k}"
            ).format(k=k)
            header += extra_cols + "\n"
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = (
                f"{ts}\t{int(epoch)}\t{metrics['hit']:.6f}\t{metrics['ndcg']:.6f}\t"
                f"{metrics['precision']:.6f}\t{metrics['recall']:.6f}\t"
                f"{metrics['mrr']:.6f}\t{metrics['map']:.6f}\t{metrics['er']:.6f}"
            )
            if extras:
                for extra_k in sorted(extras):
                    line += f"\t{extras[extra_k].get('er', 0.0):.6f}"
            line += "\n"
            if extras and os.path.exists(history_path):
                need_update = False
                try:
                    with open(history_path, "r", encoding="utf-8") as fh:
                        existing_lines = fh.readlines()
                    if existing_lines:
                        header_line = existing_lines[0].rstrip("\n")
                        for extra_k in sorted(extras):
                            if f"\ter@{extra_k}" not in header_line:
                                need_update = True
                                break
                        if need_update:
                            new_header_line = header_line + ''.join(f"\ter@{extra_k}" for extra_k in sorted(extras))
                            updated_lines = [new_header_line + "\n"]
                            for old_line in existing_lines[1:]:
                                updated_lines.append(old_line.rstrip("\n") + ''.join('\t0.000000' for _ in extras) + "\n")
                            with open(history_path, "w", encoding="utf-8") as fh:
                                fh.writelines(updated_lines)
                except Exception:
                    pass
            if not os.path.exists(history_path):
                with open(history_path, "w", encoding="utf-8") as tf:
                    tf.write(header)
                    tf.write(line)
            else:
                with open(history_path, "a", encoding="utf-8") as tf:
                    tf.write(line)
        except Exception:
            pass

    def _write_hit_sidecars(self, epoch: int, metrics: dict[str, float]) -> None:
        try:
            out_dir = self.args.output
            os.makedirs(out_dir, exist_ok=True)
            meta = {
                "epoch": int(epoch),
                "hit_topk": int(self.monitor_hit_topk),
                "hit": float(metrics['hit']),
                "ndcg": float(metrics['ndcg']),
                "precision": float(metrics['precision']),
                "recall": float(metrics['recall']),
                "mrr": float(metrics['mrr']),
                "map": float(metrics['map']),
                "er": float(metrics['er']),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "checkpoint": "BEST_HIT.pth",
                "run_name": getattr(self.args, "run_name", ""),
            }
            extras = metrics.get('multi') if isinstance(metrics.get('multi'), dict) else None
            if extras:
                meta['extra_er'] = {
                    str(k): float(info.get('er', 0.0)) for k, info in extras.items()
                }
            with open(os.path.join(out_dir, "BEST_HIT.meta.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _monitor_hits_after_validation(self, epoch: int) -> bool:
        if not self.hit_monitor_enabled:
            return False
        if epoch < self.monitor_hit_start_epoch:
            return False
        hit_val = -1.0
        ndcg_val = 0.0
        er_val = 0.0
        metrics: dict[str, float] | None = None
        if (not self.args.distributed) or (getattr(self.args, 'local_rank', 0) in [0, -1]):
            metrics = self._compute_val_hit_metrics()

        vals = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        multi_metrics = None
        if metrics:
            multi_metrics = metrics.pop('multi', None)
            vals = [
                float(metrics.get('hit', -1.0)),
                float(metrics.get('ndcg', 0.0)),
                float(metrics.get('precision', 0.0)),
                float(metrics.get('recall', 0.0)),
                float(metrics.get('mrr', 0.0)),
                float(metrics.get('map', 0.0)),
                float(metrics.get('er', 0.0)),
            ]

        device = torch.device(f'cuda:{self.args.gpu}')
        tensor = torch.tensor(vals, device=device)
        if self.args.distributed:
            dist.broadcast(tensor, src=0)
        vals = tensor.tolist()
        hit_val = vals[0]
        if hit_val < 0:
            return False
        metrics = {
            'hit': vals[0],
            'ndcg': vals[1],
            'precision': vals[2],
            'recall': vals[3],
            'mrr': vals[4],
            'map': vals[5],
            'er': vals[6],
        }
        if multi_metrics and self.args.local_rank in [0, -1]:
            metrics['multi'] = multi_metrics
        if self.args.local_rank in [0, -1]:
            print(
                f"[monitor] Epoch {epoch} Hit@{self.monitor_hit_topk}: {metrics['hit']:.4f} | "
                f"NDCG: {metrics['ndcg']:.4f} | Precision: {metrics['precision']:.4f} | "
                f"Recall: {metrics['recall']:.4f} | MRR: {metrics['mrr']:.4f} | "
                f"MAP: {metrics['map']:.4f} | ER@{self.monitor_hit_topk}: {metrics['er']:.4f}"
            )
            if multi_metrics:
                extra_msgs = []
                for k in sorted(multi_metrics):
                    extra_er = multi_metrics[k].get('er', 0.0)
                    extra_msgs.append(f"ER@{k}: {extra_er:.4f}")
                if extra_msgs:
                    print("[monitor]    " + " | ".join(extra_msgs))
        is_rank0 = (getattr(self.args, "local_rank", 0) in [0, -1]) or not self.args.distributed
        if is_rank0:
            self._append_hit_history(epoch, metrics)
        if metrics['hit'] > self.best_hit_metric + self.monitor_hit_tolerance:
            self.best_hit_metric = metrics['hit']
            self.best_hit_ndcg = metrics['ndcg']
            self.best_hit_epoch = epoch
            self.hit_no_improve = 0
            if is_rank0:
                self.save("BEST_HIT")
                self._write_hit_sidecars(epoch, metrics)
        else:
            self.hit_no_improve += 1
            if self.monitor_hit_patience > 0 and self.hit_no_improve >= self.monitor_hit_patience:
                if self.args.local_rank in [0, -1]:
                    print(f"Hit-based early stopping triggered at epoch {epoch} (no improvement for {self.monitor_hit_patience} epochs).")
                return True
        return False

    # --- helpers ------------------------------------------------------------
    def _write_best_sidecars(self, epoch: int, avg_valid_loss: float) -> None:
        try:
            out_dir = self.args.output
            os.makedirs(out_dir, exist_ok=True)
            meta = {
                "epoch": int(epoch),
                "avg_valid_loss": float(avg_valid_loss),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "run_name": getattr(self.args, "run_name", ""),
                "output_dir": out_dir,
                "checkpoint": "BEST_EVAL_LOSS.pth",
                # Extra fields to make mr=0 equivalence auditable
                "attack_mode": getattr(self.args, "attack_mode", None),
                "mr": float(getattr(self.args, "mr", 0.0)),
                # When mr == 0, training uses the clean dataset regardless of attack_mode
                "clean_equivalent": bool(float(getattr(self.args, "mr", 0.0)) == 0.0),
            }
            with open(os.path.join(out_dir, "BEST_EVAL_LOSS.meta.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
            with open(os.path.join(out_dir, "BEST_EVAL_LOSS.history.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
            tsv_path = os.path.join(out_dir, "BEST_HISTORY.tsv")
            header = "timestamp\tepoch\tavg_valid_loss\trun_name\n"
            line = f"{meta['timestamp']}\t{meta['epoch']}\t{meta['avg_valid_loss']:.6f}\t{meta['run_name']}\n"
            if not os.path.exists(tsv_path):
                with open(tsv_path, "w", encoding="utf-8") as tf:
                    tf.write(header)
                    tf.write(line)
            else:
                with open(tsv_path, "a", encoding="utf-8") as tf:
                    tf.write(line)
        except Exception:
            pass

    def _finalize_best_from_history(self) -> None:
        out_dir = self.args.output
        hist_tsv = os.path.join(out_dir, "BEST_HISTORY.tsv")
        if not os.path.exists(hist_tsv):
            return
        best_epoch = None
        best_loss = None
        with open(hist_tsv, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i == 0:
                    continue  # skip header
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                ts, epoch_str, loss_str, _ = parts[:4]
                try:
                    ep = int(epoch_str)
                    ls = float(loss_str)
                except Exception:
                    continue
                if (best_loss is None) or (ls < best_loss):
                    best_loss = ls
                    best_epoch = ep
        if best_epoch is None:
            return
        # If on-disk meta doesn't match the true best, correct it and copy checkpoint
        meta_path = os.path.join(out_dir, "BEST_EVAL_LOSS.meta.json")
        current_epoch = None
        try:
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as fh:
                    cur = json.load(fh)
                current_epoch = int(cur.get("epoch", -1))
        except Exception:
            current_epoch = None
        if current_epoch != best_epoch:
            src = os.path.join(out_dir, f"Epoch%02d.pth" % (best_epoch))
            dst = os.path.join(out_dir, "BEST_EVAL_LOSS.pth")
            if os.path.exists(src):
                import shutil
                shutil.copyfile(src, dst)
                # write corrected meta
                self._write_best_sidecars(best_epoch, best_loss if best_loss is not None else 0.0)

def main_worker(gpu, args):
    """主进程工作函数，用于分布式训练"""
    print("Distributed-related environment variables:")
    for key in ["CUDA_VISIBLE_DEVICES", "WORLD_SIZE", "LOCAL_RANK", "RANK", "MASTER_ADDR", "MASTER_PORT"]:
        print(f"{key}: {os.environ.get(key, 'Not Set')}")
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.gpu = args.local_rank
    print(f"Process Launching at GPU {args.gpu}, local_rank: {args.local_rank}")
    torch.cuda.set_device(args.gpu)
    args.distributed = args.distributed and args.world_size > 1
    if not args.distributed:
        print("Running in non-distributed mode")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Distributed training requires GPU support.")
    if args.local_rank < 0 or args.gpu < 0:
        raise RuntimeError(f"Invalid GPU index {args.gpu}. Check CUDA_VISIBLE_DEVICES and GPU allocation.")
    try:
        dist.init_process_group(backend='nccl', init_method='env://')
    except Exception as e:
        raise RuntimeError(f"Failed to initialize distributed process group: {e}")
    print(f"Launching distributed training: {args.distributed}, Total GPUs: {args.world_size}")
    print(f"Train dataset split: {args.train}, Valid dataset split: {args.valid}")
    print(f'Building train loader at GPU {gpu}')
    train_task_list = {
        'sequential': ['A-1', 'A-2', 'A-3', 'A-4', 'A-5', 'A-6', 'A-7', 'A-8'],
        'direct': ['B-1', 'B-2', 'B-3', 'B-4', 'B-5', 'B-6', 'B-7', 'B-9'],
        'explanation': ['C-1', 'C-2', 'C-3', 'C-4', 'C-5', 'C-6', 'C-7', 'C-8', 'C-9', 'C-10', 'C-11'],
    }
    # Allow per-task sampling multipliers to bias training towards preserving ranking quality
    try:
        r_seq = max(0, int(getattr(args, 'train_seq_repeat', 1)))
        r_dir = max(0, int(getattr(args, 'train_dir_repeat', 1)))
        r_exp = max(0, int(getattr(args, 'train_expl_repeat', 1)))
    except Exception:
        r_seq = r_dir = r_exp = 1

    train_sample_numbers = {}

    if r_seq > 0:
        train_sample_numbers['sequential'] = (max(1, 2 * r_seq), max(1, 2 * r_seq))
    else:
        train_task_list.pop('sequential', None)

    if r_dir > 0:
        direct_template_base = ['B-1', 'B-2', 'B-3', 'B-4', 'B-5', 'B-6', 'B-7']
        train_task_list['direct'] = direct_template_base
        train_sample_numbers['direct'] = (max(1, 2 * r_dir), max(1, 2 * r_dir))
    else:
        train_task_list.pop('direct', None)

    if r_exp > 0:
        train_sample_numbers['explanation'] = max(1, r_exp)
    else:
        train_task_list.pop('explanation', None)

    if not train_task_list:
        raise RuntimeError("No training tasks selected; ensure at least one of train_seq_repeat/train_dir_repeat/train_expl_repeat is > 0")
    try:
        train_loader = get_loader(
            args,
            train_task_list,
            train_sample_numbers,
            split=args.train,
            mode='train',
            batch_size=args.batch_size,
            workers=args.num_workers,
            distributed=args.distributed
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize train loader for split '{args.train}'. Task list: {train_task_list}, Args: {args}, Error: {e}")
    print(f'Building val loader at GPU {gpu}')
    val_task_list = {
        'sequential': ['A-1', 'A-2', 'A-3', 'A-4', 'A-5', 'A-6', 'A-7', 'A-8'],
        'direct': ['B-1', 'B-2', 'B-3', 'B-4', 'B-5', 'B-6', 'B-7', 'B-9'],
        'explanation': ['C-1', 'C-2', 'C-3', 'C-4', 'C-5', 'C-6', 'C-7', 'C-8', 'C-9', 'C-10', 'C-11'],
    }
    val_sample_numbers = {'sequential': (1, 1), 'direct': (1, 1), 'explanation': 1}
    try:
        val_loader = get_loader(
            args,
            val_task_list,
            val_sample_numbers,
            split=args.valid,
            mode='val',
            batch_size=args.batch_size,
            workers=args.num_workers,
            distributed=args.distributed
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize val loader for split '{args.valid}'. Task list: {val_task_list}, Args: {args}, Error: {e}")
    try:
        trainer = Trainer(args, train_loader, val_loader, train=True)
        trainer.train()
    finally:
        if args.distributed and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    """主函数，初始化参数和分布式环境"""
    print("Environment variables at main:", os.environ)
    cudnn.benchmark = False
    cudnn.deterministic = True

    args = parse_args()
    # Print out the effective arguments immediately after parsing
    # so that any overrides from YAML configs are visible in logs.
    print("Effective arguments after parse_args:", args)
    args.distributed = True
    

    ngpus_per_node = torch.cuda.device_count()
    print(f"Number of GPUs available: {ngpus_per_node}")
    args.world_size = ngpus_per_node

    LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
    if args.local_rank in [0, -1]:
        print(LOSSES_NAME)
    LOSSES_NAME.append('total_loss')
    args.LOSSES_NAME = LOSSES_NAME

    # --- 这里是修改过的 run_name 和 output 路径逻辑 ---
    from datetime import datetime
    # Use date+time to avoid overwriting runs with identical hyper-params on the same day
    current_time = datetime.now().strftime('%m%d_%H%M%S')

    if args.local_rank in [0, -1]:
        # 按照：suffix_mr_split-img_feat_type-img_feat_size_ratio-reduction_factor-epoch
        run_name = (
            f"{args.attack_mode}_{args.mr}"
            f"_{args.train}-{args.image_feature_type}"
            f"-{args.image_feature_size_ratio}"
            f"-{args.reduction_factor}-{args.epoch}"
        )
        args.run_name = run_name
        print("运行名称:", args.run_name)

    # 输出路径：若命令行已提供 --output 则尊重之；否则：snap/<split>/<MMDD_HHMMSS>/<run_name>
    if not getattr(args, 'output', None):
        args.output = os.path.join('snap', args.train, current_time, args.run_name)
    os.makedirs(args.output, exist_ok=True)
    # --- 修改结束 ---

    if args.distributed:
        main_worker(args.local_rank, args)
