#!/usr/bin/env bash
# =============================================================================
# scripts/train_VIP5.sh
#
# 启动 fine-tune 脚本，自动从 config.yaml 中读取：
#   - experiment.suffix （如 NoAttack、DirectBoostingAttack 等）
#   - experiment.mr     （恶意用户比例，如 0.1）
#
# 两者将拼入输出目录和日志名称中，并且日志实时刷新，不缓冲。
# =============================================================================

###################################################################################
# -----------------------------------------------------------------------------
# 注意！默认保持“总 batch = 128”的原则：
# 1个GPU，batch_size=128
# 2个GPU，batch_size=64
# 4个GPU，batch_size=32
# 8个GPU，batch_size=16
# 3/6个GPU：不支持（128 无法整除）
######################################################################################

# -----------------------------------------------------------------------------
# 1) 计算本次要用的 GPU 数量
# -----------------------------------------------------------------------------
if [ -z "$1" ]; then
  nproc_per_node=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
else
  nproc_per_node=$1
  shift
fi

# -----------------------------------------------------------------------------
# 1.1) 根据 GPU 数量自动设置 batch_size
# -----------------------------------------------------------------------------
case "$nproc_per_node" in
  1) batch_size=128 ;;
  2) batch_size=64  ;;
  4) batch_size=32  ;;
  8) batch_size=16  ;;
  *)
    echo "Error: Unsupported GPU count: $nproc_per_node. Must be 1,2,4 or 8." >&2
    exit 1
    ;;
esac

echo "Using ${nproc_per_node} GPU(s): setting batch_size=${batch_size}"

# -----------------------------------------------------------------------------
# 2) 解析位置参数
#    split               : 数据集划分（如 toys、sports…）
#    port                : torchrun 通信端口
#    img_feat_type       : 视觉特征类型（如 vitb32）
#    img_feat_size_ratio : 特征块数（如 2）
#    reduction_factor    : adapter 缩减因子（如 8）
#    epoch               : 训练轮次（如 20）
# -----------------------------------------------------------------------------
split=$1; port=$2; img_feat_type=$3; img_feat_size_ratio=$4; reduction_factor=$5; epoch=$6
shift 6

# -----------------------------------------------------------------------------
# 3) 从 config.yaml 中读取 suffix 和 mr
# -----------------------------------------------------------------------------
read_config() {
  python3 - <<'EOF'
import yaml
cfg = yaml.safe_load(open('config.yaml'))
suffix = cfg.get('experiment', {}).get('suffix', 'NoAttack')
mr     = cfg.get('experiment', {}).get('mr', 0)
mr = float(mr)
mr_str = str(float(mr))
print(suffix, mr_str)
EOF
}

read_result=($(read_config))
suffix=${read_result[0]}
mr=${read_result[1]}

# -----------------------------------------------------------------------------
# 4) 生成本次实验的标识和路径
# -----------------------------------------------------------------------------
# 允许上游提供 RUN_TS 以确保与外层日志目录/输出目录时间戳一致
date_str=${RUN_TS:-$(date +%m%d_%H%M%S)}
name="${suffix}_${mr}_${split}-${img_feat_type}-${img_feat_size_ratio}-${reduction_factor}-${epoch}"
output="snap/${split}/${date_str}/${name}"
log="log/${split}/${date_str}/fine_tuning_logs/${name}.log"
mkdir -p "$(dirname "$output")" "$(dirname "$log")"

echo "Launching training: split=${split}, suffix=${suffix}, mr=${mr}, GPUs=${CUDA_VISIBLE_DEVICES}"
echo "  output dir: ${output}"
echo "  log file : ${log}"

# -----------------------------------------------------------------------------
# 5) 启用不缓冲与行缓冲，让 tqdm 刷新可见
# -----------------------------------------------------------------------------
export PYTHONUNBUFFERED=1

# -----------------------------------------------------------------------------
# 5.1) 选择 torchrun 可执行；若不可用则回退到 `python -m torch.distributed.run`
# -----------------------------------------------------------------------------
PY_BIN="${PYTHON:-python3}"
if command -v torchrun >/dev/null 2>&1; then
  TORCHRUN_CMD="torchrun"
else
  echo "[WARN] 'torchrun' not found in PATH; falling back to '${PY_BIN} -m torch.distributed.run'" >&2
  TORCHRUN_CMD="${PY_BIN} -m torch.distributed.run"
fi

# -----------------------------------------------------------------------------
# 6.1) 默认使用 beauty clean 预训练权重，除非显式传入 --load
# -----------------------------------------------------------------------------
DEFAULT_BEAUTY_CKPT="snap/beauty/0805/NoAttack_0.0_beauty-vitb32-2-8-20/BEST_EVAL_LOSS"
extra_args=("$@")
need_default_load=true
for arg in "${extra_args[@]}"; do
  case "$arg" in
    --load|--load=*)
      need_default_load=false
      break
      ;;
  esac
done
if [ "$need_default_load" = true ]; then
  echo "[INFO] 未检测到 --load，默认加载 beauty clean checkpoint: ${DEFAULT_BEAUTY_CKPT}"
  extra_args+=("--load" "${DEFAULT_BEAUTY_CKPT}")
fi

# -----------------------------------------------------------------------------
# 6) 启动分布式训练
# -----------------------------------------------------------------------------
stdbuf -oL -eL ${TORCHRUN_CMD} \
  --nproc_per_node="$nproc_per_node" \
  --master_port "$port" \
  src/train.py \
    --config config.yaml \
    --distributed --multiGPU \
    --seed 2022 \
    --train "$split" --valid "$split" \
    --batch_size "$batch_size" \
    --optim adamw \
    --warmup_ratio 0.1 \
    --lr 1e-3 \
    --num_workers 4 \
    --clip_grad_norm 5.0 \
    --losses 'sequential,direct,explanation' \
    --backbone 't5-small' \
    --output "$output" \
    --epoch "$epoch" \
    --use_adapter \
    --unfreeze_layer_norms \
    --reduction_factor "$reduction_factor" \
    --use_single_adapter \
    --max_text_length 512 \
    --gen_max_length 64 \
    --image_feature_type "$img_feat_type" \
    --image_feature_size_ratio "$img_feat_size_ratio" \
    --whole_word_embed \
    --category_embed \
    "${extra_args[@]}" \
> "$log" 2>&1
