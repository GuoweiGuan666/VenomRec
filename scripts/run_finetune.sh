#!/usr/bin/env bash
set -euo pipefail

################################################################################
# run_finetune.sh
#
# 一行搞定 Fine‑tune：修改 config.yaml、设置 GPU、创建日志、nohup 训练
#
# 使用方法：
#   bash scripts/run_finetune.sh \
#     <split>                \ # toys|clothing|beauty|sports
#     <attack_mode>          \ # e.g. DirectBoostingAttack   NoAttack  RandomInjectionAttack   PopularItemMimickingAttack 
#     <mr>                   \ # e.g. 0.1
#     <gpu_ids>              \ # 哪几张卡，例如 "0" 或 "0,1,2"
#     <img_feat_type>        \ # vitb32|vitb16|rn50|…
#     <img_feat_size_ratio>  \ # 比如 2
#     <reduction_factor>     \ # adapter reduction，如 8
#     <epoch>                \ # 训练 epoch 数，如 20
#     [-- <额外 train_VIP5.sh 参数>…]
#
# 例子：
#
#   1 GPU（第 3 号卡）上 Fine‑tune toys 上的 DirectBoostingAttack：
#   bash scripts/run_finetune.sh toys DirectBoostingAttack 0.1 3 vitb32 2 8 20
#
#   4 GPUs（卡 0,1,2,3）上 Fine‑tune sports 上的 PopularItemMimickingAttack：
#   bash scripts/run_finetune.sh sports PopularItemMimickingAttack 0.1 0,1,2,3 vitb32 2 8 20
#
#   2 GPUs (卡 2,3）上 Fine‑tune toys 上的 NoAttack：
#   bash scripts/run_finetune.sh toys NoAttack 0 2,3 vitb32 2 8 20
#
#   1 GPU（第 3 号卡）上 Fine‑tune toys 上的 RandomInjectionAttack：
#   bash scripts/run_finetune.sh toys RandomInjectionAttack 0.1 3 vitb32 2 8 20
#
#   2 GPUs（卡 0,1）上 Fine‑tune beauty 上的 ShadowCastAttack：
#   bash scripts/run_finetune.sh beauty ShadowCastAttack 0.1 0,1 vitb32 2 8 20
#
#   若已将投毒文件整理到 data/<split>/poisoned/<subdir>，在命令末尾附加
#   “-- --poison_subdir <subdir>” 即可复用 NoAttack 同款 Fine-tune 流程。
#
#
# 查看日志：
#   tail -f log/toys/$(date +%m%d)/fine_tuning_logs/DirectBoostingAttack_0.1_3_toys-vitb32-2-8-20.out

################################################################################


###################################################################################
# -----------------------------------------------------------------------------
# 注意！一定保证 batch_size 和 使用的gpu的个数 的乘积是128
# 1个gpu，batch_size=128
# 2个gpu，batch_size=64
# 3个gpu，不能运行
# 4个gpu，batch_size=32
######################################################################################


usage() {
  echo "Usage: $0 <split> <attack_mode> <mr> <gpu_list> <img_feat_type> <img_feat_size_ratio> <reduction> <epoch> [-- extra args]"
  exit 1
}
[ $# -lt 8 ] && usage

# 1. 参数解析
split=$1
attack=$2
mr=$3
gpu_list=$4            # e.g. "0" or "0,1" or "0,1,2,3"
img_feat_type=$5
img_feat_ratio=$6
reduction=$7
epoch=$8
shift 8                # 剩下的都是传给 train_VIP5.sh 的额外参数
extra_args=("$@")

# 允许使用常见的 `-- extra args...` 形式，但不要把分隔符本身传下去。
if [ ${#extra_args[@]} -gt 0 ] && [ "${extra_args[0]}" = "--" ]; then
  extra_args=("${extra_args[@]:1}")
fi

# 先计算 GPU 数量，供自动 batch 使用
IFS=, read -ra _GPU_ARR <<< "${gpu_list}"
ngpus=${#_GPU_ARR[@]}

# 如果没有显式设置 --batch_size，则根据 GPU 数自动补全
has_batch_size=false
for arg in "${extra_args[@]}"; do
  case "$arg" in
    --batch_size|--batch_size=*|--batch-size|--batch-size=*)
      has_batch_size=true
      break
      ;;
  esac
done

if [ "$has_batch_size" = false ]; then
  case "$ngpus" in
    1) auto_bs=128 ;;
    2) auto_bs=64  ;;
    4) auto_bs=32  ;;
    8) auto_bs=16  ;;
    *)
      echo "Error: Unsupported GPU count: $ngpus. Must be 1,2,4 or 8." >&2
      exit 1
      ;;
  esac
  echo "[INFO] 未显式指定 --batch_size，自动设置 batch_size=${auto_bs} (总 128)"
  extra_args+=("--batch_size" "${auto_bs}")
fi

# 2. （可选）修改 config.yaml，用 sed
echo "🔧 更新 config.yaml: dataset=$split, suffix=$attack, mr=$mr"
sed -i -E "s|(base_folder:).*|\1 \"data/${split}\"|" config.yaml
sed -i -E "s|(suffix:).*|\1 \"${attack}\"|" config.yaml
sed -i -E "s|(mr:).*|\1 ${mr}|" config.yaml

# 3. 设置 GPU
echo "🖥  Using GPUs: ${gpu_list}"
export CUDA_VISIBLE_DEVICES="${gpu_list}"
# 计算卡数，train_VIP5.sh 参数需要第一个参数是 GPU 数量
IFS=, read -ra _ARR <<< "${gpu_list}"
ngpus=${#_ARR[@]}

# 防止验证阶段等待过久触发 NCCL watchdog 超时
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}

# 4. 统一的时间戳（精确到秒），并传给下游脚本，确保 output 与 log 使用同一时间戳
date_str=$(date +%m%d_%H%M%S)
export RUN_TS="${date_str}"
LOG_DIR="log/${split}/${date_str}/fine_tuning_logs"
mkdir -p "${LOG_DIR}"
echo "📂 Logs in: ${LOG_DIR}"

# 5. 构造实验名 & out 文件
EXPERIMENT_TAG="${attack}_${mr}"
OUT_NAME="${EXPERIMENT_TAG}_${split}-${img_feat_type}-${img_feat_ratio}-${reduction}-${epoch}.out"

# 6. 启动训练
echo "🚀 Launching training on ${ngpus} GPU(s)..."
nohup bash scripts/train_VIP5.sh \
  "${ngpus}" "${split}" 29689 "${img_feat_type}" "${img_feat_ratio}" "${reduction}" "${epoch}" \
  "${extra_args[@]}" \
  > "${LOG_DIR}/${OUT_NAME}" 2>&1 &
  

echo "✅ Launched! Check ${LOG_DIR}/${OUT_NAME}"
