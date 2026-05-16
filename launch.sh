#!/bin/bash
#
# Usage: ./launch.sh <mode> <model_size> [steps] [nodes]
#
# Modes:     throughput  (50 steps, with W&B)
#            train       (N steps, with W&B and Tensorboard)
#
# Sizes:     125m, 350m, 760m, 1.5b, 3b, 8b
#
# Steps:     required for train mode (e.g., 1000, 5000, 15000)
# Nodes:     optional, default 4 (max 8)
#
# Examples:  ./launch.sh throughput 760m
#            ./launch.sh throughput 8b 50 1
#            ./launch.sh train 760m 5000
#            ./launch.sh train 1.5b 3000 8

set -euo pipefail

source "$(dirname "$0")/config.sh"

MODE=${1:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes]}
MODEL_SIZE=${2:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes]}

################ Mode config ################
case $MODE in
    throughput)
        TRAINING_STEPS=${3:-50}
        NODES=${4:-4}
        TIME=00:30:00
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        LOGGING_EXTRA=""
        WANDB=true

        CKPT_ENABLED=false
        SAVE_INTERVAL=0
        MAX_RESTARTS=0
        ;;
    train)
        TRAINING_STEPS=${3:?Usage: ./launch.sh train <model_size> <steps> [nodes]}
        NODES=${4:-4}
        TIME=02:30:00
        EVAL_INTERVAL=1000
        EVAL_ITERS=10
        LR_WARMUP_ITERS=200
        LOGGING_EXTRA="
    --tensorboard-dir \$TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard"
        WANDB=true

        SAVE_INTERVAL=${SAVE_INTERVAL:-500}
        MAX_RESTARTS=${MAX_RESTARTS:-3}
        CKPT_ENABLED=true
        ;;
esac

################ Model config ################
case $MODEL_SIZE in
    125m)
        NUM_LAYERS=12;  HIDDEN=768;  FFN=2048;  HEADS=12; KV_HEADS=4
        MBS=16
        ;;
    350m)
        NUM_LAYERS=24; HIDDEN=1024; FFN=2816;  HEADS=16; KV_HEADS=4
        MBS=8
        ;;
    760m)
        NUM_LAYERS=24; HIDDEN=1536; FFN=4096;  HEADS=16; KV_HEADS=4
        MBS=4
        ;;
    1.5b)
        NUM_LAYERS=48; HIDDEN=1600; FFN=4352;  HEADS=20; KV_HEADS=4
        MBS=4
        ;;
    3b)
        NUM_LAYERS=32; HIDDEN=3072; FFN=8192;  HEADS=24; KV_HEADS=8
        MBS=4
        ;;
    8b)
        NUM_LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; KV_HEADS=8
        MBS=2
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Choose: 125m, 350m, 760m, 1.5b, 3b, 8b"
        exit 1
        ;;
esac

GBS=256
SEQ_LEN=4096
JOB_NAME="gipfel-${MODE}-${MODEL_SIZE}-${TRAINING_STEPS}s-${NODES}n"

################ W&B block ################
if [ "$WANDB" = true ]; then
    WANDB_BLOCK='
# WANDB
WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "[$(date)] WANDB enabled."
    WANDB_ARGS+=(
        --wandb-save-dir "$LOG_DIR"
        --wandb-project "$PROJECT_NAME"
        --wandb-exp-name "$EXP_NAME-$SLURM_JOB_ID"
    )
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi'
else
    WANDB_BLOCK='
WANDB_ARGS=()
export WANDB_MODE=disabled'
fi

################ Generate script ################
mkdir -p logs

SCRIPT="logs/${JOB_NAME}.sbatch"


cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
HEADER

cat >> "$SCRIPT" << SBATCH_DIRECTIVES
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --requeue
SBATCH_DIRECTIVES

cat >> "$SCRIPT" << 'BODY_HEAD'

echo "START TIME: \$(date)"

################ Configs ################
BODY_HEAD

cat >> "$SCRIPT" << BODY_WORKDIR
WORKDIR=${WORKDIR}
MEGATRON_LM_DIR=\$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/cache
BODY_WORKDIR

cat >> "$SCRIPT" << CONFIGS

# Training config
MBS=${MBS}
GBS=${GBS}
SEQ_LEN=${SEQ_LEN}
TRAINING_STEPS=${TRAINING_STEPS}

# Logging
PROJECT_NAME=gipfelsturm
EXP_NAME=${MODE}-${MODEL_SIZE}-\${SLURM_NNODES}n
LOG_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/\$PROJECT_NAME/\$EXP_NAME
TENSORBOARD_DIR=\$LOG_DIR/tensorboard

CKPT_ENABLED=${CKPT_ENABLED}
SAVE_INTERVAL=${SAVE_INTERVAL}
MAX_RESTARTS=${MAX_RESTARTS}

CHECKPOINT_ROOT=/iopsstor/scratch/cscs/\$USER/gipfelsturm/checkpoints
CHECKPOINT_DIR=\$CHECKPOINT_ROOT/\$EXP_NAME

CONFIGS

cat >> "$SCRIPT" << 'SETUP'

#########################################

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR $CHECKPOINT_DIR

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)

SETUP

cat >> "$SCRIPT" << MODEL
NETWORK_SIZE_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN}
    --ffn-hidden-size ${FFN}
    --num-attention-heads ${HEADS}
    --group-query-attention
    --num-query-groups ${KV_HEADS}
    --max-position-embeddings \$SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length \$SEQ_LEN
)
MODEL

cat >> "$SCRIPT" << TRAINING

TRAINING_ARGS=(
    --micro-batch-size \$MBS
    --global-batch-size \$GBS
    --train-iters \$TRAINING_STEPS
    --log-interval 1
    --eval-interval ${EVAL_INTERVAL}
    --eval-iters ${EVAL_ITERS}
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer adam
    --dataloader-type single
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 50
)


REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr 3e-4
    --lr-decay-style constant
    --lr-warmup-iters ${LR_WARMUP_ITERS}
)
TRAINING

cat >> "$SCRIPT" << 'REST'

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
REST

cat >> "$SCRIPT" << LOGGING_EXTRA
${LOGGING_EXTRA}
)
LOGGING_EXTRA

cat >> "$SCRIPT" << 'TOKENIZER'

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/data/gpt2-vocab.json
    --merge-file $WORKDIR/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $DATA_PREFIX
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 1
)

checkpoint_dir_for_iter() {
    local iter="$1"
    printf "%s/iter_%07d" "$CHECKPOINT_DIR" "$iter"
}

checkpoint_dir_looks_complete() {
    local dir="$1"

    [[ -d "$dir" ]] || return 1


    find "$dir" -type f \( \
        -name "metadata.json" -o \
        -name "common.pt" -o \
        -name "*.distcp" -o \
        -name "model_optim_rng.pt" -o \
        -name "model_rng.pt" \
    \) | grep -q .
}

repair_latest_checkpoint_tracker() {
    [[ "$CKPT_ENABLED" == "true" ]] || return 1
    [[ -d "$CHECKPOINT_DIR" ]] || return 1

    local tracker="$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"

    local dirs
    mapfile -t dirs < <(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "iter_*" | sort -V -r)

    for dir in "${dirs[@]}"; do
        if checkpoint_dir_looks_complete "$dir"; then
            local base
            base=$(basename "$dir")
            local iter="${base#iter_}"

            # Convert 0000500 -> 500 safely.
            iter=$((10#$iter))

            echo "$iter" > "$tracker.tmp"
            mv "$tracker.tmp" "$tracker"
            echo "[$(date)] Latest usable checkpoint: iteration $iter"
            return 0
        fi
    done

    echo "[$(date)] No usable checkpoint found in $CHECKPOINT_DIR"
    return 1
}

build_checkpoint_args() {
    CHECKPOINT_ARGS=()

    if [[ "$CKPT_ENABLED" != "true" ]]; then
        return 0
    fi

    CHECKPOINT_ARGS+=(
        --save "$CHECKPOINT_DIR"
        --save-interval "$SAVE_INTERVAL"
        --ckpt-format torch_dist

        # Important mitigation for the segfault issue:
        # avoid the fully parallel save path initially.
        --no-ckpt-fully-parallel-save
    )


    if repair_latest_checkpoint_tracker; then
        CHECKPOINT_ARGS+=(
            --load "$CHECKPOINT_DIR"
            --exit-on-missing-checkpoint
        )
    else
        echo "[$(date)] Starting without --load because no valid checkpoint exists."
    fi
}



TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

build_training_cmd() {
    build_checkpoint_args

    TRAINING_CMD_ARRAY=(
        torchrun
        "${TORCHRUN_ARGS[@]}"
        "$MEGATRON_LM_DIR/pretrain_gpt.py"

        "${TRANSFORMER_ENGINE_ARGS[@]}"
        "${NETWORK_SIZE_ARGS[@]}"
        "${TRAINING_ARGS[@]}"
        "${REGULARIZATION_ARGS[@]}"
        "${LEARNING_RATE_ARGS[@]}"
        "${INITIALIZATION_ARGS[@]}"
        "${MIXED_PRECISION_ARGS[@]}"
        "${DISTRIBUTED_ARGS[@]}"
        "${LOGGING_ARGS[@]}"
        "${TOKENIZER_ARGS[@]}"
        "${DATA_ARGS[@]}"
        "${CHECKPOINT_ARGS[@]}"
        "${WANDB_ARGS[@]}"
    )

    printf -v TRAINING_CMD "%q " "${TRAINING_CMD_ARRAY[@]}"
}
TOKENIZER

cat >> "$SCRIPT" << 'WANDB_PLACEHOLDER'
WANDB_PLACEHOLDER

# Replace placeholder with actual W&B block
sed -i '/^WANDB_PLACEHOLDER$/d' "$SCRIPT"
cat >> "$SCRIPT" << WANDB_INSERT
${WANDB_BLOCK}
WANDB_INSERT

cat >> "$SCRIPT" << 'FOOTER'

ATTEMPT=0

while true; do
    build_training_cmd

    echo "[$(date)] CMD: $TRAINING_CMD"

    set +e
    srun -lu \
        --mpi=pmix \
        --network=disable_rdzv_get \
        --environment=alps3 \
        --cpus-per-task "$SLURM_CPUS_PER_TASK" \
        --wait 60 \
        bash -lc "numactl --membind=0-3 $TRAINING_CMD"

    EXIT_CODE=$?
    set -e

    if [[ "$EXIT_CODE" -eq 0 ]]; then
        echo "[$(date)] Training exited cleanly."
        break
    fi

    ATTEMPT=$((ATTEMPT + 1))

    echo "[$(date)] Training failed with exit code $EXIT_CODE."
    echo "[$(date)] Restart attempt $ATTEMPT / $MAX_RESTARTS."

    if [[ "$ATTEMPT" -gt "$MAX_RESTARTS" ]]; then
        echo "[$(date)] Too many failures. Exiting."
        exit "$EXIT_CODE"
    fi

    # Re-check checkpoint state before retrying.
    if [[ "$CKPT_ENABLED" == "true" ]]; then
        repair_latest_checkpoint_tracker || true
    fi

    sleep 30
done

echo "END TIME: $(date)"
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT"

if [[ "${NO_SUBMIT:-0}" == "1" ]]; then
    echo "NO_SUBMIT=1 set, not submitting."
    bash -n "$SCRIPT"
    echo "Syntax check passed: $SCRIPT"
    exit 0
fi

sbatch "$SCRIPT"
