#!/bin/bash
#SBATCH --job-name=nanogpt-baseline
#SBATCH --partition=normal
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --array=0-4 
#SBATCH --output=/home/%u/proj/tinyllm/logs/%x-%A_%a.out
#SBATCH --error=/home/%u/proj/tinyllm/logs/%x-%A_%a.err


set -euo pipefail


#==============================#
#  BLOCK 0: HELPERS / FUNCTIONS#
#==============================#
log_section() {
  echo
  echo "=============================="
  echo "$1"
  echo "=============================="
}

die() { echo "ERROR: $*" >&2; exit 1; }

ensure_dir() {
  local dir="$1"
  [[ -d "$dir" ]] || mkdir -p "$dir"
}

ensure_file() {
  local cpath="$1"
  [[ -f "$cpath" ]] || die "Error: can not find file $cpath"
}

#==============================#
#  BLOCK 1: USER CONFIG        #
#==============================#
PROJECT_DIR="${PROJECT_DIR:-$HOME/}"
ensure_dir "$PROJECT_DIR"

REPO_DIR="${REPO_DIR:-$PROJECT_DIR/repos/nanoGPT}"
[[ -d "$REPO_DIR" ]] || die "REPO_DIR does not exist: $REPO_DIR"
[[ -f "$REPO_DIR/train.py" ]] || die "Not a nanoGPT repo: $REPO_DIR"

MODULE_SET="${MODULE_SET:-"nanogpt-3"}"
ENVNAME="${ENVNAME:-/home/$USER/envs/nanogpt-3/bin/activate}"
ensure_file "$ENVNAME"

RUNS_ROOT="${RUNS_ROOT:-$PROJECT_DIR/nanogpt-runs}"
ensure_dir "$RUNS_ROOT"

EXP_NAME="${EXP_NAME:-baseline}"

EXP_DESC="${EXP_DESC:-"baseline"}"

DATASET="${DATASET:-shakespeare_char}"
ensure_dir "$REPO_DIR/data/$DATASET"

N_SEEDS="${N_SEEDS:-5}"
BASE_SEED="${BASE_SEED:-1337}"

CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/config/train_shakespeare_char.py}"
ensure_file "$CONFIG_PATH"

# Постобработка: вызов summarize.py
DO_SUMMARIZE="${DO_SUMMARIZE:-0}"


#==============================#
#  BLOCK 2: DECODE ARRAY TASK  #
#==============================#
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"

SEED_IDX=$(( TASK_ID ))

(( 0 <= SEED_IDX && SEED_IDX < N_SEEDS )) || die "TASK_ID=$TASK_ID out of range for N_SEEDS"
SEED=$(( BASE_SEED + SEED_IDX ))


# RUN_DIR name: JOBID_TASKID_exp_timestamp
TS="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_NAME="${JOB_ID}_${TASK_ID}_${EXP_NAME}_${TS}"
RUN_DIR="${RUNS_ROOT}/${RUN_NAME}"
ensure_dir "$RUN_DIR"

mkdir -p "$RUN_DIR"/{meta,logs,out}

# Дублируем весь stdout/stderr в run-логи (и в slurm-логи тоже останется)
exec > >(tee -a "$RUN_DIR/logs/stdout.log") 2> >(tee -a "$RUN_DIR/logs/stderr.log" >&2)

log_section "BLOCK 2: RUN IDENTIFIERS"
echo "JOB_ID      : $JOB_ID"
echo "TASK_ID     : $TASK_ID"
echo "SEED        : $SEED"
echo "EXP_NAME    : $EXP_NAME"
echo "EXP_DESC    : $EXP_DESC"
echo "DATASET     : $DATASET"
echo "CONFIG      : $CONFIG_PATH"
echo "RUN_DIR     : $RUN_DIR"
echo "REPO_DIR    : $REPO_DIR"
echo "HOST        : $(hostname)"
echo "TS          : $TS"


#==============================#
#  BLOCK 3: ENVIRONMENT SETUP  #
#==============================#
cd "$REPO_DIR"

module purge
module restore $MODULE_SET
source $ENVNAME # python environment

log_section "BLOCK 3: ENVIRONMENT SETUP"
echo "MODULES     : $MODULE_SET"
echo "ENVNAME     : $ENVNAME"

# module describe $MODULE_SET > "$RUN_DIR/meta/modules_set.txt"
# pip freeze > "$RUN_DIR/meta/pip_freeze.txt"

#==============================#
#  BLOCK 4: METADATA SNAPSHOT  #
#==============================#
log_section "BLOCK 4: METADATA SNAPSHOT"

# GPU info
nvidia-smi | tee "$RUN_DIR/logs/nvidia-smi.txt" || true

# git snapshot
GIT_COMMIT=""
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
  echo "$GIT_COMMIT" > "$RUN_DIR/meta/git_commit.txt" || true
  git diff > "$RUN_DIR/meta/git_diff.patch" 2>/dev/null || true
fi

# config snapshot
cp -f "$CONFIG_PATH" "$RUN_DIR/meta/config_snapshot.py"

# meta JSON
RUN_DIR="$RUN_DIR" RUN_NAME="$RUN_NAME" JOB_ID="$JOB_ID" TASK_ID="$TASK_ID" \
SEED="$SEED" SEED_IDX="$SEED_IDX" BASE_SEED="$BASE_SEED" GIT_COMMIT="$GIT_COMMIT" \
EXP_NAME="$EXP_NAME" EXP_DESC="$EXP_DESC" DATASET="$DATASET" CONFIG_PATH="$CONFIG_PATH" \
REPO_DIR="$REPO_DIR" MODULE_SET="$MODULE_SET" ENVNAME="$ENVNAME" TS="$TS" \
python - <<'PY'
import json, os, socket, subprocess
from pathlib import Path

def getenv(k, default=None):
    v = os.environ.get(k)
    return v if v not in (None, "") else default

run_dir = Path(getenv("RUN_DIR")).resolve()
(run_dir / "meta").mkdir(parents=True, exist_ok=True)

meta = {
    "run_name": getenv("RUN_NAME"),
    "exp_name": getenv("EXP_NAME"),
    "exp_desc": getenv("EXP_DESC"),
    "dataset": getenv("DATASET"),
    "config": getenv("CONFIG_PATH"),
    "seed": int(getenv("SEED", "0")),
    "seed_idx": int(getenv("SEED_IDX", "-1")),
    "base_seed": int(getenv("BASE_SEED", "0")),
    "timestamp": getenv("TS"),
    "run_dir": str(run_dir),
    "repo_dir": getenv("REPO_DIR"),
    "host": socket.gethostname(),
    "module_set": getenv("MODULE_SET"),
    "envname": getenv("ENVNAME"),
    "git_commit": getenv("GIT_COMMIT"),

    "slurm": {
        "job_id": getenv("SLURM_JOB_ID"),
        "array_job_id": getenv("SLURM_ARRAY_JOB_ID"),
        "task_id": getenv("SLURM_ARRAY_TASK_ID"),
        "job_name": getenv("SLURM_JOB_NAME"),
        "partition": getenv("SLURM_JOB_PARTITION"),
        "nodelist": getenv("SLURM_NODELIST"),
        "num_nodes": getenv("SLURM_JOB_NUM_NODES"),
        "cpus_per_task": getenv("SLURM_CPUS_PER_TASK"),
        "mem_per_node": getenv("SLURM_MEM_PER_NODE"),
        "mem_per_cpu": getenv("SLURM_MEM_PER_CPU"),
        "gpus": getenv("SLURM_GPUS"),
        "gpus_on_node": getenv("SLURM_GPUS_ON_NODE"),
        "submit_dir": getenv("SLURM_SUBMIT_DIR"),
    },
}

out_path = run_dir / "meta" / "run_meta.json"
out_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
print(f"Wrote {out_path}")
PY


#==============================#
#  BLOCK 5: TRAINING RUN       #
#==============================#
log_section "BLOCK 5: TRAINING RUN"

CMD=(python train.py "$CONFIG_PATH" --dataset="$DATASET" --seed="$SEED" --out_dir="$RUN_DIR/out" --device=cuda --compile=False --wandb_log=False)
printf "CMD: "; printf "%q " "${CMD[@]}"; echo
"${CMD[@]}"

#==============================#
#  BLOCK 6: POSTPROCESS (LIGHT)#
#==============================#
log_section "BLOCK 6: POSTPROCESS (LIGHT)"

if [[ "$DO_SUMMARIZE" == "1" ]]; then
  python summarize.py --run_dir "$RUN_DIR" || true
else
  echo "Skipping summarize.py (DO_SUMMARIZE=$DO_SUMMARIZE)"
fi

#==============================#
#  BLOCK 7: FINISH             #
#==============================#
log_section "BLOCK 7: FINISH"
echo "DONE. RUN_DIR=$RUN_DIR"