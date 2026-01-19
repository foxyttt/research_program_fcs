#!/bin/bash

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <experiment_group_name> <dataset>"
  exit 1
fi

NAME=$1
DATASET=$2

mkdir -p "$NAME"/{config,runs,res}

mkdir "$NAME/config/baseline"

touch "$NAME/config/baseline/config.py"
touch "$NAME/config/baseline/run.bash"

cat > "$NAME/config/baseline/run.bash" <<EOF
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="\$(cd -- "\$(dirname -- "\${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_PATH="\$SCRIPT_DIR/../../template.sbatch"

EXP_GROUP="$NAME"
CUR_EXP_NAME="baseline"
EXP_NAME="\${EXP_GROUP}_\${CUR_EXP_NAME}"

CONFIG_PATH="\$SCRIPT_DIR/config.py"
DATASET="$DATASET"

RUNS_ROOT="\$SCRIPT_DIR/../../runs"

export EXP_DESC="Baseline for $NAME"

EXTRA_SBATCH_ARGS=(--time=06:00:00)

sbatch "\${EXTRA_SBATCH_ARGS[@]}" \\
  --job-name="nanogpt-$NAME-\${EXP_NAME}" \\
  --export=ALL,EXP_NAME="\$EXP_NAME",EXP_DESC="\$EXP_DESC",CONFIG_PATH="\$CONFIG_PATH",RUNS_ROOT="\$RUNS_ROOT",DATASET="\$DATASET" \\
  "\$TEMPLATE_PATH"
EOF

chmod +x "$NAME/config/baseline/run.bash"

cp "template.sbatch" "$NAME/template.sbatch"
cp "summarize.py" "$NAME/summarize.py"

VENV_DIR="$NAME/venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install scipy
  deactivate
fi


echo "Experiment group '$NAME' created."
echo "Put config to '$NAME/config/baseline/config.py' and run '$NAME/config/baseline/run/bash'"
echo "Last will run template.sbatch"
echo "After run you can run summarize.py (do not forget activate '$VENV_DIR' if run --mode filter)"
