#!/bin/bash
#SBATCH --job-name=eval_analyze
#SBATCH --output=eval_analyze_%j.out
#SBATCH --error=eval_analyze_%j.err
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-user=shirabn14@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -e

cd $SLURM_SUBMIT_DIR

echo "Running on: $(hostname)"
echo "Working dir: $(pwd)"
echo "Starting at: $(date)"

nvidia-smi

source venv/bin/activate

python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "========================================"
echo "Step 1: forced transcript evaluation"
echo "========================================"
python evaluate_forced_transcript.py

echo "========================================"
echo "Step 2: basic false positive analysis"
echo "========================================"
python analyze_false_positives.py

echo "========================================"
echo "Step 3: merge evaluation with original labels"
echo "========================================"
python merge_eval_with_original_labels.py

echo "========================================"
echo "Step 4: analyze high-confidence FP original labels"
echo "========================================"
python analyze_fp_original_labels.py

echo "Finished at: $(date)"
