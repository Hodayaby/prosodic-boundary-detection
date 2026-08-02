#!/bin/bash

#SBATCH --job-name=eval_test_model2

#SBATCH --output=eval_test_model2_%j.out

#SBATCH --error=eval_test_model2_%j.err

#SBATCH --partition=L4-4h

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4

#SBATCH --mem=32G

#SBATCH --time=02:00:00

#SBATCH --mail-user=shirabn14@gmail.com 

#SBATCH --mail-type=BEGIN,END,FAIL



set -e



cd $SLURM_SUBMIT_DIR



echo "========================================"

echo "Running TEST evaluation for model2"

echo "Running on: $(hostname)"

echo "Working dir: $(pwd)"

echo "Starting at: $(date)"

echo "Job ID: $SLURM_JOB_ID"

echo "========================================"



nvidia-smi



source venv/bin/activate



echo "Python:"

which python

python --version



echo "Torch / CUDA:"

python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"



echo "Checking required files:"

ls whisper-binary-boundary-model2

ls data/splits/test_chunks.csv

ls evaluate_boundary_improved.py



echo "========================================"

echo "Starting evaluation on TEST split"

echo "========================================"



python evaluate_boundary_final2.py \
  --eval-csv data/splits/test_chunks.csv \
  --output-dir evaluation_outputs3_test_t05 \
  --threshold 0.5



echo "========================================"

echo "Finished at: $(date)"

echo "========================================"



echo "Evaluation output directory:"

ls -lah evaluation_outputs3_test_t05

