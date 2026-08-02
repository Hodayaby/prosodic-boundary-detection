#!/bin/bash

#SBATCH --job-name=train_whisper3

#SBATCH --output=train_whisper3_%j.out

#SBATCH --error=train_whisper3_%j.err

#SBATCH --partition=L4-12h

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4

#SBATCH --mem=40G

#SBATCH --time=12:00:00

#SBATCH --mail-user=shirabn14@gmail.com 

#SBATCH --mail-type=BEGIN,END,FAIL



set -e



cd $SLURM_SUBMIT_DIR



echo "========================================"

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

ls data/splits/train_chunks.csv

ls data/splits/valid_chunks.csv

ls data/splits/test_chunks.csv

ls train_whisper_binary.py



echo "========================================"

echo "Starting training"

echo "========================================"



python train_whisper_binary.py



echo "========================================"

echo "Finished at: $(date)"

echo "========================================"



echo "Saved model directory:"

ls -lah whisper-binary-boundary-model3

