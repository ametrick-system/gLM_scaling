#!/bin/bash
#SBATCH --job-name=glm-preflight
#SBATCH --partition=gpu_rtx6000
#SBATCH --account=pi_ac3458
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --constraint=gpu:rtx_pro_6000_blackwell
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/preflight_%j.out
#SBATCH --error=logs/preflight_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amy.metrick@yale.edu
# ---------------------------------------------------------------------------
# Verify the software stack on a real Blackwell node, as a BATCH job.
#
#     mkdir -p logs && sbatch preflight.slurm
#     # then, when it finishes:
#     cat logs/preflight_*.out
#
# Needs no storage setup. The inductor cache defaults under $HOME and this job
# writes nothing else, so you do not have to resolve the group-space symlinks
# before running it.
#
# RETARGETING: command-line options override #SBATCH directives, so you can
# point this at whatever card is actually free without editing the file:
#     sbatch -p scavenge_gpu -C gpu:h100 scripts/preflight.slurm   # fastest queue
#     sbatch -p gpu_h100     -C gpu:h100 scripts/preflight.slurm
#     sbatch -p gpu          -C gpu:l40s scripts/preflight.slurm
#
# Prefer scavenge_gpu for this job. The named GPU partitions throttle a single
# user hard (part_gpu_b200 and part_gpu_h200 to 6 concurrent jobs, part_gpu and
# part_gpu_h100 to 12, part_gpu_rtx6000 to 16), whereas scavenge_gpu reaches
# all 388 GPUs on the cluster with no per-user cap -- it just runs at lowest
# priority and is preemptible. For a 20-minute check that is pure upside: it
# backfills almost immediately, and being preempted costs a resubmit.
# Verify the design fits that card first -- for the 10^16-10^19 window the
# largest model is 1.20B, needing 21.6 GB of AdamW state, so 24 GB and 32 GB
# cards are out and a40 exceeds the 48 h wall clock. Viable: l40s, blackwell,
# h100, h200, b200.
#
# Why batch rather than salloc: `salloc` defaults to the `interactive` QOS,
# which gpu_devel allows and gpu_rtx6000 appears not to ("Invalid qos
# specification"). Submitting with --qos=normal avoids the question entirely --
# and more usefully, this job uses the same partition/account/constraint/QOS
# directives as sweep.slurm, so a successful preflight proves the submission
# path as well as the stack. That is the thing you actually want verified
# before launching 90 tasks.
#
# Runs check_env.py under BOTH stacks so you can compare them in one go.
# Two working stacks is cheap insurance; if only one passes you know which.
# ---------------------------------------------------------------------------
set -o pipefail

module load miniconda
conda activate glm

echo "node    $(hostname)"
echo "gpu     $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
echo "python  $(command -v python)"

python /home/asm242/gLM_scaling/tests/check_env.py