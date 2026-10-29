#!/bin/bash
#SBATCH --job-name=gdna-corpora
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/corpora_%j.out
#SBATCH --error=logs/corpora_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amy.metrick@yale.edu
# McCleary CPU job (no GPU). Streams the chr1+2+3 multiz100way MAF, builds the
# orthologous core ONCE, writes one .bin + .mask.bin + .meta.json per condition
# in the manifest. ~1-2 h. --mem=32G holds ~20 species x 143 Mb core in RAM.
#
# 1) download the 3 chrom MAFs to scratch (12.8 GB; age-purged, regenerable):
#      D=~/palmer_scratch/multiz100way; mkdir -p "$D"
#      for c in chr1 chr2 chr3; do
#        wget -P "$D" https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz100way/maf/$c.maf.gz
#      done
# 2) submit (GDNA_MAF is space-separated; output to scratch, corpora are large):
#      GDNA_MAF="$D/chr1.maf.gz $D/chr2.maf.gz $D/chr3.maf.gz" \
#      GDNA_MANIFEST=conditions.json GDNA_OUT=~/palmer_scratch/corpora \
#      sbatch build_corpora.mccleary.slurm
#
# No --account (day partition defaults to the gerstein assoc; add
# #SBATCH --account=gerstein only if SLURM rejects it).
set -uo pipefail
GDNA_ROOT="${GDNA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${GDNA_ROOT}" || { echo "ABORT: cannot cd to ${GDNA_ROOT}" >&2; exit 1; }
: "${GDNA_MANIFEST:?set GDNA_MANIFEST=conditions.json}"
: "${GDNA_MAF:?set GDNA_MAF to the space-separated chr1/chr2/chr3 MAF paths}"
for need in env.sh build_corpora.py prepare.py "${GDNA_MANIFEST}"; do
  [[ -f "${need}" ]] || { echo "ABORT: ${need} missing under ${GDNA_ROOT}" >&2; exit 1; }
done

module load miniconda
conda activate glm
mkdir -p logs

python build_corpora.py --maf ${GDNA_MAF} \
  --manifest "${GDNA_MANIFEST}" \
  --out-dir "${GDNA_OUT:-corpora}"
