#!/usr/bin/env bash
set -euo pipefail

### CONFIG ###########################################################

# Directory with ALL your generated images
SRC_DIR="/export/data/abespalo/eval/top_40_eval/eval_coconut_base_50k/step_13000"

# Path to ref embeddings file
REF_EMBED_FILE="/export/data/abespalo/datasets/unsplash-research-dataset-lite-latest/unsplash_images_all/ref_embeddings.npy"

# CMMD script (run from repo root so this is just main.py)
MAIN_PY="main.py"

# How many images per subset (e.g. 120)
SUBSET_SIZE=64

# How many bootstrap runs
N_RUNS=10

# Other flags
BATCH_SIZE=32
MAX_COUNT=25000

# Python executable (env)
PYTHON_BIN="python"

# Output log file
LOG_FILE="cmmd_bootstrap_results.txt"

#####################################################################

echo "Writing results to: ${LOG_FILE}"
echo "# CMMD bootstrap runs" > "${LOG_FILE}"
echo "# SRC_DIR=${SRC_DIR}" >> "${LOG_FILE}"
echo "# SUBSET_SIZE=${SUBSET_SIZE}, N_RUNS=${N_RUNS}" >> "${LOG_FILE}"
echo "# REF_EMBED_FILE=${REF_EMBED_FILE}" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"

for ((i=1; i<=N_RUNS; i++)); do
    echo "=== Run ${i}/${N_RUNS} ==="

    # Make temporary directory for this subset
    TMPDIR="$(mktemp -d)"

    # Sample subset of files (non-recursive)
    mapfile -t FILES < <(find "${SRC_DIR}" -maxdepth 1 -type f | shuf | head -n "${SUBSET_SIZE}")

    # Symlink them into temp dir
    for f in "${FILES[@]}"; do
        ln -s "${f}" "${TMPDIR}/$(basename "${f}")"
    done

    # Run CMMD on this subset
    # NOTE: first positional arg is "" so ref_dir is not set
    # second positional arg is eval_dir = TMPDIR
    OUTPUT="$(${PYTHON_BIN} "${MAIN_PY}" "" "${TMPDIR}" \
        --ref_embed_file="${REF_EMBED_FILE}" \
        --batch_size="${BATCH_SIZE}" \
        --max_count="${MAX_COUNT}" 2>&1 || true)"

    # Try to grab the last floating-point number from the output as the score
    SCORE="$(printf '%s\n' "${OUTPUT}" | grep -Eo '[0-9]+\.[0-9]+' | tail -n1 || echo "NA")"

    echo "Run ${i} CMMD: ${SCORE}"
    {
        echo "### Run ${i}"
        echo "CMMD: ${SCORE}"
        echo "Raw output:"
        echo "${OUTPUT}"
        echo ""
    } >> "${LOG_FILE}"

    # Clean up
    rm -rf "${TMPDIR}"
done

echo "Done. Results logged in ${LOG_FILE}"
