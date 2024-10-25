#!/bin/bash
if [ -z "$ACCOUNT" ]; then
    echo "run export ACCOUNT=your_archer2_account_name before using this script"
    exit 2
fi
repo_root=$( cd -- "$(realpath $( dirname -- "${BASH_SOURCE[0]}" ))/../../" &> /dev/null && pwd )

# General set up script
${repo_root}/utils/set_up_run.bash $*

# Archer-specific set up
run_name="$1"
run_dir="${repo_root}/runs/${run_name}"
archer_dir="${repo_root}/hpc/archer2"
script_name="${2%.py}.py"

# Copy Slurm script, setting account, image directory
sed "${archer_dir}/sub_fd-singularity.slurm" -e "s|\[SET_ACCOUNT_HERE\]|$ACCOUNT|" -e "s|\[SET_IMAGE_DIR_HERE\]|${repo_root}/hpc/archer2/singularity|"  -e "s|\[SET_SCRIPT_NAME_HERE\]|${script_name}|"> "${run_dir}/sub_fd-singularity.slurm"
