#!/bin/bash
this_dir=$( cd -- "$(realpath $( dirname -- "${BASH_SOURCE[0]}" ))" &> /dev/null && pwd )

check_ret() {
    ret_code="$?"
    if [ $# -ne 1 ]; then
        echo "check_ret called with $# args (rather than 1) "
        exit 99 
    fi
    lbl=$1
    if [ $ret_code -ne 0 ]; then
        echo "$lbl failed with return code $ret_code"
        exit $ret_code
    fi
}

for cmd in "docker" "singularity"
do
    if ! command -v "$cmd" &> /dev/null
    then
        echo "$cmd command not on path or doesn't exist"
        exit 1
    fi
done

if ! [[ "$*" == *"--no-cache"* ]]; then
    echo "N.B. Pass --no-cache to this script to disable layer caching and force all images to rebuild from scratch"
fi

# Build the docker images
compose_file="${this_dir}/docker-compose.yml"
build_cmd="docker compose -f ${compose_file} build $*"
eval "$build_cmd"
check_ret "compose build with ($build_cmd)"

test_cmd="docker compose up --abort-on-container-failure firedrake"
eval "$test_cmd"
check_ret "firedrake tests ($test_cmd)"

# Clean up containers (but don't bother checking return code)
docker container rm petsc-env firedrake &> /dev/null

# Convert to sandbox singularity
singularity build --force \
   --sandbox ./firedrake-singularity \
   docker-daemon://firedrake:latest
check_ret "singularity sandbox build"

# Build singularity image
singularity build --force \
firedrake.sif \
./firedrake-singularity
check_ret "singularity image build"