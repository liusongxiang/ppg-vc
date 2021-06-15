# cuda related
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# path related
export PRJ_ROOT="./"
if [ -e "${PRJ_ROOT}/tool/venv/bin/activate" ]; then
    # shellcheck disable=SC1090
    . "${PRJ_ROOT}/tool/venv/bin/activate"
  fi

# python related
export OMP_NUM_THREADS=1
export PYTHONIOENCODING=UTF-8
export MPL_BACKEND=Agg
