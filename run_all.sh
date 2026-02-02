#!/usr/bin/env bash

NINST=${1:-0}

function run_ours {
    local impl=$1
    local prec=$2

    local ts=$(date +"%y%m%d_%H%M%S")
    local id=$(openssl rand -hex 2)

    local extra_flag=""

    if [ "$prec" = "fp32" ]; then
        extra_flag="--fp32"
    fi

    cmd="python main_train_tinyvit_mnist.py --impl $impl $extra_flag"
    echo "Running: $cmd"
    $cmd 2>&1 | tee logs/${ts}_${id}_ours_${impl}_${prec}
}

function run_te {
    local impl=$1
    local prec=$2

    local ts=$(date +"%y%m%d_%H%M%S")
    local id=$(openssl rand -hex 2)

    local extra_flag=""

    if [ "$prec" = "fp32" ]; then
        extra_flag="--fp32"
    fi

    cmd="python main_te_train_tinyvit_mnist.py --recipe $impl $extra_flag"
    echo "Running: $cmd"
    $cmd 2>&1 | tee logs/${ts}_${id}_te_${impl}_${prec}
}

# Run N copies of a function with the same args, in parallel, then wait.
run_parallel() {
  copies=$1; shift
  target=$1; shift

  pids=()
  for i in $(seq 1 $copies); do
    RUN_IDX=$i $target "$@" &
    pids+=($!)
    echo "  started $target $* (RUN_IDX=$i, PID=${pids[-1]})"
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait $pid; then failed=1; fi
  done
  return $failed
}

mkdir -p logs

if [ "$NINST" -gt 1 ]; then
    # dummy run to ensure data are downloaded, avoiding parallel runs error out
    python main_train_tinyvit_mnist.py --impl torch
    echo "Running $NINST instances"
    run_parallel $NINST run_ours torch bf16
    run_parallel $NINST run_ours torch fp32
    run_parallel $NINST run_ours cublaslt bf16
    run_parallel $NINST run_ours cublaslt fp32
    run_parallel $NINST run_ours cublaslt_mxfp8 bf16
    run_parallel $NINST run_ours cublaslt_mxfp8 fp32
    run_parallel $NINST run_ours cublaslt_nvfp4 bf16
    run_parallel $NINST run_ours cublaslt_nvfp4 fp32
    run_parallel $NINST run_ours cublaslt_nvf4_fw_mxf8_bw bf16
    run_parallel $NINST run_ours cublaslt_nvf4_fw_mxf8_bw fp32

    run_parallel $NINST run_te base bf16
    run_parallel $NINST run_te base fp32
    run_parallel $NINST run_te fp8 bf16
    run_parallel $NINST run_te fp8 fp32
    run_parallel $NINST run_te mxfp8 bf16
    run_parallel $NINST run_te mxfp8 fp32
    run_parallel $NINST run_te nvfp4 bf16
    run_parallel $NINST run_te nvfp4 fp32
    # Note: TE does not support nvfp4 in fp32 mode due to internal RHT
else
    echo "Skipping — NINST <= 0"
    run_ours torch bf16
    run_ours torch fp32
    run_ours cublaslt bf16
    run_ours cublaslt fp32
    run_ours cublaslt_mxfp8 bf16
    run_ours cublaslt_mxfp8 fp32
    run_ours cublaslt_nvfp4 bf16
    run_ours cublaslt_nvfp4 fp32
    run_ours cublaslt_nvf4_fw_mxf8_bw bf16
    run_ours cublaslt_nvf4_fw_mxf8_bw fp32

    run_te base bf16
    run_te base fp32
    run_te fp8 bf16
    run_te fp8 fp32
    run_te mxfp8 bf16
    run_te mxfp8 fp32
    run_te nvfp4 bf16
    run_te nvfp4 fp32
    # Note: TE does not support nvfp4 in fp32 mode due to internal RHT
fi

