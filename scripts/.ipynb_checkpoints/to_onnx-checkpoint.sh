#!/bin/bash

ROOT = /home/u2127085/United-Perception/
T=`date +%m%d%H%M`
export ROOT=$ROOT
cfg=$2
export PYTHONPATH=$ROOT:$PYTHONPATH
# in this challenge, we only need cls tasks
export DEFAULT_TASKS=cls
CPUS_PER_TASK=${CPUS_PER_TASK:-2}

python -m up to_onnx \
  --ng=$1 \
  --launch=pytorch \
  --config=$cfg \
  --save_prefix=toonnx \
  --input_size=3x112x112 \
  2>&1 | tee log.deploy.$(basename $cfg) 
