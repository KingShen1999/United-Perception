#!/bin/bash

ROOT = /home/u2127085/United-Perception/
T=`date +%m%d%H%M`
export ROOT=$ROOT
cfg=$2
export PYTHONPATH=$ROOT:$PYTHONPATH
# in this challenge, we only need cls tasks
export DEFAULT_TASKS=cls
python -m up train \
  --ng=$1 \
  --launch=pytorch \
  --config=$cfg \
  --display=10 \
  2>&1 | tee log.train.$T.$(basename $cfg) 
