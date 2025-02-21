#!/bin/bash

timestamp() {
  date +"%Y%m%d%H%M%S"
}

SLURM_JOB_NAME=46647662_sha_gcl_baseline
SLURM_JOB_ID=0014000

error_exit()
{
#   ----------------------------------------------------------------
#   Function for exit due to fatal program error
#       Accepts 1 argument:
#           string containing descriptive error message
#   Source: http://linuxcommand.org/lc3_wss0140.php
#   ----------------------------------------------------------------
    echo "$(timestamp) ERROR ${PROGNAME}: ${1:-"Unknown Error"}" 1>&2
    echo "$(timestamp) ERROR ${PROGNAME}: Exiting Early."
    exit 1
}

error_check()
{
#   ----------------------------------------------------------------
#   This function simply checks a passed return code and if it is
#   non-zero it returns 1 (error) to the calling script.  This was
#   really only created because I needed a method to store STDERR in
#   other scripts and also check $? but also leave open the ability
#   to add in other stuff, too.
#
#   Accepts 1 arguments:
#       return code from prior command, usually $?
#  ----------------------------------------------------------------
    TO_CHECK=${1:-0}

    if [ "$TO_CHECK" != '0' ]; then
        return 1
    fi

}

export PROJECT_DIR=${HOME}/gcl
export MODEL_NAME="${SLURM_JOB_NAME}_${SLURM_JOB_ID}_continue"
export LOGDIR=${PROJECT_DIR}/log
export DATA_DIR_VG_RCNN=${HOME}/datasets
export WEIGHT=${PROJECT_DIR}/checkpoints/${SLURM_JOB_NAME}/model_${SLURM_JOB_ID}.pth
MODEL_DIRNAME=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/


if [ -d "$MODEL_DIRNAME" ]; then
  error_exit "Aborted: ${MODEL_DIRNAME} exists." 2>&1 | tee -a ${LOGDIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
else
  export CUDA_VISIBLE_DEVICES=0
  export BATCH_SIZE=8
  export MAX_ITER=60000
  export CONFIG_FILE=configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml
  export DATA_DIR_VG_RCNN=${HOME}/datasets
  export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c); ((NUM_GPUS++))
  export USE_GT_BOX=True
  export USE_GT_OBJECT_LABEL=True
  export PRE_VAL=True
  export PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

  # Relation Augmentation: Semantic
  export NUM2AUG=4
  export MAX_BATCHSIZE_AUG=16
  export ALL_EDGES_FPATH=/gpfs/gpfs0/project/SDS/research/sds-rise/zhanwen/datasets/visual_genome/vg_gbnet/all_edges.pkl
  export STRATEGY='cooccurrence-pred_cov'
  export BOTTOM_K=30
  export USE_SEMANTIC=True
  export USE_GRAFT=False

  ${PROJECT_DIR}/scripts/train.sh
fi
