# MODEL.ROI_RELATION_HEAD.USE_GT_BOX True
# MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True
# SOLVER.VAL_PERIOD 5000
# SOLVER.CHECKPOINT_PERIOD 5000
# GLOVE_DIR /home/share/datasets/vg/glove
# OUTPUT_DIR /home/share/datasets/output/SHA_GCL_VG_PredCls_test

#!/bin/bash
timestamp() {
  date +"%Y-%m-%d%H%M%S"
}

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

get_mode()
{
  if [[ ${USE_GT_BOX} == "True" ]] && [[ ${USE_GT_OBJECT_LABEL} == "True" ]]; then
    echo "predcls"
  elif [[ ${USE_GT_BOX} == "True" ]] && [[ ${USE_GT_OBJECT_LABEL} == "False" ]]; then
    echo "sgcls"
  elif [[ ${USE_GT_BOX} == "False" ]] && [[ ${USE_GT_OBJECT_LABEL} == "False" ]]; then
    echo "sgdet"
  else
    error_exit "Illegal USE_GT_BOX=${USE_GT_BOX} and USE_GT_OBJECT_LABEL=${USE_GT_OBJECT_LABEL} provided."
  fi
}

export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCHELASTIC_MAX_RESTARTS=0
export MODE=$(get_mode)
echo "TRAINING ${MODE} model ${MODEL_NAME}"
cd ${PROJECT_DIR}
MODEL_DIRNAME=${PROJECT_DIR}/checkpoints/${MODEL_NAME}/
if [ -d "$MODEL_DIRNAME" ]; then
  error_exit "Aborted: ${MODEL_DIRNAME} exists." 2>&1 | tee -a ${LOGDIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out
fi
mkdir ${MODEL_DIRNAME} &&
cp -r ${PROJECT_DIR}/.git/ ${MODEL_DIRNAME} &&
cp -r ${PROJECT_DIR}/tools/ ${MODEL_DIRNAME} &&
cp -r ${PROJECT_DIR}/scripts/ ${MODEL_DIRNAME} &&
cp -r ${PROJECT_DIR}/maskrcnn_benchmark/ ${MODEL_DIRNAME} &&
python -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=$NUM_GPUS \
  ${PROJECT_DIR}/tools/relation_train_net.py \
  --config-file ${CONFIG_FILE} \
  SOLVER.IMS_PER_BATCH ${BATCH_SIZE} \
  SOLVER.MAX_ITER ${MAX_ITER} \
  GLOBAL_SETTING.DATASET_CHOICE 'VG' \
  GLOBAL_SETTING.RELATION_PREDICTOR 'TransLike_GCL' \
  GLOBAL_SETTING.BASIC_ENCODER 'Hybrid-Attention' \
  GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE 'divide4' \
  GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE 'KL_logit_TopDown' \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX ${USE_GT_BOX} \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL ${USE_GT_OBJECT_LABEL} \
  TEST.IMS_PER_BATCH ${NUM_GPUS} \
  SOLVER.PRE_VAL ${PRE_VAL} \
  DTYPE "float16" \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR ${PROJECT_DIR}/datasets/vg/ \
  MODEL.PRETRAINED_DETECTOR_CKPT_VG ./checkpoints/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR ${PROJECT_DIR}/checkpoints/${MODEL_NAME} 2>&1 | tee ${MODEL_DIRNAME}/log_train.log &&
echo "Finished training ${MODE} model ${MODEL_NAME}" || echo "Failed to train ${MODE} model ${MODEL_NAME}"