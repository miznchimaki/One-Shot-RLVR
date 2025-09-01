#!/bin/bash
set -x

source ${HOME}/.bashrc  # for Centos
source ${HOME}/depends/anaconda3/etc/profile.d/conda.sh  # for Ubuntu 22.04
conda activate base
cd ${HOME}/projects/One-Shot-RLVR/

VLLM_ATTENTION_BACKEND_VAR=${1:-"XFORMERS"}
TRAIN_FILES=${2:-"${HOME}/projects/One-Shot-RLVR/data/train/one_shot_rlvr/pi1_r128.parquet"}
VAL_FILES=${3:-"${HOME}/projects/One-Shot-RLVR/data/test/math500.parquet"}
TRAIN_BATCH_SIZE=${4:-"128"}
VAL_BATCH_SIZE=${5:-"530"}
MAX_PROMPT_LENGTH=${6:-"1024"}
MAX_RESPONSE_LENGTH=${7:-"3072"}
REWARD_MANAGER=${8:-"naive"}
ACTOR_ROLLOUT_REF_MODEL_PATH=${9:-"${HOME}/ckpts/Qwen2.5-Math-1.5B"}
ACTOR_LR=${10:-"1e-6"}
USE_REMOVE_PADDING=${11:-"True"}
ACTOR_PPO_MINI_BATCH_SIZE=${12:-"128"}
ACTOR_USE_DYNAMIC_SIZE=${13:-"True"}
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${14:-"24000"}
ACTOR_USE_KL_LOSS=${15:-"True"}
ACTOR_KL_LOSS_COEF=${16:-"0.001"}
ACTOR_KL_LOSS_TYPE=${17:-"low_var_kl"}
ENABLE_GRADIENT_CHECKPOINTING=${18:-"True"}
ACTOR_FSDP_PARAM_OFFLOAD=${19:-"False"}
ACTOR_FSDP_GRAD_OFFLOAD=${20:-"False"}
ACTOR_FSDP_OPTIMIZER_OFFLOAD=${21:-"False"}
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=${22:-"2"}
ROLLOUT_NAME=${23:-"vllm"}
ROLLOUT_TEMPERATURE=${24:-"0.6"}
ROLLOUT_VAL_TEMPERATURE=${25:-"0.6"}
ROLLOUT_GPU_MEMORY_UTILIZATION=${26:-"0.7"}
ROLLOUT_N=${27:-"8"}
ROLLOUT_N_VAL=${28:-"1"}
REF_FSDP_PARAM_OFFLOAD=${29:-"True"}
KL_CTRL_KL_COEF=${30:-"0.001"}
CRITIC_WARMUP=${31:-"0"}
TRAINER_LOGGER=${32:-"['console', 'wandb']"}
TRAINER_PROJECT_NAME=${33:-"RL-post-training"}
TRAINER_EXPERIMENT_NAME=${34:-"One-Shot-RLVR-reproduce"}
TRAINER_VAL_BEFORE_TRAIN=${35:-"True"}
TRAINER_N_GPUS_PER_NODE=${36:-"8"}
TRAINER_NNODES=${37:-"1"}
TRAINER_SAVE_FREQ=${38:-"20"}
TRAINER_TEST_FREQ=${39:-"20"}
TRAINER_DEFAULT_HDFS_DIR=${40:-"null"}
TRAINER_TOTAL_EPOCHS=${41:-"2000"}
TRAINER_TOTAL_TRAINING_STEPS=${42:-""}


export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND_VAR}
OUTPUT_DIR=`python3 -m verl.trainer.main_ppo --cfg hydra --package hydra.run.dir`
if [ -d "${OUTPUT_DIR}" ]; then
    rm -rf "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

RL_PARAMS_ARR=(
    "algorithm.adv_estimator=grpo"
    "data.train_files=${TRAIN_FILES}"
    "data.val_files=${VAL_FILES}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.val_batch_size=${VAL_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "reward_model.reward_manager=${REWARD_MANAGER}"
    "actor_rollout_ref.model.path=${ACTOR_ROLLOUT_REF_MODEL_PATH}"
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
    "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.use_dynamic_bsz=${ACTOR_USE_DYNAMIC_SIZE}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.actor.use_kl_loss=${ACTOR_USE_KL_LOSS}"
    "actor_rollout_ref.actor.kl_loss_coef=${ACTOR_KL_LOSS_COEF}"
    "actor_rollout_ref.actor.kl_loss_type=${ACTOR_KL_LOSS_TYPE}"
    "actor_rollout_ref.model.enable_gradient_checkpointing=${ENABLE_GRADIENT_CHECKPOINTING}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_FSDP_PARAM_OFFLOAD}"
    "+actor_rollout_ref.actor.fsdp_config.grad_offload=${ACTOR_FSDP_GRAD_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_FSDP_OPTIMIZER_OFFLOAD}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE}"
    "actor_rollout_ref.rollout.name=${ROLLOUT_NAME}"
    "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
    "+actor_rollout_ref.rollout.val_temperature=${ROLLOUT_VAL_TEMPERATURE}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "+actor_rollout_ref.rollout.n_val=${ROLLOUT_N_VAL}"
    "actor_rollout_ref.ref.fsdp_config.param_offload=${REF_FSDP_PARAM_OFFLOAD}"
    "algorithm.kl_ctrl.kl_coef=${KL_CTRL_KL_COEF}"
    "trainer.critic_warmup=${CRITIC_WARMUP}"
    "trainer.logger=${TRAINER_LOGGER}"
    "trainer.project_name=${TRAINER_PROJECT_NAME}"
    "trainer.experiment_name=${TRAINER_EXPERIMENT_NAME}"
    "+trainer.val_before_train=${TRAINER_VAL_BEFORE_TRAIN}"
    "trainer.n_gpus_per_node=${TRAINER_N_GPUS_PER_NODE}"
    "trainer.nnodes=${TRAINER_NNODES}"
    "trainer.save_freq=${TRAINER_SAVE_FREQ}"
    "trainer.test_freq=${TRAINER_TEST_FREQ}"
    "trainer.default_hdfs_dir=${TRAINER_DEFAULT_HDFS_DIR}"
)
if [ -z "${TRAINER_TOTAL_TRAINING_STEPS}" ]; then
    RL_PARAMS_ARR+=( "trainer.total_epochs=${TRAINER_TOTAL_EPOCHS}" )
else
    RL_PARAMS_ARR+=( "trainer.total_training_steps=${TRAINER_TOTAL_TRAINING_STEPS}" )
fi


python3 -m verl.trainer.main_ppo "${RL_PARAMS_ARR[@]}" 2>&1 | tee "${OUTPUT_DIR}/user_outputs.log"
