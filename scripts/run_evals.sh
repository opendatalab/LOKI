MODEL_CONFIG_PATHS_STRING=$1

TASK_CONFIG_PATHS_STRING=$2

VERBOSITY=$3


IFS="," read -r -a MODEL_CONFIG_PATHS <<< "$MODEL_CONFIG_PATHS_STRING"

IFS="," read -r -a TASK_CONFIG_PATHS <<< "$TASK_CONFIG_PATHS_STRING"

echo "model configs: ${MODEL_CONFIG_PATHS[@]}"
echo "task configs: ${TASK_CONFIG_PATHS[@]}"

for MODEL_CONFIG_PATH in "${MODEL_CONFIG_PATHS[@]}"
do
    for TASK_CONFIG_PATH in "${TASK_CONFIG_PATHS[@]}"
    do
        python -m accelerate.commands.launch --num_processes=1 run.py --model_config_path "${MODEL_CONFIG_PATH}" --task_config_path "${TASK_CONFIG_PATH}" --verbosity "${VERBOSITY}"
    done
done
