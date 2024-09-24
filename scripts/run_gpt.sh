closeai_proxy_on
export OPENAI_API_KEY="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"

bash scripts/run_evals.sh "configs/models/gpt_config.yaml" "configs/tasks/image/image_mc_test.yaml,configs/tasks/image/image_mc_val.yaml,configs/tasks/image/image_tf_test.yaml,configs/tasks/image/image_tf_val.yaml" INFO 8 1
bash scripts/run_evals.sh "configs/models/gpt_config.yaml" "configs/tasks/video/video_mc_test.yaml,configs/tasks/video/video_mc_val.yaml,configs/tasks/video/video_tf_test.yaml,configs/tasks/video/video_tf_val.yaml" INFO 8 1
bash scripts/run_evals.sh "configs/models/gpt_config.yaml" "configs/tasks/text/text_mc.yaml,configs/tasks/text/text_tf.yaml" INFO 8 1
bash scripts/run_evals.sh "configs/models/gpt_config.yaml" "configs/tasks/3D/3D_mc_test.yaml,configs/tasks/3D/3D_mc_val.yaml,configs/tasks/3D/3D_tf_test.yaml,configs/tasks/3D/3D_tf_val.yaml" INFO 8 1

