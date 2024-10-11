# make sure to switch to vila environment to evaluate vila
# bash scripts/run_evals.sh "configs/models/vila_1.5_13b_config.yaml" "configs/tasks/image/image_mc_test.yaml,configs/tasks/image/image_mc_val.yaml,configs/tasks/image/image_tf_test.yaml,configs/tasks/image/image_tf_val.yaml" INFO 2 1
# bash scripts/run_evals.sh "configs/models/vila_1.5_13b_config.yaml" "configs/tasks/video/video_mc_test.yaml,configs/tasks/video/video_mc_val.yaml,configs/tasks/video/video_tf_test.yaml,configs/tasks/video/video_tf_val.yaml" INFO 1 1
# bash scripts/run_evals.sh "configs/models/vila_1.5_13b_config.yaml" "configs/tasks/text/text_mc.yaml,configs/tasks/text/text_tf.yaml" INFO 4 1
# bash scripts/run_evals.sh "configs/models/vila_1.5_13b_config.yaml" "configs/tasks/3D/3D_mc_test.yaml,configs/tasks/3D/3D_mc_val.yaml,configs/tasks/3D/3D_tf_test.yaml,configs/tasks/3D/3D_tf_val.yaml" INFO 4 1


# bash scripts/run_evals.sh "configs/models/vila_1.5_40b_config.yaml" "configs/tasks/image/image_mc_test.yaml,configs/tasks/image/image_mc_val.yaml,configs/tasks/image/image_tf_test.yaml,configs/tasks/image/image_tf_val.yaml" INFO 1 1
bash scripts/run_evals.sh "configs/models/vila_1.5_40b_config.yaml" "configs/tasks/video/video_mc_test.yaml,configs/tasks/video/video_mc_val.yaml,configs/tasks/video/video_tf_test.yaml,configs/tasks/video/video_tf_val.yaml" INFO 1 1
# bash scripts/run_evals.sh "configs/models/vila_1.5_40b_config.yaml" "configs/tasks/text/text_mc.yaml,configs/tasks/text/text_tf.yaml" INFO 1 1
# bash scripts/run_evals.sh "configs/models/vila_1.5_40b_config.yaml" "configs/tasks/3D/3D_mc_test.yaml,configs/tasks/3D/3D_mc_val.yaml,configs/tasks/3D/3D_tf_test.yaml,configs/tasks/3D/3D_tf_val.yaml" INFO 1 1

# bash scripts/run_evals.sh "configs/models/longvila_config.yaml" "configs/tasks/image/image_mc_test.yaml,configs/tasks/image/image_mc_val.yaml,configs/tasks/image/image_tf_test.yaml,configs/tasks/image/image_tf_val.yaml" INFO 2 1
# bash scripts/run_evals.sh "configs/models/longvila_config.yaml" "configs/tasks/video/video_mc_test.yaml,configs/tasks/video/video_mc_val.yaml,configs/tasks/video/video_tf_test.yaml,configs/tasks/video/video_tf_val.yaml" INFO 1 1
# bash scripts/run_evals.sh "configs/models/longvila_config.yaml" "configs/tasks/text/text_mc.yaml,configs/tasks/text/text_tf.yaml" INFO 4 1
# bash scripts/run_evals.sh "configs/models/longvila_config.yaml" "configs/tasks/3D/3D_mc_test.yaml,configs/tasks/3D/3D_mc_val.yaml,configs/tasks/3D/3D_tf_test.yaml,configs/tasks/3D/3D_tf_val.yaml" INFO 4 1