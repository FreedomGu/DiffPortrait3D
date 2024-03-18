#!/bin/bash
torchrun --master_port 12000 inference.py \
--train_batch_size 1 \
--model_config control_model/ControlNet/models/cldm_v15_diffportrait3D.yaml \
--test_dataset pano_head \
--control_mode controlnet_important \
--local_image_dir ./test_log/image_log/demo_inference \
--image_folder  ./test_samples/sample_images \
--resume_dir ./checkpoints/model_state-540000-001.th \
--sequence_path ./test_samples/sample_camera_condition \
--denoise_from_fea_map \
--fea_condition_root ./test_samples/sample_3D_aware_feature \
$@

