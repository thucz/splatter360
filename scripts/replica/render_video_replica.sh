# Replica render video
output_dir="./outputs/splat360_log_depth_near0.1-100k/"
checkpoint_path="$output_dir/checkpoints/last.ckpt"

CUDA_VISIBLE_DEVICES=7 python -m src.main \
    +experiment=replica \
    model.encoder.shim_patch_size=8 \
    model.encoder.downscale_factor=8 \
    model.encoder.depth_sampling_type="log_depth" \
    output_dir=$output_dir \
    dataset.near=0.1 \
    mode="test" \
    dataset/view_sampler=evaluation \
    checkpointing.load=$checkpoint_path \
    dataset.view_sampler.index_path="assets/evaluation_index_replica_video.json" \
    test.save_video=true \
    test.save_image=false \
    test.compute_scores=false \
    test.eval_depth=true # render depth video