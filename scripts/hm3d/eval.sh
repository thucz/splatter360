

# checkpoint_path="outputs/2024-10-28/13-07-48/checkpoints/epoch_24-step_100000.ckpt"

# CUDA_VISIBLE_DEVICES=5 python -m src.main \
#     +experiment=replica_new\
#     model.encoder.shim_patch_size=8 \
#     model.encoder.downscale_factor=8 \
#     mode="test" \
#     dataset/view_sampler=evaluation \
#     checkpointing.load=$checkpoint_path \
#     dataset.view_sampler.index_path="assets/evaluation_index_replica.json"

output_dir="./outputs/splat360_log_depth_near0.1-100k/"
checkpoint_path="$output_dir/checkpoints/last.ckpt"
CUDA_VISIBLE_DEVICES=0 python -m src.main \
    +experiment=hm3d \
    model.encoder.shim_patch_size=8 \
    model.encoder.downscale_factor=8 \
    model.encoder.depth_sampling_type="log_depth" \
    output_dir=$output_dir \
    dataset.near=0.1 \
    mode="test" \
    dataset/view_sampler=evaluation \
    checkpointing.load=$checkpoint_path \
    dataset.view_sampler.index_path="assets/evaluation_index_hm3d.json"