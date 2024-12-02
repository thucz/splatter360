




max_steps=100000
output_dir="./outputs/splat360_log_depth_near0.1-100k/"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.main \
#     +experiment=hm3d data_loader.train.batch_size=1 \
#     model.encoder.shim_patch_size=8 \
#     model.encoder.downscale_factor=8 \
#     trainer.max_steps=$max_steps \
#     model.encoder.depth_sampling_type="log_depth" \
#     output_dir=$output_dir \
#     dataset.near=0.1





# checkpoint_path="$output_dir/checkpoints/last.ckpt"
# CUDA_VISIBLE_DEVICES=0 python -m src.main \
#     +experiment=hm3d \
#     model.encoder.shim_patch_size=8 \
#     model.encoder.downscale_factor=8 \
#     model.encoder.depth_sampling_type="log_depth" \
#     output_dir=$output_dir \
#     dataset.near=0.1 \
#     mode="test" \
#     dataset/view_sampler=evaluation \
#     checkpointing.load=$checkpoint_path \
#     dataset.view_sampler.index_path="assets/evaluation_index_hm3d.json"\
#     test.eval_depth=true


# # eval on Replica with 3 context views
# checkpoint_path="$output_dir/checkpoints/last.ckpt"
# CUDA_VISIBLE_DEVICES=7 python -m src.main \
#     +experiment=replica \
#     model.encoder.shim_patch_size=8 \
#     model.encoder.downscale_factor=8 \
#     model.encoder.depth_sampling_type="log_depth" \
#     output_dir=$output_dir \
#     dataset.near=0.1 \
#     mode="test" \
#     dataset/view_sampler=evaluation \
#     checkpointing.load=$checkpoint_path \
#     dataset.view_sampler.index_path="assets/evaluation_index_replica_3views.json"\
#     dataset.view_sampler.num_context_views=3 \
#     test.eval_depth=true


# # eval on HM3D with 3 context views
# checkpoint_path="$output_dir/checkpoints/last.ckpt"
# CUDA_VISIBLE_DEVICES=7 python -m src.main \
#     +experiment=hm3d \
#     model.encoder.shim_patch_size=8 \
#     model.encoder.downscale_factor=8 \
#     model.encoder.depth_sampling_type="log_depth" \
#     output_dir=$output_dir \
#     dataset.near=0.1 \
#     mode="test" \
#     dataset/view_sampler=evaluation \
#     checkpointing.load=$checkpoint_path \
#     dataset.view_sampler.index_path="assets/evaluation_index_hm3d_3views.json"\
#     dataset.view_sampler.num_context_views=3 \
#     test.eval_depth=true

# eval on HM3D with narrow baseline (interval: 50 frame)
checkpoint_path="$output_dir/checkpoints/last.ckpt"
CUDA_VISIBLE_DEVICES=5  python -m src.main \
    +experiment=hm3d \
    model.encoder.shim_patch_size=8 \
    model.encoder.downscale_factor=8 \
    model.encoder.depth_sampling_type="log_depth" \
    output_dir=$output_dir \
    dataset.near=0.1 \
    mode="test" \
    dataset/view_sampler=evaluation \
    checkpointing.load=$checkpoint_path \
    dataset.view_sampler.index_path="assets/evaluation_index_hm3d_narrow.json"\
    test.eval_depth=true

# # eval on Replica with narrow baseline (interval: 50 frame)
# checkpoint_path="$output_dir/checkpoints/last.ckpt"
# CUDA_VISIBLE_DEVICES=5  python -m src.main \
#     +experiment=replica \
#     model.encoder.shim_patch_size=8 \
#     model.encoder.downscale_factor=8 \
#     model.encoder.depth_sampling_type="log_depth" \
#     output_dir=$output_dir \
#     dataset.near=0.1 \
#     mode="test" \
#     dataset/view_sampler=evaluation \
#     checkpointing.load=$checkpoint_path \
#     dataset.view_sampler.index_path="assets/evaluation_index_replica_narrow.json"\
#     test.eval_depth=true
# # Replica render video
# checkpoint_path="$output_dir/checkpoints/last.ckpt"
# CUDA_VISIBLE_DEVICES=0 python -m src.main \
#     +experiment=replica \
#     model.encoder.shim_patch_size=8 \
#     model.encoder.downscale_factor=8 \
#     model.encoder.depth_sampling_type="log_depth" \
#     output_dir=$output_dir \
#     dataset.near=0.1 \
#     mode="test" \
#     dataset/view_sampler=evaluation \
#     checkpointing.load=$checkpoint_path \
#     dataset.view_sampler.index_path="assets/evaluation_index_replica_video.json" \
#     test.save_video=true \
#     test.save_image=false \
#     test.compute_scores=false


# # HM3D render video
# checkpoint_path="$output_dir/checkpoints/last.ckpt"
# CUDA_VISIBLE_DEVICES=0 python -m src.main \
#     +experiment=hm3d \
#     model.encoder.shim_patch_size=8 \
#     model.encoder.downscale_factor=8 \
#     model.encoder.depth_sampling_type="log_depth" \
#     output_dir=$output_dir \
#     dataset.near=0.1 \
#     mode="test" \
#     dataset/view_sampler=evaluation \
#     checkpointing.load=$checkpoint_path \
#     dataset.view_sampler.index_path="assets/evaluation_index_hm3d_video.json" \
#     test.save_video=true \
#     test.save_image=false \
#     test.compute_scores=false \
#     test.eval_depth=true # render depth video

# # Replica render video
# checkpoint_path="$output_dir/checkpoints/last.ckpt"
# CUDA_VISIBLE_DEVICES=0 python -m src.main \
#     +experiment=replica \
#     model.encoder.shim_patch_size=8 \
#     model.encoder.downscale_factor=8 \
#     model.encoder.depth_sampling_type="log_depth" \
#     output_dir=$output_dir \
#     dataset.near=0.1 \
#     mode="test" \
#     dataset/view_sampler=evaluation \
#     checkpointing.load=$checkpoint_path \
#     dataset.view_sampler.index_path="assets/evaluation_index_replica_video.json" \
#     test.save_video=true \
#     test.save_image=false \
#     test.compute_scores=false \
#     test.eval_depth=true # render depth video