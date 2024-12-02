


max_steps=100000
output_dir="./outputs/splat360_log_depth_near0.5-100k/"
CUDA_VISIBLE_DEVICES=4,5 python -m src.main \
    +experiment=hm3d data_loader.train.batch_size=1 \
    model.encoder.shim_patch_size=8 \
    model.encoder.downscale_factor=8 \
    trainer.max_steps=$max_steps \
    model.encoder.depth_sampling_type="log_depth" \
    output_dir=$output_dir \
    dataset.near=0.5

    # \
    # checkpointing.load=$resume_path