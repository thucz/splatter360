


max_steps=100000
output_dir="./outputs/splat360-100k/"
# resume_path="outputs/2024-10-28/13-07-48/checkpoints/epoch_29-step_120000.ckpt"
CUDA_VISIBLE_DEVICES=4 python -m src.main \
    +experiment=replica_new_erp data_loader.train.batch_size=1 \
    model.encoder.shim_patch_size=8 \
    model.encoder.downscale_factor=8 \
    trainer.max_steps=$max_steps \
    output_dir=$output_dir \

    # \
    # checkpointing.load=$resume_path
