export CUDA_VISIBLE_DEVICES=7

# python -m src.scripts.generate_evaluation_index +evaluation=hm3d \
#     dataset.name="hm3d" \
#     dataset.roots=["/wudang_vuc_3dc_afs/chenzheng/hm3d_dataset_pt"] \
#     +dataset.rgb_roots=["/wudang_vuc_3dc_afs/chenzheng/hm3d_dataset"] \
#     index_generator.output_path="outputs/evaluation_index_hm3d" \
#     dataset.test_len=500

python -m src.scripts.generate_video_evaluation_index \
    --index_input ./assets/evaluation_index_hm3d.json \
    --index_output ./assets/evaluation_index_hm3d_video.json

python -m src.scripts.generate_video_evaluation_index \
    --index_input ./assets/evaluation_index_replica.json \
    --index_output ./assets/evaluation_index_replica_video.json
