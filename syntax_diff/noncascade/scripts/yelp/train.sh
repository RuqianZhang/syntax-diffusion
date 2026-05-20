cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES=3 python train_noncascade.py --dataset_name yelp --self_condition --scale_shift \
    --wandb_name yelp_noncascade_0505 --wandb_project yelp_noncascade \
    --num_train_steps 250000 --max_seq_len 100 --sampler ddim --sampling_timesteps 250 --num_samples 100 \
    --train_batch_size 128 --eval_batch_size 128 --num_dense_connections 3 --tx_dim 768 --tx_depth 12 \
    --class_conditional --num_classes 3 