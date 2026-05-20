cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES=2 python train_cascade.py --dataset_name yelp --self_condition --scale_shift --wandb_name yelp_t2t_syn_0201 \
    --num_train_steps 250000 --max_seq_len 50 --class_conditional --num_classes 3 --sampler ddim --sampling_timesteps 250 --num_samples 100 \
    --train_batch_size 128 --eval_batch_size 128 --num_dense_connections 3 --tx_dim 768 --tx_depth 12 \
    --context_max_seq_len 16
