cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES=2 python train_noncascade.py --dataset_name emotion --self_condition --scale_shift --wandb_name emotion_nosoftmax_0601 \
    --num_train_steps 100000 --max_seq_len 64 --sampler ddim --sampling_timesteps 250 --num_samples 100 \
    --train_batch_size 128 --eval_batch_size 128 --num_dense_connections 3 --tx_dim 768 --tx_depth 12 \
    --class_conditional --num_classes 6 
