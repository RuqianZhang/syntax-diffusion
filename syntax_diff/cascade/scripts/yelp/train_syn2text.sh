cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES=3 python train_cascade.py --dataset_name yelp --self_condition --scale_shift --wandb_name yelp_syn2text_0507 \
    --num_train_steps 250000 --cascade --max_seq_len 50 --class_conditional --num_classes 3 --sampler ddim --num_samples 100 --sampling_timesteps 250 \
    --cascade_syntax_path "${SYNTAX_DIFFUSION_CKPT_DIR:-./ckpts}/yelp/yelp-syn-0402" \
    --train_batch_size 128 --eval_batch_size 128 --num_dense_connections 3 --tx_dim 768 --tx_depth 12
