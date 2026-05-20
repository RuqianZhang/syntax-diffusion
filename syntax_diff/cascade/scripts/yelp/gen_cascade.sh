cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES=2 python train_cascade.py --dataset_name yelp --wandb_name yelp_gen_0709\
    --cascade_syntax_path "${SYNTAX_DIFFUSION_CKPT_DIR:-./ckpts}/yelp/yelp-syn-0402" \
    --cascade_gen --resume_dir "${SYNTAX_DIFFUSION_CKPT_DIR:-./ckpts}/yelp/yelp-syn2text-0506" \
    --sampler ddim --sampling_timesteps 50 --num_samples 1000
