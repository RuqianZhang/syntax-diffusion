cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES=1 python train_cascade.py --dataset_name yelp \
    --wandb_name yelp_t2t_cascade_0411 --wandb_project yelp_cascade_t2t \
    --cascade_syntax_path "${SYNTAX_DIFFUSION_CKPT_DIR:-./ckpts}/yelp/syn-0201" \
    --cascade_gen --resume_dir "${SYNTAX_DIFFUSION_CKPT_DIR:-./ckpts}/yelp/syn2text-0203" \
    --sampler ddim --sampling_timesteps 50 --num_samples 1000
