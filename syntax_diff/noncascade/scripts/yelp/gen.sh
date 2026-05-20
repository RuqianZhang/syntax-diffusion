cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES=1 python train_noncascade.py --dataset_name yelp \
    --wandb_name yelp_noncascade_gen_0510 --wandb_project yelp_noncascade \
    --eval_test --resume_dir "${SYNTAX_DIFFUSION_CKPT_DIR:-./ckpts}/yelp/yelp-0410" \
    --sampler ddim --sampling_timesteps 50 --num_samples 1000
