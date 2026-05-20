cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES=2 python train_noncascade.py --dataset_name emotion --wandb_name emotion_gen_0604 \
    --eval_test --resume_dir "${SYNTAX_DIFFUSION_CKPT_DIR:-./ckpts}/emotion/emotion-0601" \
    --sampler ddim --sampling_timesteps 50 --num_samples 1000
