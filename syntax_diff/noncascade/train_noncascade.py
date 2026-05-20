import argparse
from transformers import AutoConfig
import json, os
import torch

from diffusion.noncascade_diffusion_not2t import GaussianDiffusion, Trainer
from model.diffusion_transformer import DiffusionTransformer

ATTN_HEAD_DIM=64

def get_diffusion_lm_dims(args):
    config = AutoConfig.from_pretrained(args.enc_dec_model)
    lm_dim = config.d_model
    return lm_dim

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm_dim = get_diffusion_lm_dims(args)
    
    assert args.tx_dim%ATTN_HEAD_DIM==0, f'Transformer dimension must be divisible by {ATTN_HEAD_DIM}'
    model = DiffusionTransformer(
        tx_dim = args.tx_dim,
        tx_depth = args.tx_depth,
        heads = args.tx_dim//ATTN_HEAD_DIM,
        lm_dim = lm_dim,
        max_seq_len = args.max_seq_len,
        self_condition = args.self_condition,
        scale_shift = args.scale_shift,
        dropout = 0 if args.disable_dropout else 0.1,
        class_conditional=args.class_conditional,
        class_emb_dim=lm_dim,
        num_classes=args.num_classes,
        class_unconditional_prob=args.class_unconditional_prob,
        seq2seq=args.seq2seq,
        context_dim=lm_dim,
        num_class_emb=args.num_class_emb,
        num_dense_connections=args.num_dense_connections,
    ).to(device)

    args.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    diffusion = GaussianDiffusion(
        model,
        max_seq_len = model.max_seq_len,
        sampling_timesteps = args.sampling_timesteps,     # number of sampling steps
        loss_type = args.loss_type,            # L1 or L2
        objective = args.objective,
        train_schedule= args.train_schedule, 
        sampling_schedule = args.sampling_schedule,
        scale = args.scale,
        sampler = args.sampler,
        train_prob_self_cond = args.train_prob_self_cond,
        seq2seq_unconditional_prob = args.seq2seq_unconditional_prob,
    ).to(device)

    trainer = Trainer(
        args=args,
        diffusion=diffusion,
        dataset_name=args.dataset_name,
        train_batch_size = args.train_batch_size,
        eval_batch_size = args.eval_batch_size,
        gradient_accumulate_every = args.gradient_accumulation_steps,
        train_lr = args.learning_rate,
        train_num_steps = args.num_train_steps,
        lr_schedule = args.lr_schedule,
        num_warmup_steps = args.lr_warmup_steps,
        ema_update_every = args.ema_update_every,
        ema_decay = args.ema_decay,
        adam_betas = (args.adam_beta1, args.adam_beta2),
        adam_weight_decay = args.adam_weight_decay,
        save_and_sample_every = args.save_and_sample_every,
        num_samples = args.num_samples,
        results_folder = args.output_dir,
        context_max_seq_len=args.context_max_seq_len,
    )

    if args.eval:
        trainer.load(args.resume_dir)
        if trainer.diffusion.diffusion_model.seq2seq:
            trainer.sample_seq2seq(cls_free_guidance=2.0)
        else:
            if args.class_conditional:
                for class_id in range(model.num_classes):
                    trainer.sample(class_id=class_id)
            else:
                trainer.sample()
        return
        
    if args.eval_test:
        trainer.load(args.resume_dir)
        if trainer.diffusion.diffusion_model.seq2seq:
            trainer.sample_seq2seq(test=True, cls_free_guidance=2.0)
        else:
            if args.class_conditional:
                for class_id in range(model.num_classes):
                    trainer.sample(class_id=class_id, test=True, cls_free_guidance=2.0)
            else:
                trainer.sample(test=True)
        return

    if args.resume_training:
        trainer.load(args.resume_dir)
    if args.init_path:
        trainer.load(args.init_path, init_only=True)

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default='./ckpts')
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    # Optimization hyperparameters
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_train_steps", type=int, default=100000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_schedule", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_update_every", type=int, default=1)
    # Diffusion Hyperparameters
    parser.add_argument("--objective", type=str, default="pred_x0", choices=["pred_noise", "pred_x0", "pred_v",])
    parser.add_argument("--loss_type", type=str, default="l2", choices=["l1", "l2", "smooth_l1"])
    parser.add_argument("--train_schedule", type=str, default="cosine", choices=["beta_linear", "simple_linear", "cosine", 'sigmoid'])
    parser.add_argument("--sampling_schedule", type=str, default=None, choices=["beta_linear", "cosine", "simple_linear", None])
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--sampling_timesteps", type=int, default=2000)
    # Generation Arguments
    parser.add_argument("--save_and_sample_every", type=int, default=10000)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--self_condition", action="store_true", default=False)
    parser.add_argument("--train_prob_self_cond", type=float, default=0.5)
    parser.add_argument("--sampler", type=str, default='ddim', choices=["ddpm", "ddim"])
    # Model hyperparemeters
    parser.add_argument("--enc_dec_model", type=str, default="facebook/bart-base")
    parser.add_argument("--tx_dim", type=int, default=768)
    parser.add_argument("--tx_depth", type=int, default=12)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--scale_shift", action="store_true", default=False)
    parser.add_argument("--num_dense_connections", type=int, default=0)
    parser.add_argument("--disable_dropout", action="store_true", default=False)
    parser.add_argument("--class_conditional", action="store_true", default=False)
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--class_unconditional_prob", type=float, default=.1)
    parser.add_argument("--num_class_emb", type=int, default=4)
    parser.add_argument("--seq2seq", action="store_true", default=False)
    parser.add_argument("--seq2seq_unconditional_prob", type=float, default=0.1)
    parser.add_argument("--context_max_seq_len", type=int, default=12)
    # Load and eval model
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--eval_test", action="store_true", default=False)
    parser.add_argument("--resume_training", action="store_true", default=False)
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument("--init_path", type=str, default=None)

    args = parser.parse_args()
    assert not (args.eval and args.resume_training)
    if args.eval or args.resume_training:
        assert args.resume_dir is not None

    if args.eval or args.resume_training or args.eval_test:
        with open(os.path.join(args.resume_dir, 'args.json'), 'rt') as f:
            saved_args = json.load(f)
        args_dict = vars(args)
        # Hold out sampling/evaluation parameters
        heldout_params = {'wandb_name', 'wandb_project', 'eval_batch_size', 'output_dir', 'resume_dir', 'eval', 'eval_test', 'num_samples', 'sampling_timesteps', 'sampling_schedule', 'scale', 'sampler', 'resume_training'}
        # Overwrite args with saved args
        for k,v in saved_args.items():
            if k in heldout_params:
                continue
            args_dict[k] = v
    main(args)
