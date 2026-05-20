import math
from pathlib import Path
import random 
from functools import partial
from collections import namedtuple, Counter
import os
import numpy as np
import json
from datetime import timedelta

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from tqdm.auto import tqdm
from ema_pytorch import EMA
from transformers import get_scheduler, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers import T5ForConditionalGeneration



from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
import wandb

import diffusion.constant as constant
import diffusion.optimizer as optimizer
import dataset_utils.text_dataset as text_dataset
from utils.torch_utils import compute_grad_norm
import utils.file_utils as file_utils
from evaluation import evaluation


ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start', 'pred_v'])

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data

def log(t, eps = 1e-12):
    return torch.log(t.clamp(min = eps)) 

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims)) 
    
# noise schedules
def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)

def beta_linear_schedule(t, clip_min = 1e-9):
    return torch.exp(-1e-4 - 10 * (t ** 2)).clamp(min = clip_min, max = 1.)

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = torch.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)

# converting gamma to alpha, sigma or logsnr
def log_snr_to_alpha(log_snr):
    alpha = torch.sigmoid(log_snr)
    return alpha

def alpha_to_shifted_log_snr(alpha, scale = 1):
    return log((alpha / (1 - alpha))).clamp(min=-15, max=15) + 2*np.log(scale).item()

def time_to_alpha(t, alpha_schedule, scale):
    alpha = alpha_schedule(t)
    shifted_log_snr = alpha_to_shifted_log_snr(alpha, scale = scale)
    return log_snr_to_alpha(shifted_log_snr)

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        max_seq_len,
        sampling_timesteps = 250,
        loss_type = 'l1',
        objective = 'pred_noise',
        train_schedule = 'cosine',
        sampling_schedule = None,
        scale = 1.,
        sampler = 'ddim',
        train_prob_self_cond = 0.5,
        seq2seq_unconditional_prob = 0.1,
    ):
        super().__init__()
        assert sampler in {'ddpm', 'ddim'}, 'sampler must be ddpm or ddim'
        self.sampler = sampler

        self.diffusion_model = model
        if self.diffusion_model.class_conditional:
            if self.diffusion_model.class_unconditional_prob > 0: 
                self.class_unconditional_bernoulli = torch.distributions.Bernoulli(probs=self.diffusion_model.class_unconditional_prob)

        self.lm_dim = self.diffusion_model.lm_dim
        self.self_condition = self.diffusion_model.self_condition
        self.max_seq_len = max_seq_len
        assert self.max_seq_len % 2 == 0
        self.objective = objective
        self.loss_type = loss_type
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be one of pred_noise, pred_x0, pred_v'

        if train_schedule == "simple_linear":
            alpha_schedule = simple_linear_schedule
        elif train_schedule == "beta_linear":
            alpha_schedule = beta_linear_schedule
        elif train_schedule == "cosine":
            alpha_schedule = cosine_schedule
        elif train_schedule == "sigmoid":
            alpha_schedule = sigmoid_schedule
        else:
            raise ValueError(f'invalid noise schedule {train_schedule}')
        
        self.train_schedule = partial(time_to_alpha, alpha_schedule=alpha_schedule, scale=scale)

        # Sampling schedule
        if sampling_schedule is None:
            sampling_alpha_schedule = None
        elif sampling_schedule == "simple_linear":
            sampling_alpha_schedule = simple_linear_schedule
        elif sampling_schedule == "beta_linear":
            sampling_alpha_schedule = beta_linear_schedule
        elif sampling_schedule == "cosine":
            sampling_alpha_schedule = cosine_schedule
        elif sampling_schedule == "sigmoid":
            sampling_alpha_schedule = sigmoid_schedule
        else:
            raise ValueError(f'invalid sampling schedule {sampling_schedule}')
        
        if exists(sampling_alpha_schedule):
            self.sampling_schedule = partial(time_to_alpha, alpha_schedule=sampling_alpha_schedule, scale=scale)
        else:
            self.sampling_schedule = self.train_schedule

        self.sampling_timesteps = sampling_timesteps
        # probability for self conditioning during training
        self.train_prob_self_cond = train_prob_self_cond
        self.seq2seq_unconditional_prob = seq2seq_unconditional_prob

    def predict_start_from_noise(self, z_t, t, noise, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        return (z_t - (1-alpha).sqrt() * noise) / alpha.sqrt().clamp(min = 1e-8)
        
    def predict_noise_from_start(self, z_t, t, x0, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        return (z_t - alpha.sqrt() * x0) / (1-alpha).sqrt().clamp(min = 1e-8)

    def predict_start_from_v(self, z_t, t, v, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        x = alpha.sqrt() * z_t - (1-alpha).sqrt() * v

        return x
    
    def predict_noise_from_v(self, z_t, t, v, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        eps = (1-alpha).sqrt() * z_t + alpha.sqrt() * v

        return eps
    
    def predict_v_from_start_and_eps(self, z_t, t, x, noise, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        v = alpha.sqrt() * noise - x* (1-alpha).sqrt()

        return v

    def diffusion_model_predictions(self, z_t, mask, t, *, x_self_cond = None, class_id=None, context=None, context_mask=None,
                                    sampling=False, cls_free_guidance=1.0):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        time_cond = time_to_alpha(t)
        model_output, _ = self.diffusion_model(z_t, mask, time_cond, x_self_cond, class_id=class_id, context=context, context_mask=context_mask)

        if cls_free_guidance!=1.0:
            if exists(class_id):
                unc_class_id = torch.full_like(class_id, fill_value=self.diffusion_model.num_classes)
            else:
                unc_class_id = None
            unc_model_output,_ = self.diffusion_model(z_t, mask, time_cond, x_self_cond, class_id=unc_class_id, context=context, context_mask=context_mask)
            model_output = model_output*cls_free_guidance + unc_model_output*(1-cls_free_guidance)

        pred_v = None
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(z_t, t, pred_noise, sampling=sampling)
        elif self.objective =='pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(z_t, t, x_start, sampling=sampling)
            pred_v = self.predict_v_from_start_and_eps(z_t, t, x_start, pred_noise, sampling=sampling)
        elif self.objective == 'pred_v':
            pred_v = model_output
            x_start = self.predict_start_from_v(z_t, t, pred_v, sampling=sampling)
            pred_noise = self.predict_noise_from_v(z_t, t, pred_v, sampling=sampling)
        else:
            raise ValueError(f'invalid objective {self.objective}')
        
        return ModelPrediction(pred_noise, x_start, pred_v)

    def get_sampling_timesteps(self, batch_size, *, device):
        times = torch.linspace(1., 0., self.sampling_timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch_size) # b*(T+1)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0) # 2*b*T
        times = times.unbind(dim = -1) 
        return times    


    @torch.no_grad()
    def ddpm_sample(self, shape, lengths, class_id, context, context_mask, cls_free_guidance=1.0, z_t=None):
        batch_size, device = shape[0], next(self.diffusion_model.parameters()).device
        time_pairs = self.get_sampling_timesteps(batch_size, device = device)

        if not exists(z_t):
            z_t = torch.randn(shape, device=device)
        x_start = None

        mask = [[True]*length + [False]*(self.max_seq_len//2-length) + [True]*length + [False]*(self.max_seq_len//2-length) for length in lengths]
        mask = torch.tensor(mask, dtype=torch.bool, device=device)
        
        for time, time_next in tqdm(time_pairs, desc = 'sampling step', total = self.sampling_timesteps):
            # get alpha sigma of time and next time
            alpha = self.sampling_schedule(time)
            alpha_next = self.sampling_schedule(time_next)
            alpha, alpha_next = map(partial(right_pad_dims_to, z_t), (alpha, alpha_next))
            alpha_now = alpha/alpha_next

            model_output = self.diffusion_model_predictions(z_t, mask, time, class_id=class_id, x_self_cond=x_start, context=context, context_mask=context_mask, sampling=True, cls_free_guidance=cls_free_guidance)
            x_start = model_output.pred_x_start
            eps = model_output.pred_noise

            if time_next[0] <= 0:
                z_t = x_start
                continue
            noise = torch.randn_like(z_t)
            z_t = 1/alpha_now.sqrt() * (z_t - (1-alpha_now)/(1-alpha).sqrt() * eps) + torch.sqrt(1 - alpha_now) * noise

        return z_t, mask

    @torch.no_grad()
    def ddim_sample(self, shape, lengths, class_id, context, context_mask, cls_free_guidance=1.0, z_t=None):
        batch_size, device = shape[0], next(self.diffusion_model.parameters()).device
        time_pairs = self.get_sampling_timesteps(batch_size, device = device)

        if not exists(z_t):
            z_t = torch.randn(shape, device=device)
        x_start = None

        mask = [[True]*length + [False]*(self.max_seq_len//2-length) + [True]*length + [False]*(self.max_seq_len//2-length) for length in lengths]
        mask = torch.tensor(mask, dtype=torch.bool, device=device)

        for time, time_next in tqdm(time_pairs, desc = 'sampling step', total = self.sampling_timesteps):
            # get alpha sigma of time and next time
            alpha = self.sampling_schedule(time)
            alpha_next = self.sampling_schedule(time_next)
            alpha, alpha_next = map(partial(right_pad_dims_to, z_t), (alpha, alpha_next))

            model_output = self.diffusion_model_predictions(z_t, mask, time, class_id=class_id, x_self_cond=x_start, context=context, context_mask=context_mask, sampling=True, cls_free_guidance=cls_free_guidance)
            x_start = model_output.pred_x_start
            eps = model_output.pred_noise
            if time_next[0] <= 0:
                z_t = x_start
                continue
            z_t = x_start * alpha_next.sqrt() + eps * (1-alpha_next).sqrt()
            
        return z_t, mask

    @torch.no_grad()
    def sample(self, batch_size, length, class_id=None, context=None, context_mask=None, cls_free_guidance=1.0):
        max_seq_len, lm_dim = self.max_seq_len, self.lm_dim
        if self.sampler == 'ddpm':
            sample_fn = self.ddpm_sample
        elif self.sampler == 'ddim':
            sample_fn = self.ddim_sample
        else:
            raise ValueError(f'invalid sampler {self.sampler}')
        return sample_fn((batch_size, max_seq_len, lm_dim), length, class_id, context, context_mask, cls_free_guidance)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def forward(self, z_0, mask, class_id, context, context_mask, *args, **kwargs):
        batch_size, l, d, device, max_seq_len, = *z_0.shape, z_0.device, self.max_seq_len
        assert l == max_seq_len, f'length must be {self.max_seq_len}'
        split_index = l//2
        
        times = torch.zeros((batch_size,), device = device).float().uniform_(1/self.sampling_timesteps, 1.)
        
        # noise sample
        noise = torch.randn_like(z_0)
        alpha = self.train_schedule(times)
        alpha_next = self.sampling_schedule(times - 1/self.sampling_timesteps)
        alpha, alpha_next = map(partial(right_pad_dims_to, z_0), (alpha, alpha_next))
        z_t = alpha.sqrt()*z_0 + (1-alpha).sqrt()*noise

        if self.diffusion_model.class_conditional and self.diffusion_model.class_unconditional_prob > 0:
            assert exists(class_id)
            class_unconditional_mask = self.class_unconditional_bernoulli.sample(class_id.shape).bool()
            class_id[class_unconditional_mask] = self.diffusion_model.num_classes
        
        self_cond = None
        if self.self_condition and (random.random() < self.train_prob_self_cond):
            with torch.no_grad():
                model_output = self.diffusion_model_predictions(z_t, mask, times, class_id=class_id, context=context, context_mask=context_mask)
                self_cond = model_output.pred_x_start.detach()
        
        predictions = self.diffusion_model_predictions(z_t, mask, times, x_self_cond=self_cond, class_id=class_id, context=context, context_mask=context_mask)
        if self.objective == 'pred_x0':
            target = z_0
            pred = predictions.pred_x_start
        elif self.objective == 'pred_noise':
            target = noise
            pred = predictions.pred_noise
        elif self.objective == 'pred_v':
            target = alpha.sqrt() * noise - (1-alpha).sqrt() * z_0
            assert exists(predictions.pred_v)
            pred = predictions.pred_v

        syntax_pred = pred[:, :split_index]; syntax_target = target[:, :split_index]
        text_pred = pred[:, split_index:]; text_target = target[:, split_index:]
        syntax_mask = mask[:, :split_index]; text_mask = mask[:, split_index:]

        syntax_loss = self.loss_fn(syntax_pred, syntax_target, reduction='none')
        text_loss = self.loss_fn(text_pred, text_target, reduction='none')
        syntax_loss = rearrange([reduce(syntax_loss[i][:torch.sum(syntax_mask[i])], 'l d -> 1', 'mean') for i in range(z_0.shape[0])], 'b 1 -> b 1')
        text_loss = rearrange([reduce(text_loss[i][:torch.sum(text_mask[i])], 'l d -> 1', 'mean') for i in range(z_0.shape[0])], 'b 1 -> b 1')
        syntax_loss = syntax_loss.mean()
        text_loss = text_loss.mean()

        # syntax_weight = text_loss.item() / (syntax_loss.item() + text_loss.item() + 1e-8)
        # text_weight = syntax_loss.item() / (syntax_loss.item() + text_loss.item() + 1e-8)
        text_weight = 0.5
        syntax_weight = 0.5

        return text_weight * text_loss + syntax_weight * syntax_loss, text_weight

# trainer class
class Trainer(object):
    def __init__(
        self,
        args,
        diffusion,
        dataset_name,
        *,
        train_batch_size = 64,
        eval_batch_size = 64,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        lr_schedule = 'cosine',
        num_warmup_steps = 500,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        adam_weight_decay = 0.01,
        save_and_sample_every = 5000,
        num_samples = 25,
        results_folder = './results',
        context_max_seq_len = None,
    ):
        super().__init__()
        set_seeds(77)
        self.args = args

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=90))

        if args.wandb_name is None:
            self.accelerator = Accelerator(
                kwargs_handlers=[ddp_kwargs, init_process_kwargs]
            )
        else:
            self.accelerator = Accelerator(
                log_with='wandb',
                kwargs_handlers=[ddp_kwargs, init_process_kwargs]
            )
        self.num_devices = self.accelerator.num_processes
        args.num_devices = self.num_devices

        if self.accelerator.is_main_process:
            if args.output_dir is None:
                args.output_dir = file_utils.get_output_dir(args)
                with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
                    json.dump(args.__dict__, f, indent=2)
            results_folder = args.output_dir
            run = args.wandb_project if args.wandb_project else os.path.split(__file__)[-1].split(".")[0]
            if args.wandb_name:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder, "name": args.wandb_name}})
            else:
                self.accelerator.init_trackers(run, config=args)

        self.diffusion = diffusion
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.max_seq_len = diffusion.max_seq_len
        self.enc_dec_model = args.enc_dec_model
        self.context_max_seq_len = context_max_seq_len
        self.split_index = diffusion.max_seq_len//2

        # Init Encoder-decoder model
        if 'bart' in args.enc_dec_model:
            self.bart_model = BartForConditionalGeneration.from_pretrained(args.enc_dec_model)
        elif 'flan-t5' in args.enc_dec_model:
            self.bart_model = T5ForConditionalGeneration.from_pretrained(args.enc_dec_model)
        elif 'long-t5' in args.enc_dec_model:
            self.bart_model = AutoModelForSeq2SeqLM.from_pretrained(args.enc_dec_model)
        else:
            raise ValueError(f'invalid enc_dec_model {args.enc_dec_model}')
        
        self.syntax_tokenizer = text_dataset.create_tokenizer(path=text_dataset.get_data_root())
        self.text_tokenizer = AutoTokenizer.from_pretrained(args.enc_dec_model)
        self.class_conditional = self.diffusion.diffusion_model.class_conditional
        self.seq2seq = self.diffusion.diffusion_model.seq2seq
        self.seq2seq_unconditional_prob = self.diffusion.seq2seq_unconditional_prob
        self.bart_model.eval()
        
        # optimizer
        self.opt = optimizer.get_adamw_optimizer(diffusion.parameters(), lr = train_lr, betas = adam_betas, weight_decay=adam_weight_decay)
        # scheduler
        lr_scheduler = get_scheduler(
            lr_schedule,
            optimizer=self.opt,
            num_warmup_steps=num_warmup_steps*self.num_devices,
            num_training_steps=train_num_steps*self.num_devices,
        )
        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion, beta = ema_decay, update_every = ema_update_every, power=3/4)
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)
        # step counter state
        self.step = 0

        # prepare model, optimizer with accelerator
        self.diffusion, self.bart_model, self.opt, self.lr_scheduler = self.accelerator.prepare(self.diffusion, self.bart_model, self.opt, lr_scheduler)
        for param in self.bart_model.parameters():
                param.requires_grad = False
        self.encoder = self.bart_model.get_encoder()
        for param in self.encoder.parameters():
                param.requires_grad = False

        # dataset and dataloader
        self.dataset_name = dataset_name
        dataset = text_dataset.get_dataset(dataset_name)
        self.dataset = dataset.shuffle(seed=77)
        if args.eval_test:
            self.num_samples = min(self.num_samples,len(self.dataset['test']))
            print(f'Using {self.num_samples} samples for test')
        else:
            self.num_samples = min(self.num_samples,len(self.dataset['valid']))
            print(f'Using {self.num_samples} samples for evaluation')
        self.dataloader = text_dataset.get_dataloader(args, self.dataset['train'], self.bart_model.config, self.text_tokenizer, self.syntax_tokenizer, self.max_seq_len,
                                                         seq2seq=self.seq2seq, context_max_seq_len=self.context_max_seq_len)
        self.val_dataloader = text_dataset.get_dataloader(args, self.dataset['valid'], self.bart_model.config, self.text_tokenizer, self.syntax_tokenizer, self.max_seq_len,
                                                             seq2seq=self.seq2seq, context_max_seq_len=self.context_max_seq_len)
        self.test_dataloader = text_dataset.get_dataloader(args, self.dataset['test'], self.bart_model.config, self.text_tokenizer, self.syntax_tokenizer, self.max_seq_len,
                                                         seq2seq=self.seq2seq, context_max_seq_len=self.context_max_seq_len)
        # half the length of input
        training_lengths = [min(sum(self.dataloader.dataset[idx]['attention_mask'])//2, self.max_seq_len//2) for idx in range(self.dataloader.dataset.num_rows)]
        length_counts = Counter(training_lengths)
        probs = torch.tensor([length_counts[idx]/self.dataloader.dataset.num_rows for idx in range(self.max_seq_len+1)])
        assert probs[0] == 0, 'Can\'t have examples of length 0'
        self.length_categorical = torch.distributions.Categorical(probs=probs)

        if self.class_conditional:
            training_labels = [self.dataloader.dataset[idx]['label'] for idx in range(self.dataloader.dataset.num_rows)]
            label_counts = Counter(training_labels)
            probs = torch.tensor([label_counts[idx]/self.dataloader.dataset.num_rows for idx in range(self.diffusion.diffusion_model.num_classes)])
            self.class_categorical = torch.distributions.Categorical(probs=probs)

        self.dataloader = self.accelerator.prepare(self.dataloader)
        self.data_iter = cycle(self.dataloader)
        self.val_iter = cycle(self.val_dataloader)

    def save(self, best=False):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.diffusion),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'scheduler': self.lr_scheduler.state_dict(),
        }
        if best:
            torch.save(data, str(self.results_folder / f'best_model.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model.pt'))

    def load(self, file_path=None, best=False):
        file_path = Path(file_path) if exists(file_path) else self.results_folder
        accelerator = self.accelerator
        device = accelerator.device
        if best:
            data = torch.load(str(file_path / f'best_model.pt'), map_location=device)
        else:
            data = torch.load(str(file_path / f'model.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.diffusion)
        # For backwards compatibility with earlier models
        model.load_state_dict(data['model'])
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_local_main_process:
            self.ema.load_state_dict(data['ema'])
        self.step = data['step']
        if 'scheduler' in data:
            self.lr_scheduler.load_state_dict(data['scheduler'])
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    @torch.no_grad()
    def sample(self, num_samples=None, class_id=None, seed=77, test=False, cls_free_guidance=1.0):
        num_samples = default(num_samples, self.num_samples)
        accelerator = self.accelerator
        device = accelerator.device
        self.ema.ema_model.eval()
        torch.manual_seed(seed) 
        torch.cuda.empty_cache()

        prefix = ''
        if device.type == 'mps':
            constant.generate_kwargs['beam']['num_beams']=1
        kwargs = constant.generate_kwargs['beam']

        def get_class_id(n):
            if exists(class_id):
                return torch.tensor([class_id]*n, dtype=torch.long, device=device)
            if self.class_conditional:
                 return self.class_categorical.sample((n,)).to(device)
            return None
        
        ref_labels = [] if self.class_conditional else None
        pred_syntax = []
        pred_texts = []

        # Extract references
        if exists(class_id):
            prefix += f'class{class_id}'
            class_subset = self.dataset.filter(lambda x: x['label'] == class_id)
            if test:
                ref_texts = class_subset['test']['text']
            else:
                ref_texts = class_subset['valid']['text']
        elif test:
            ref_texts= self.dataset['test']['text']
        else:
            ref_texts= self.dataset['valid']['text']
        num_samples = min(num_samples, len(ref_texts))
        ref_texts = ref_texts[:num_samples]
        batch_size = min(num_samples, self.eval_batch_size)

        while len(pred_texts) < num_samples:
            gen_class_id = get_class_id(batch_size)
            if self.class_conditional:
                ref_labels.extend([tensor.item() for tensor in gen_class_id])

            pred_z_0, mask = self.ema.ema_model.sample(batch_size=batch_size, length=self.length_categorical.sample((batch_size,)), class_id=gen_class_id, cls_free_guidance=cls_free_guidance)
            pred_z_0, mask = pred_z_0.to(device), mask.to(device)
            # decode syntax
            encoder_output = BaseModelOutput(last_hidden_state=pred_z_0[:,:self.split_index].clone())
            syntax_ids = self.bart_model.generate(encoder_outputs=encoder_output, attention_mask=mask[:,:self.split_index].clone(), **kwargs)
            # print(syntax_ids)
            gen_syntax = [self.syntax_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in syntax_ids]
            print(gen_syntax)
            pred_syntax.extend(gen_syntax)
            # decode text
            encoder_output = BaseModelOutput(last_hidden_state=pred_z_0[:,self.split_index:].clone())
            text_ids = self.bart_model.generate(encoder_outputs=encoder_output, attention_mask=mask[:,self.split_index:].clone(), **kwargs)
            # print(text_ids)
            gen_texts = [self.text_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in text_ids]
            print(gen_texts)
            pred_texts.extend(gen_texts)

        assert len(pred_texts) == len(pred_syntax)
        assert len(pred_texts) >= num_samples
        pred_texts = pred_texts[:num_samples]
        pred_syntax = pred_syntax[:num_samples]
        if self.class_conditional:
            ref_labels = ref_labels[:num_samples]

        path = os.path.join(self.results_folder, f'{"class_" if self.class_conditional else ""}_sample.txt')
        file_utils.save_samples(path, pred_texts, pred_syntax, class_id=ref_labels)

        # Log samples: syntax | text
        data = []
        if self.class_conditional:
            columns = ['class', 'syntax', 'text']
        else: 
            columns = ['syntax', 'text']
        for i in range(len(pred_texts)):
            if self.class_conditional:
                row = [ref_labels[i], pred_syntax[i], pred_texts[i]]
            else:
                row = [pred_syntax[i], pred_texts[i]]
            data.append(row)
        table = wandb.Table(columns=columns, data=data)
        accelerator.log({f"sample/{prefix}_samples": table})
        
        # Compute metrics
        metrics = {}
        metrics[f"sample/{prefix}perplexity"] = evaluation.compute_perplexity(pred_texts)
        metrics[f"sample/{prefix}unique_wordcount"] = evaluation.compute_wordcount(pred_texts)
        ngram_metrics = evaluation.compute_diversity(pred_texts)
        for k, v in ngram_metrics.items():
            metrics[f"sample/{prefix}{k}"] = v
        if metrics[f"sample/{prefix}perplexity"] <= 5000:
            for mauve_model_id in ["gpt2-large"]:
                metrics[f"sample/{prefix}mauve"], _ = evaluation.compute_mauve(pred_texts, ref_texts, mauve_model_id)
        if self.class_conditional:
            metrics[f"sample/{prefix}accuracy"] = evaluation.compute_classifier(pred_texts, ref_labels, self.dataset_name)
        # text overlap
        ngram_overlap_metrics = evaluation.compute_corpus_ngram_overlap(pred_texts, ref_texts)
        for k, v in ngram_overlap_metrics.items():
            metrics[f"sample/{prefix}{k}"] = v
        # syntax overlap
        ngram_syntax_overlap_metrics = evaluation.compute_corpus_ngram_syntax_overlap(pred_texts, ref_texts)
        for k, v in ngram_syntax_overlap_metrics.items():
            metrics[f"sample/{prefix}{k}"] = v
        # syntax generated vs text generated
        ngram_syntax_overlap_sentence_metrics = evaluation.compute_sentence_ngram_syntax_overlap(pred_texts, pred_syntax)
        for k, v in ngram_syntax_overlap_sentence_metrics.items():
            metrics[f"sample/{prefix}{k}"] = v
        accelerator.log(metrics)
        print(metrics)
        
        torch.cuda.empty_cache()

    @torch.no_grad()
    def sample_seq2seq(self, num_samples=None, class_id=None, seed=77, test=False, cls_free_guidance=1.0):
        num_samples = default(num_samples, self.num_samples)
        accelerator = self.accelerator
        device = accelerator.device
        self.ema.ema_model.eval()
        torch.manual_seed(seed) 
        torch.cuda.empty_cache()

        prefix = ''
        if device.type == 'mps':
            constant.generate_kwargs['beam']['num_beams']=1
        kwargs = constant.generate_kwargs['beam']
        prefix += f'seq2seq{cls_free_guidance}/' if cls_free_guidance != 1.0 else ''

        def get_class_id(n):
            if exists(class_id):
                return torch.tensor([class_id]*n, dtype=torch.long, device=device)
            if self.class_conditional:
                 return self.class_categorical.sample((n,)).to(device)
            return None
        
        ref_texts = []
        ref_labels = [] if self.class_conditional else None
        cond_texts = []
        pred_syntax = []
        pred_texts = []

        if test:
            if exists(class_id):
                prefix += f'class{class_id}'
                class_subset = self.dataset['test'].filter(lambda x: x['label'] == class_id)
                num_samples = min(num_samples, len(class_subset))
                batch_size = min(num_samples, self.eval_batch_size)
                dataloader = text_dataset.get_class_dataloader(self.args, class_subset, self.bart_model.config, self.text_tokenizer, self.syntax_tokenizer, self.max_seq_len,
                                                         seq2seq=self.seq2seq, context_max_seq_len=self.context_max_seq_len, batch_size=batch_size)
            else:
                dataloader = self.test_dataloader
        else:
            dataloader = self.val_dataloader

        for batch in dataloader:
            data = batch.to(device)
            context = self.encoder(input_ids = data['context_input_ids'], attention_mask = data['context_attention_mask']).last_hidden_state.float()
            context_mask = data['context_attention_mask'].bool()
            batch_size = context.shape[0]
            gen_class_id = get_class_id(batch_size)
            if self.class_conditional:
                ref_labels.extend([tensor.item() for tensor in gen_class_id])
            cond_texts.extend([self.text_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in data['context_input_ids']])
            ref_text_ids = data['input_ids'][:,self.split_index:]
            ref_texts.extend([self.text_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in ref_text_ids])

            pred_z_0, mask = self.ema.ema_model.sample(batch_size=batch_size, length=self.length_categorical.sample((batch_size,)), class_id=gen_class_id, context=context, context_mask=context_mask, cls_free_guidance=cls_free_guidance)
            pred_z_0, mask = pred_z_0.to(device), mask.to(device)

            # decode syntax
            encoder_output = BaseModelOutput(last_hidden_state=pred_z_0[:,:self.split_index].clone())
            syntax_ids = self.bart_model.generate(encoder_outputs=encoder_output, attention_mask=mask[:,:self.split_index].clone(), **kwargs)
            gen_syntax = [self.syntax_tokenizer.decode(g, skip_special_tokens=False, clean_up_tokenization_spaces=True) for g in syntax_ids]
            print(gen_syntax)
            pred_syntax.extend(gen_syntax)
            # decode text
            encoder_output = BaseModelOutput(last_hidden_state=pred_z_0[:,self.split_index:].clone())
            text_ids = self.bart_model.generate(encoder_outputs=encoder_output, attention_mask=mask[:,self.split_index:].clone(), **kwargs)
            gen_texts = [self.text_tokenizer.decode(g, skip_special_tokens=False, clean_up_tokenization_spaces=True).strip() for g in text_ids]
            print(gen_texts)
            pred_texts.extend(gen_texts)

            if len(pred_texts) >= num_samples:
                break

        assert len(pred_texts) == len(ref_texts) == len(pred_syntax) == len(cond_texts)
        assert len(pred_texts) >= num_samples
        pred_texts = pred_texts[:num_samples]
        ref_texts = ref_texts[:num_samples]
        pred_syntax = pred_syntax[:num_samples]
        cond_texts = cond_texts[:num_samples]
        if self.class_conditional:
            ref_labels = ref_labels[:num_samples]

        path = os.path.join(self.results_folder, f'{"class_" if self.class_conditional else ""}seq2seq_sample.txt')
        file_utils.save_samples(path, pred_texts, pred_syntax, ref_text=ref_texts, cond_text=cond_texts, class_id=ref_labels)
        # Log samples: context | syntax | text | reference
        data = []
        if self.class_conditional:
            columns = ['class', 'context', 'syntax', 'text', 'ref']
        else: 
            columns = ['context', 'syntax', 'text', 'ref']
        for i in range(len(pred_texts)):
            if self.class_conditional:
                row = [ref_labels[i], cond_texts[i], pred_syntax[i], pred_texts[i], ref_texts[i]]
            else:
                row = [cond_texts[i], pred_syntax[i], pred_texts[i], ref_texts[i]]
            data.append(row)
        table = wandb.Table(columns=columns, data=data)
        accelerator.log({f"seq2seq/{prefix}_samples": table})

        # Compute metrics
        metrics = {}
        # rouge_metrics = evaluation.compute_rouge(pred_texts, ref_texts)
        # for k, v in rouge_metrics.items():
        #     metrics[f"seq2seq/{prefix}{k}"] = v
        metrics[f"seq2seq/{prefix}perplexity"] = evaluation.compute_perplexity(pred_texts)
        metrics[f"seq2seq/{prefix}unique_wordcount"] = evaluation.compute_wordcount(pred_texts)
        ngram_metrics = evaluation.compute_diversity(pred_texts)
        for k, v in ngram_metrics.items():
            metrics[f"seq2seq/{prefix}{k}"] = v
        metrics[f"seq2seq/{prefix}bertscore"] = evaluation.compute_bertscore(pred_texts, ref_texts)
        # metrics[f'seq2seq/{prefix}bleu'] = evaluation.compute_bleu(pred_texts, ref_texts)
        # Only evaluate MAUVE if generations are reasonable to speed up validation early on
        if metrics[f"seq2seq/{prefix}perplexity"] <= 5000:
            for mauve_model_id in ["gpt2-large"]:
                metrics[f"seq2seq/{prefix}mauve"], _ = evaluation.compute_mauve(pred_texts, ref_texts, mauve_model_id)
        if self.class_conditional:
            metrics[f"seq2seq/{prefix}accuracy"] = evaluation.compute_classifier(pred_texts, ref_labels, self.dataset_name)
        ngram_overlap_metrics = evaluation.compute_corpus_ngram_overlap(pred_texts, ref_texts)
        for k, v in ngram_overlap_metrics.items():
            metrics[f"seq2seq/{prefix}{k}"] = v
        accelerator.log(metrics)
        print(metrics)
        torch.cuda.empty_cache()

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.
                # avg_text_weight = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.data_iter).to(device)
                    with torch.no_grad():
                        encoder_outputs = self.encoder(input_ids = data['input_ids'][:,:self.split_index], attention_mask = data['attention_mask'][:,:self.split_index])                  
                        syntax_z_0 = encoder_outputs.last_hidden_state
                        encoder_outputs = self.encoder(input_ids = data['input_ids'][:,self.split_index:], attention_mask = data['attention_mask'][:,self.split_index:])                  
                        text_z_0 = encoder_outputs.last_hidden_state
                        z_0 = torch.cat([syntax_z_0, text_z_0], dim=1)
                    
                    context = None
                    context_mask = None
                    with accelerator.autocast():
                        if self.seq2seq and random.random() < (1-self.seq2seq_unconditional_prob):
                            context = self.encoder(input_ids = data['context_input_ids'], attention_mask = data['context_attention_mask']).last_hidden_state.float()
                            context_mask = data['context_attention_mask'].bool()

                    mask = data['attention_mask'].bool()
                    loss, _ = self.diffusion(z_0, mask, class_id=(data['label'] if self.class_conditional else None), context=context, context_mask=context_mask)
                    loss = loss / self.gradient_accumulate_every
                    # text_weight = text_weight / self.gradient_accumulate_every
                    total_loss += loss.item()
                    # avg_text_weight += text_weight
                    # print(avg_text_weight)
                    self.accelerator.backward(loss)                

                accelerator.clip_grad_norm_(self.diffusion.parameters(), self.args.clip_grad_norm)
                grad_norm = compute_grad_norm(self.diffusion.parameters())
                accelerator.wait_for_everyone()
                self.opt.step()
                self.lr_scheduler.step()
                self.opt.zero_grad()
                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    logs = {
                        "loss": total_loss,
                        # "avg_text_weight": avg_text_weight,
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm,
                        "step": self.step, 
                        "epoch": (self.step*self.gradient_accumulate_every)/len(self.dataloader), 
                        "samples": self.step*self.train_batch_size*self.gradient_accumulate_every*self.num_devices
                    }
                    self.ema.to(device)
                    self.ema.update()

                    # Log to WandB
                    if self.step % 50 == 0:
                        self.diffusion.eval()
                        self.ema.ema_model.eval()
                        with torch.no_grad():
                            total_val_loss = 0.
                            total_val_ema_loss = 0.
                            for _ in range(self.gradient_accumulate_every):
                                data = next(self.val_iter).to(device)
                                encoder_outputs = self.encoder(input_ids = data['input_ids'][:,:self.split_index], attention_mask = data['attention_mask'][:,:self.split_index])                  
                                syntax_z_0 = encoder_outputs.last_hidden_state
                                encoder_outputs = self.encoder(input_ids = data['input_ids'][:,self.split_index:], attention_mask = data['attention_mask'][:,self.split_index:])                  
                                text_z_0 = encoder_outputs.last_hidden_state
                                z_0 = torch.cat([syntax_z_0, text_z_0], dim=1)
                                
                                context = None
                                context_mask = None
                                if self.seq2seq and random.random() < (1-self.seq2seq_unconditional_prob):
                                    context = self.encoder(input_ids = data['context_input_ids'], attention_mask = data['context_attention_mask']).last_hidden_state.float()
                                    context_mask = data['context_attention_mask'].bool()
                                
                                mask = data['attention_mask'].bool()
                                loss, _ = self.diffusion(z_0, mask, class_id=(data['label'] if self.class_conditional else None), context=context, context_mask=context_mask)
                                loss = loss / self.gradient_accumulate_every
                                total_val_loss += loss.item()
                                loss, _ = self.ema.ema_model(z_0, mask, class_id=(data['label'] if self.class_conditional else None), context=context, context_mask=context_mask)
                                loss = loss / self.gradient_accumulate_every
                                total_val_ema_loss += loss.item()

                            logs["val_loss"] = total_val_loss
                            logs["val_ema_loss"] = total_val_ema_loss
                            pbar.set_postfix(**logs)  
                        self.diffusion.train()
                    accelerator.log(logs, step=self.step)              
                    if self.step % self.save_and_sample_every == 0:
                        if self.seq2seq:
                            self.sample_seq2seq(num_samples=100)
                        else: 
                            self.sample(num_samples=100)
                        self.save()
                        self.diffusion.train() 
                pbar.update(1)
            accelerator.wait_for_everyone()
        self.save()
        accelerator.print('training complete')
