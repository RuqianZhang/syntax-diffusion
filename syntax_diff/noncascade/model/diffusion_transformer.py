import math
import torch
from torch import nn
from einops import rearrange, repeat

from model.x_transformer import Encoder

# Helper Functions
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def exists(val):
    return val is not None

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None):
        seq_len = x.shape[1]
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if not exists(pos):
            pos = torch.arange(seq_len, device = x.device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        tx_dim,
        tx_depth,
        heads,
        lm_dim = None,
        max_seq_len = 64,
        self_condition = False,
        scale_shift = False,
        dropout = 0.1,
        class_conditional=False,
        class_emb_dim=0,
        num_classes=0,
        class_unconditional_prob=0,
        seq2seq=False,
        context_dim=0,
        num_class_emb=4,
        num_dense_connections=0,
    ):
        super().__init__()

        self.lm_dim = lm_dim
        self.self_condition = self_condition
        self.scale_shift = scale_shift
        self.max_seq_len = max_seq_len
        self.seq2seq = seq2seq

        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.class_unconditional_prob = class_unconditional_prob
        self.class_emb_dim = class_emb_dim

        sinu_pos_emb = SinusoidalPosEmb(tx_dim)
        fourier_dim = tx_dim
        time_emb_dim = tx_dim*4
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.time_pos_embed_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, tx_dim)
            )
        
        self.pos_emb_s = AbsolutePositionalEmbedding(tx_dim, max_seq_len//2) 
        self.pos_emb_t = AbsolutePositionalEmbedding(tx_dim, max_seq_len//2)        
        
        self.encoder = Encoder(
            dim=tx_dim,
            depth=tx_depth,
            heads=heads,
            attn_dropout = dropout,
            ff_dropout = dropout,
            ff_glu=True,
            time_emb_dim=tx_dim*4 if self.scale_shift else None,
            class_conditional = self.class_conditional,
            num_classes = num_classes,
            num_class_emb = num_class_emb,
            seq2seq=self.seq2seq,
            num_dense_connections=num_dense_connections,
        )

        if self.class_conditional:
            assert num_classes > 0
            self.num_class_embs = (num_classes+1) * num_class_emb
            self.class_cond_embedding = nn.Parameter(torch.randn(self.num_class_embs, class_emb_dim))
            self.codebook = nn.Sequential(
                nn.Linear(class_emb_dim, class_emb_dim * 2),
                nn.ReLU(),
                nn.Linear(class_emb_dim * 2, class_emb_dim * 4),
                nn.ReLU(),
                nn.Linear(class_emb_dim * 4, class_emb_dim * 2),
                nn.ReLU(),
                nn.Linear(class_emb_dim * 2, class_emb_dim)
            )
            self.class_weights = nn.Parameter(torch.randn(num_classes+1, self.num_class_embs))
            self.class_emb_proj = nn.Linear(class_emb_dim, tx_dim)

        if self.seq2seq:
            self.null_embedding_seq2seq = nn.Embedding(1, context_dim)
            self.context_proj = nn.Linear(context_dim, tx_dim)
        
        if self.self_condition:
            self.input_proj_s = nn.Linear(lm_dim*2, tx_dim); self.input_proj_t = nn.Linear(lm_dim*2, tx_dim)
            self.init_self_cond_s = nn.Parameter(torch.randn(1, lm_dim)); self.init_self_cond_t = nn.Parameter(torch.randn(1, lm_dim))
            nn.init.normal_(self.init_self_cond_s, std = 0.02); nn.init.normal_(self.init_self_cond_t, std = 0.02)
        else:
            self.input_proj_s = nn.Linear(lm_dim, tx_dim); self.input_proj_t = nn.Linear(lm_dim, tx_dim)
        self.norm = nn.LayerNorm(tx_dim)
        self.output_proj_s = nn.Linear(tx_dim, lm_dim); self.output_proj_t = nn.Linear(tx_dim, lm_dim)
        init_zero_(self.output_proj_s); init_zero_(self.output_proj_t)


    def forward(self, x, mask, time, x_self_cond = None, class_id = None, context = None, context_mask = None):
        """
        x: input, [batch_size, length, lm_dim]
        mask: bool tensor where False indicates masked positions, [batch, length] 
        time: timestep, [batch_size]
        """
        split = x.shape[1]//2

        time_emb = self.time_mlp(time*1000)
        time_emb = rearrange(time_emb, 'b d -> b 1 d')

        pos_emb_s = self.pos_emb_s(x[:, :split])
        pos_emb_t = self.pos_emb_t(x[:, split:])
        pos_emb = torch.cat([pos_emb_s, pos_emb_t], dim=0)
        
        x_s = x[:, :split]; x_t = x[:, split:]
        if self.self_condition:
            if exists(x_self_cond):
                x_s_self_cond = x_self_cond[:, :split]; x_t_self_cond = x_self_cond[:, split:]
                x_s = torch.cat((x_s, x_s_self_cond), dim=-1); x_t = torch.cat((x_t, x_t_self_cond), dim=-1)
            else:
                repeated_x_s_self_cond = repeat(self.init_self_cond_s, '1 d -> b l d', b=x.shape[0], l=split)
                repeated_x_t_self_cond = repeat(self.init_self_cond_t, '1 d -> b l d', b=x.shape[0], l=split)
                x_s = torch.cat((x_s, repeated_x_s_self_cond), dim=-1); x_t = torch.cat((x_t, repeated_x_t_self_cond), dim=-1)
        x_s_input = self.input_proj_s(x_s); x_t_input = self.input_proj_t(x_t)
        x_input = torch.cat([x_s_input, x_t_input], dim=1)
        tx_input = x_input+pos_emb+self.time_pos_embed_mlp(time_emb)

        if self.seq2seq:
            if context is None:
                context = repeat(self.null_embedding_seq2seq.weight, '1 d -> b 1 d', b=x.shape[0])
                context_mask = torch.tensor([[True] for _ in range(x.shape[0])], dtype=bool, device=x.device)
            context = self.context_proj(context)

        if self.class_conditional:
            codebook_class_emb = self.codebook(self.class_cond_embedding)
            codebook_class_emb = codebook_class_emb / codebook_class_emb.norm(dim=1, keepdim=True)
            class_mask = torch.tensor([[True]*(self.num_class_embs) for _ in range(x.shape[0])], dtype=bool, device=x.device)
            if class_id is None:
                class_id = torch.full((x.shape[0],), self.num_classes, device=x.device)
            else:
                assert exists(class_id)
            class_weights = self.class_weights[class_id] # b,c
            class_weights = torch.softmax(class_weights, dim=-1)

            class_cond = codebook_class_emb.unsqueeze(0) * class_weights.unsqueeze(-1)
            class_emb = self.class_emb_proj(class_cond)
            
        x, attn_maps = self.encoder(tx_input, mask=mask, context=context, context_mask=context_mask, time_emb=time_emb,
                                    class_emb=(class_emb if self.class_conditional else None), class_mask=(class_mask if self.class_conditional else None))
        x_s = self.norm(x[:, :split])
        x_t = self.norm(x[:, split:])
        x = torch.cat([x_s, x_t], dim=1)

        out_s = self.output_proj_s(x[:, :split]); out_t = self.output_proj_t(x[:, split:])
        out = torch.cat([out_s, out_t], dim=1)

        return out, attn_maps