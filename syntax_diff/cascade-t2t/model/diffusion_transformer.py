import math
import torch
from torch import nn
from einops import rearrange, repeat

from model.x_transformer import AbsolutePositionalEmbedding, Encoder

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

class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        tx_dim, # transformer的维数
        tx_depth,
        heads,
        lm_dim = None,
        max_seq_len = 64,
        self_condition = False,
        dropout = 0.1,
        scale_shift = False,
        class_conditional=False,
        class_emb_dim=0,
        num_classes=0,
        class_unconditional_prob=0,
        context_dim=0,
        cascade=False,
        cascade_cond_dim=0,
        num_class_emb=4,
        num_dense_connections=0,
    ):
        super().__init__()

        self.lm_dim = lm_dim
        self.self_condition = self_condition
        self.scale_shift = scale_shift
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.class_unconditional_prob = class_unconditional_prob
        self.max_seq_len = max_seq_len
        self.cascade = cascade
        self.tx_depth = tx_depth
        self.class_emb_dim = class_emb_dim

        # time embeddings
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
        # position embedding
        self.pos_emb = AbsolutePositionalEmbedding(tx_dim, max_seq_len)

        # 修改：cascade_time embeddings
        if self.cascade:
            self.cascade_time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, time_emb_dim),
                nn.GELU(),
                nn.Linear(time_emb_dim, time_emb_dim)
            )
        
        # encoder
        self.encoder = Encoder(
            dim=tx_dim,
            depth=tx_depth,
            heads=heads,
            attn_dropout = dropout,    # dropout post-attention
            ff_dropout = dropout,       # feedforward dropout
            ff_glu=True,
            cascade=self.cascade,
            class_conditional = self.class_conditional,
            time_emb_dim=tx_dim*4 if self.scale_shift else None,
            num_classes = num_classes,
            num_class_emb = num_class_emb,
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
        
        self.context_proj = nn.Linear(context_dim, tx_dim)
        
        # 在没有上下文的情况下生成一个默认的嵌入向量
        if self.cascade:
            self.null_embedding_cascade = nn.Embedding(1, cascade_cond_dim)
            self.cascade_proj = nn.Linear(cascade_cond_dim, tx_dim)
        
        if self.self_condition:
            # 将输入和自条件拼接在一起
            self.input_proj = nn.Linear(lm_dim*2, tx_dim)
            self.init_self_cond = nn.Parameter(torch.randn(1, lm_dim))
            nn.init.normal_(self.init_self_cond, std = 0.02)
        else:
            self.input_proj = nn.Linear(lm_dim, tx_dim)
        self.norm = nn.LayerNorm(tx_dim)
        self.output_proj = nn.Linear(tx_dim, lm_dim)
        init_zero_(self.output_proj)


    def forward(self, x, mask, time, x_self_cond = None, class_id = None, context = None, context_mask = None,
                cascade_cond = None, cascade_mask = None, cascade_time = None):
        """
        x: input, [batch_size, length, lm_dim]
        mask: bool tensor where False indicates masked positions, [batch, length] 
        time: timestep, [batch]
        """

        time_emb = self.time_mlp(time*1000)
        time_emb = rearrange(time_emb, 'b d -> b 1 d')

        # TODO
        if self.cascade:
            if cascade_time is None:
                cascade_time = torch.zeros_like(time)
            cascade_time_emb = self.cascade_time_mlp(cascade_time*1000)
            cascade_time_emb = rearrange(cascade_time_emb, 'b d -> b 1 d')
            time_emb = time_emb + cascade_time_emb

        pos_emb = self.pos_emb(x)
        
        if self.self_condition:
            if exists(x_self_cond):
                x = torch.cat((x, x_self_cond), dim=-1)
            else:
                repeated_x_self_cond = repeat(self.init_self_cond, '1 d -> b l d', b=x.shape[0], l=x.shape[1])
                x = torch.cat((x, repeated_x_self_cond), dim=-1)
        x_input = self.input_proj(x)
        tx_input = x_input+pos_emb+self.time_pos_embed_mlp(time_emb)

        context = self.context_proj(context)
        
        if self.cascade:
            if cascade_cond is None:
                cascade_cond = repeat(self.null_embedding_cascade.weight, '1 d -> b 1 d', b=x.shape[0])
                cascade_mask = torch.tensor([[True] for _ in range(x.shape[0])], dtype=bool, device=x.device)
            cascade_emb = self.cascade_proj(cascade_cond)

        if self.class_conditional:
            codebook_class_emb = self.codebook(self.class_cond_embedding)
            codebook_class_emb = codebook_class_emb / codebook_class_emb.norm(dim=1, keepdim=True)
            class_mask = torch.tensor([[True]*(self.num_class_embs) for _ in range(x.shape[0])], dtype=bool, device=x.device)
            if class_id is None:
                class_id = torch.full((x.shape[0],), self.num_classes, device=x.device)
            else:
                assert exists(class_id)
            class_weights = self.class_weights[class_id] # b,c
            # softmax
            class_weights = torch.softmax(class_weights, dim=-1)

            class_cond = codebook_class_emb.unsqueeze(0) * class_weights.unsqueeze(-1)
            class_emb = self.class_emb_proj(class_cond)
            
        x, attn_maps = self.encoder(tx_input, mask=mask, context=context, context_mask=context_mask, time_emb=time_emb,
                                    cascade_emb=(cascade_emb if self.cascade else None), cascade_mask=(cascade_mask if self.cascade else None),
                                    class_emb=(class_emb if self.class_conditional else None), class_mask=(class_mask if self.class_conditional else None))
        x = self.norm(x)

        return self.output_proj(x), attn_maps