import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial, wraps
from inspect import isfunction
from collections import namedtuple
from einops import rearrange

# constants
DEFAULT_DIM_HEAD = 64

AttentionMaps = namedtuple('AttentionMaps', [
    'pre_softmax_attn',
    'post_softmax_attn'
])

# helpers
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

class equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x == self.val

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(t, groups = 1):
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = -1)
    return rearrange(t, '... g d -> ... (g d)')

# init helpers
def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

# keyword argument helpers
def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, l2norm_embed = False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None):
        seq_len = x.shape[1]
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if not exists(pos):
            pos = torch.arange(seq_len, device = x.device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb

# residual and residual gates
class Residual(nn.Module):
    def __init__(self, dim, scale_residual = False, scale_residual_constant = 1.):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None
        self.scale_residual_constant = scale_residual_constant

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant

        return x + residual

class TimeConditionedResidual(nn.Module):
    def __init__(self, time_emb_dim, dim_out):
        super().__init__()
        self.scale_shift = ScaleShift(time_emb_dim, dim_out)

    def forward(self, x, residual, time_emb):
        return self.scale_shift(x, time_emb) + residual

# token shifting
def shift(t, amount, mask = None):
    if amount == 0:
        return t
    else:
        amount = min(amount, t.shape[1])

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return F.pad(t, (0, 0, amount, -amount), value = 0.)

class ShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask = mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)

# feedforward
class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * self.act(gate)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        glu = False,
        dropout = 0.,
        no_bias = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        activation = nn.GELU()

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim, bias = not no_bias),
            activation
        ) if not glu else GLU(dim, inner_dim, activation)

        self.ff = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias = not no_bias)
        )

    def forward(self, x):
        return self.ff(x)

# attention
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = DEFAULT_DIM_HEAD,
        heads = 8,
        dropout = 0.,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads

        q_dim = k_dim = dim_head * heads
        v_dim = out_dim = dim_head * heads

        self.to_q = nn.Linear(dim, q_dim, bias = False)
        self.to_k = nn.Linear(dim, k_dim, bias = False)
        self.to_v = nn.Linear(dim, v_dim, bias = False)
        self.dropout = nn.Dropout(dropout)
        # attention softmax function
        self.attn_fn = partial(F.softmax, dtype = torch.float32)
        self.to_out = nn.Linear(out_dim, dim, bias = False)

    def forward(
        self,
        x,
        cond = None,
        mask = None,
        cond_mask = None,
    ):
        b, n, _, h, scale, device = *x.shape, self.heads, self.scale, x.device
        kv_input = default(cond, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        k, v= map(lambda t: maybe(rearrange)(t, 'b n (h d) -> b h n d', h = h), (k, v))

        input_mask = None
        if any(map(exists, (mask, cond_mask))):
            q_mask = default(mask, lambda: torch.ones((b, n), device = device).bool())
            k_mask = q_mask if not exists(cond) else cond_mask
            k_mask = default(k_mask, lambda: torch.ones((b, k.shape[-2]), device = device).bool())
            q_mask = rearrange(q_mask, 'b i -> b 1 i 1')
            k_mask = rearrange(k_mask, 'b j -> b 1 1 j')
            input_mask = q_mask * k_mask

        # Attention map
        kv_einsum_eq = 'b h j d'
        dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale
        mask_value = max_neg_value(dots)
        pre_softmax_attn = dots.clone()

        if exists(input_mask):
            dots.masked_fill_(~input_mask, mask_value)
            del input_mask
        dtype = dots.dtype

        attn = self.attn_fn(dots, dim = -1)
        attn = attn.type(dtype)
        
        post_softmax_attn = attn.clone()

        attn = self.dropout(attn)
        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        attn_maps = AttentionMaps(
            pre_softmax_attn = pre_softmax_attn,
            post_softmax_attn = post_softmax_attn
        )

        return self.to_out(out), attn_maps

class ScaleShift(nn.Module):
    def __init__(self, time_emb_dim, dim_out):
        super().__init__()
        self.time_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, dim_out * 2)
            )
        init_zero_(self.time_mlp[-1])
        

    def forward(self, x, time_emb):
        scale, shift = self.time_mlp(time_emb).chunk(2, dim = 2)
        x = x * (scale + 1) + shift

        return x

class AttentionLayers(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads = 8,
        cascade = False,
        class_conditional = False,
        scale_residual = False,
        scale_residual_constant = 1.,
        shift_tokens = 0,
        time_emb_dim = None,
        num_dense_connections = 0,
        **kwargs
    ):
        super().__init__()
        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, kwargs = groupby_prefix_and_trim('attn_', kwargs)

        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])
        self.num_dense_connections = num_dense_connections
        self.cascade = cascade
        self.class_conditional = class_conditional

        norm_class = nn.LayerNorm
        norm_fn = partial(norm_class, dim)

        if self.class_conditional:
            if self.cascade:
                default_block = ('a', 'c', 's', 'p', 'f')
            else:
                default_block = ('a', 'c', 'p', 'f')
        else:
            if self.cascade:
                default_block = ('a', 'c', 's', 'f')
            else:
                default_block = ('a', 'c', 'f')

        # calculate layer block order
        layer_types = default_block * depth
        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        # calculate token shifting
        shift_tokens = cast_tuple(shift_tokens, len(layer_types))
        self.scale_shift = exists(time_emb_dim)

        # iterate and construct layers
        for _, (layer_type, layer_shift_tokens) in enumerate(zip(self.layer_types, shift_tokens)):
            if layer_type == 'a':
                layer = Attention(dim, heads = heads, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads = heads, **attn_kwargs)
            elif layer_type == 's':
                layer = Attention(dim, heads = heads, **attn_kwargs)
            elif layer_type == 'p':
                layer = Attention(dim, heads = heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens
                layer = ShiftTokens(range(shift_range_lower, shift_range_upper), layer)

            if self.scale_shift and layer_type in ['f']:
                residual = TimeConditionedResidual(time_emb_dim, dim)
            else:
                residual_fn = Residual
                residual = residual_fn(dim, scale_residual = scale_residual, scale_residual_constant = scale_residual_constant)

            norm = norm_fn()

            self.layers.append(nn.ModuleList([
                norm,
                layer,
                residual
            ]))
        
        self.dense_projections = nn.ModuleList([nn.Linear(dim*2, dim) for _ in range(num_dense_connections)])
        
    def forward(
        self,
        x,
        mask = None,
        time_emb = None,
        cascade_emb = None,
        cascade_mask = None,
        class_emb = None,
        class_mask = None,
    ):
        assert not (self.cascade ^ exists(cascade_emb)), 'cascade_emb must be passed in if cascade is set to True'
        assert not (self.class_conditional ^ exists(class_emb)), 'class_emb must be passed in if class_conditional is set to True'

        dense_hiddens = []
        attn_idx = 0
        attn_maps = []
        for _, (layer_type, (norm, block, residual_fn)) in enumerate(zip(self.layer_types, self.layers)):
            if layer_type == 'a':
                dense_idx = attn_idx - (self.num_attn_layers - self.num_dense_connections)
                if dense_idx >= 0:
                    assert len(dense_hiddens) > 0, 'dense connections must be in order'
                    x = self.dense_projections[dense_idx](torch.cat([x, dense_hiddens.pop()], dim=-1))
                attn_idx += 1
            
            residual = x
            x = norm(x)

            if layer_type == 'a':
                out, attn_map = block(x, mask = mask)
            elif layer_type == 's':
                out, attn_map = block(x, cond = cascade_emb, mask = mask, cond_mask = cascade_mask)
            elif layer_type == 'p':
                out, attn_map = block(x, cond = class_emb, mask = mask, cond_mask = class_mask)
            elif layer_type == 'f':
                out = block(x)

            if self.scale_shift and layer_type in ['f']:
                x = residual_fn(out, residual, time_emb)
            else:
                x = residual_fn(out, residual)
            
            if layer_type == 'f' and len(dense_hiddens) < self.num_dense_connections:
                dense_hiddens.append(x)

            if layer_type in ('a','c','s','p'):
                attn_maps.append(attn_map)
            
        return x, attn_maps

class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
