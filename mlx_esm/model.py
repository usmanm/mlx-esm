import math
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_esm.data import Tokenizer


def count_parameters(params: Union[list, dict, mx.array]) -> int:
  if isinstance(params, mx.array):
    return params.size
  if isinstance(params, list):
    return sum([count_parameters(p) for p in params])
  if isinstance(params, dict):
    return sum([count_parameters(p) for p in params.values()])
  raise ValueError(f"unknown module type: {type(params)}")


class Embedding(nn.Module):
  def __init__(
    self,
    vocab_size: int,
    embed_dims: int,
    scale: Optional[float] = None,
    pad_idx: Optional[int] = None,
  ):
    super(Embedding, self).__init__()

    self.embed_dims = embed_dims
    self.vocab_size = vocab_size
    self.scale = scale or math.sqrt(1 / embed_dims)
    self.pad_idx = pad_idx

    self.weight = mx.random.normal([vocab_size, embed_dims]) * self.scale

    # The entries at pad_idx do not contribute to the gradient, so
    # the embedding vector at pad_idx will default to all zeros.
    #
    # TODO: Unclear how to disable updating the embedding vector at pad_idx
    # during training. In PyTorch, this seems to be implemented in C-level code.
    # See: https://github.com/pytorch/pytorch/blob/b85568a54a9c60986235ad1e0cc5dffc71b9d5b1/aten/src/ATen/native/Embedding.cpp#L108
    if self.pad_idx is not None:
      self.weight[self.pad_idx] = 0

  def __call__(self, x: mx.array) -> mx.array:
    # x: (B x L)
    # W: (V x C)
    # y: (B x L x C)
    y = self.weight[x]

    return y

  def __repr__(self):
    args = f"vocab_size={self.vocab_size}, embed_dims={self.embed_dims}"
    if self.pad_idx is not None:
      args += f", pad_idx={self.pad_idx}"
    return f"Embedding({args})"


# Transformers ditched recurrance in favor of self-attention. This helps with
# parallelization and makes it faster to train on GPUs, but the model loses
# the ability to understand the order of the sequence. To fix this, positional
# encodings are added to the input embeddings.
#
# For a deep dive into position encodings, see:
# https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
# https://github.com/facebookresearch/esm/blob/main/esm/modules.py#L260
class SinusoidalPositionalEmbedding(nn.Module):
  def __init__(self, embed_dims: int, pad_idx: int):
    super(SinusoidalPositionalEmbedding, self).__init__()
    assert embed_dims % 2 == 0, "embed_dims must be even"

    self.embed_dims = embed_dims
    self.pad_idx = pad_idx
    self._embeddings = None

  def embeddings(self, max_pos: int):
    if self._embeddings is None or max_pos > self._embeddings.shape[0]:
      # Creates a series of values that represent the frequencies W_k for the sinusoidal functions
      # where each subsequent frequency is an exponential step smaller than the previous one. We
      # represent this as a row vector of size half_dim.
      half_dim = self.embed_dims // 2
      freqs = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -(math.log(10000) / (half_dim - 1)))

      # Create a 2-D column-vector representing the position indices of shape (max_pos, 1)
      positions = mx.arange(max_pos, dtype=mx.float32)[..., None]

      # Create a 2-D matrix of shape (max_pos, half_dim).
      args = positions * freqs[None, ...]

      # Create a final 2-D matrix of shape (max_pos, embed_dim) by concatenating the
      # sin and cos of the scaled positions.
      embedding = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)

      # No impact of padding token.
      embedding[0, :] = 0

      self._embeddings = embedding

    return self._embeddings

  def positions(self, x: mx.array) -> mx.array:
    mask = x != self.pad_idx
    # We add 1 because postition 0 is reserved for the padding token.
    positions = mx.ones(x.shape, dtype=mx.int32) * (mx.arange(x.shape[1], dtype=mx.int32) + 1)
    return positions * mask

  def __call__(self, x: mx.array) -> mx.array:
    seq_len = x.shape[1]
    max_pos = seq_len + 1

    # (>=L, C)
    emb = self.embeddings(max_pos)[:max_pos, :]
    # (B, L)
    pos = self.positions(x)

    # (B, L, C)
    y = emb[pos]

    return y

  def __repr__(self):
    args = f"embed_dims={self.embed_dims}"
    if self.pad_idx is not None:
      args += f", pad_idx={self.pad_idx}"
    return f"SinusoidalPositionalEmbedding({args})"


# https://github.com/facebookresearch/esm/blob/main/esm/modules.py#L44
class LayerNorm(nn.Module):
  def __init__(self, embed_dims: int, eps=1e-12):
    """
    Construct a layernorm layer in the TF style (eps inside the sqrt).
    """
    super(LayerNorm, self).__init__()

    self.embed_dims = embed_dims
    self.eps = eps
    self.weight = mx.ones(embed_dims)
    self.bias = mx.zeros(embed_dims)

  def __call__(self, x: mx.array) -> mx.array:
    means = x.mean(-1, keepdims=True)
    variances = x.var(-1, keepdims=True)

    v = (x - means) / mx.sqrt(variances + self.eps)
    return (self.weight * v) + self.bias

  def __repr__(self):
    return f"LayerNorm(embed_dims={self.embed_dims})"


class MultiHeadAttention(nn.Module):
  def __init__(self, embed_dims: int, num_heads: int, bias: bool = True, add_bias_kv: bool = True):
    super(MultiHeadAttention, self).__init__()

    assert embed_dims % num_heads == 0, "embed_dims must be divisible by num_heads"

    self.embed_dims = embed_dims
    self.num_heads = num_heads
    self.bias = bias

    # TODO: implement adding bias to the key and value projections
    self.add_bias_kv = add_bias_kv

    # We use the same dimensions for queries, keys & values.
    qdims = embed_dims
    kdims = embed_dims
    vdims = embed_dims

    self.k_proj = nn.Linear(kdims, embed_dims, bias=bias)
    self.v_proj = nn.Linear(vdims, embed_dims, bias=bias)
    self.q_proj = nn.Linear(qdims, embed_dims, bias=bias)
    self.out_proj = nn.Linear(embed_dims, embed_dims, bias=bias)

  def __call__(self, queries: mx.array, keys: mx.array, values: mx.array) -> mx.array:
    H = self.num_heads

    queries = self.q_proj(queries)
    B, L, C = queries.shape
    assert self.embed_dims == C, "queries has incorrect embed_dims"

    scale = math.sqrt(1.0 / C)
    queries = queries * scale

    keys = self.k_proj(keys)
    values = self.v_proj(values)

    _, S, _ = keys.shape
    K = C // H
    assert K * H == C, "embed_dims must be divisible by num_heads"

    # Reshape the queries, keys, and values so we can compute the attention
    # on all heads in parallel.
    queries = queries.reshape(B, L, H, K).transpose([0, 2, 1, 3])  # (B, H, L, K)
    keys = keys.reshape(B, S, H, K).transpose([0, 2, 3, 1])  # (B, H, K, S)
    values = values.reshape(B, S, H, K).transpose([0, 2, 1, 3])  # (B, H, S, K)

    scores = queries @ keys  # (B, H, L, S)
    scores = nn.softmax(scores, axis=-1)

    values_hat = scores @ values  # (B, H, L, K)
    values_hat = values_hat.transpose([0, 2, 1, 3])  # (B, L, H, K)
    values_hat = values_hat.reshape(B, L, C)  # (B, L, C)

    return self.out_proj(values_hat)

  def __repr__(self):
    args = f"embed_dims={self.embed_dims}, "
    args += f"num_heads={self.num_heads}, "
    args += f"bias={self.bias}"
    return f"MultiHeadAttention({args})"


# The Transformer architecture is the driver of this latest wave of AI.
# Its simple architecture makes it easy to parallelize and train on GPUs,
# a bit upgrade from the RNNs and LSTMs of the past. This has unlocked our
# to train really large-scale language models, which have emergent properties
# that are useful for a wide variety of tasks. Meta trained a large LM on
# protein sequences, and with a few additional layers, it was able to achieve
# state-of-the-art results on protein folding and contact prediction.
#
# For an in-depth look at the Transformer architecture, see:
# https://nlp.seas.harvard.edu/annotated-transformer/
# https://jalammar.github.io/illustrated-transformer/
# https://github.com/facebookresearch/esm/blob/main/esm/modules.py#L84
class TransformerLayer(nn.Module):
  def __init__(
    self,
    embed_dims: int,
    ffn_embed_dims: int,
    num_attn_heads: int,
  ):
    super(TransformerLayer, self).__init__()

    self.embed_dims = embed_dims
    self.ffn_embed_dims = ffn_embed_dims
    self.num_attn_heads = num_attn_heads

    self.self_attn_layer_norm = LayerNorm(self.embed_dims)
    self.self_attn = nn.MultiHeadAttention(
      self.embed_dims,
      self.num_attn_heads,
      bias=True,
    )

    self.final_layer_norm = LayerNorm(self.embed_dims)
    self.fc1 = nn.Linear(self.embed_dims, self.ffn_embed_dims)
    self.fc2 = nn.Linear(self.ffn_embed_dims, self.embed_dims)

  def __call__(self, x: mx.array) -> mx.array:
    residual = x
    x = self.self_attn_layer_norm(x)
    x = self.self_attn(x, x, x)
    x = residual + x

    residual = x
    x = self.final_layer_norm(x)
    x = nn.gelu(self.fc1(x))
    x = self.fc2(x)
    x = residual + x

    return x

  def __repr__(self):
    args = f"embed_dims={self.embed_dims}, "
    args += f"ffn_embed_dims={self.ffn_embed_dims}, "
    args += f"num_attn_heads={self.num_attn_heads}"
    return f"TransformerLayer({args})"


class ESM1(nn.Module):
  # These defaults have been scaled down here. The original defaults are:
  # num_layers: 33
  # embed_dims: 1280
  # ffn_embed_dims: 5120
  # num_attn_heads: 20
  # final_bias: True
  def __init__(
    self,
    tokenizer: Tokenizer,
    num_layers: int = 4,
    embed_dims: int = 64,
    ffn_embed_dims: int = 256,
    num_attn_heads: int = 4,
    final_bias: bool = True,
  ):
    super(ESM1, self).__init__()

    self.tokenizer = tokenizer
    self.pad_idx = tokenizer.pad_idx
    self.num_layers = num_layers
    self.embed_dims = embed_dims
    self.ffn_embed_dims = ffn_embed_dims
    self.num_attn_heads = num_attn_heads
    self.vocab_size = tokenizer.vocab_size
    self.final_bias = final_bias

    self.init_submodules()

  def init_submodules(self):
    self.embed_tokens = Embedding(
      self.vocab_size,
      self.embed_dims,
      pad_idx=self.pad_idx,
      scale=math.sqrt(self.embed_dims),
    )
    self.embed_positions = SinusoidalPositionalEmbedding(self.embed_dims, self.pad_idx)

    self.transformer_layers = nn.Sequential(
      *[
        TransformerLayer(
          self.embed_dims,
          self.ffn_embed_dims,
          self.num_attn_heads,
        )
        for _ in range(self.num_layers)
      ]
    )

    self.out = nn.Linear(self.embed_dims, self.vocab_size, bias=self.final_bias)

  def __call__(self, x: mx.array) -> mx.array:
    # (B, L)
    assert x.ndim == 2

    tok_embed = self.embed_tokens(x)
    pos_embed = self.embed_positions(x)
    assert tok_embed.shape == pos_embed.shape

    logits = tok_embed + pos_embed
    logits = self.transformer_layers(logits)
    logits = self.out(logits)

    assert x.shape == logits.shape[:2] and logits.shape[2] == self.vocab_size

    return logits

  def num_parameters(self):
    return count_parameters(self.parameters())
