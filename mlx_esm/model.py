from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_esm.data import Tokenizer


class ESM2(nn.Module):
  # The defaults have been scaled down here. The original defaults are:
  # num_layers: 33
  # embed_dims: 1280
  # num_attn_heads: 20
  def __init__(self, num_layers: int = 4, embed_dims: int = 128, num_attn_heads: int = 4):
    super(ESM2, self).__init__()
    self.num_layers = num_layers
    self.embed_dims = embed_dims
    self.num_attn_heads = num_attn_heads

    self.tokenizer = Tokenizer()

    # TODO: there is no equivalent for padding_idx in MLX. Figure out a way to
    # disable it during training.
    self.embed = nn.Embedding(self.tokenizer.vocab_size, embed_dims)

  def __call__(
    self, x: mx.array, y: Optional[mx.array] = None
  ) -> Tuple[mx.array, Optional[int]]:
    x = self.embed(x)
    return x, None
