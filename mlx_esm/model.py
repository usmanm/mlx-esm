import mlx.core as mx
import mlx.nn as nn


class Base(nn.Module):
  def __call__(self, _: mx.array) -> mx.array:
    raise NotImplementedError

  def loss(self, logits: mx.array, targets: mx.array) -> mx.array:
    return mx.mean(nn.losses.cross_entropy(logits, targets))


class Embedding(nn.Module):
  def __init__(self, num_embeddings: int, dims: int):
    super(Embedding, self).__init__()

    self.weight = mx.random.normal([num_embeddings, dims])

  def __call__(self, x: mx.array) -> mx.array:
    x = self.weight[x]
    x = x.reshape(x.shape[0], -1)
    return x


class MLP(Base):
  # NB: This is a simple MLP model with a single hidden layers. Implementing this to
  # make sure the training pipeline works.
  def __init__(
    self,
    embed_dims: int = 16,
    hidden_dims: int = 128,
    vocab_size: int = 32,
    context_size: int = 128,
  ):
    super(MLP, self).__init__()

    self.embed_dims = embed_dims
    self.vocab_size = vocab_size
    self.context_size = context_size
    self.hidden_dims = hidden_dims

    output_size = vocab_size * context_size

    self.layers = nn.Sequential(
      # Embedding layer
      Embedding(vocab_size, embed_dims),
      # Hidden layer
      nn.Linear(embed_dims * context_size, hidden_dims, bias=True),
      nn.Tanh(),
      # Output layer
      nn.Linear(hidden_dims, output_size, bias=True),
    )

  def __call__(self, x: mx.array) -> mx.array:
    x = self.layers(x)

    # We are getting back B x (L * V) tensor, where B = batch size, L = context
    # size, and V = vocab size. We need to reshape this tensor to B x L x V to
    # get the logits.
    logits = x.reshape(-1, self.context_size, self.vocab_size)

    return logits


class ESM1(Base):
  # These defaults have been scaled down here. The original defaults are:
  # num_layers: 33
  # embed_dims: 1280
  # num_attn_heads: 20
  # vocab_size: 32 (27 protein tokens + 5 special tokens)
  def __init__(
    self,
    num_layers: int = 4,
    embed_dims: int = 128,
    num_attn_heads: int = 4,
    vocab_size: int = 32,
  ):
    super(ESM1, self).__init__()

    self.num_layers = num_layers
    self.embed_dims = embed_dims
    self.num_attn_heads = num_attn_heads
    self.vocab_size = vocab_size

    # TODO: there is no equivalent for padding_idx in MLX. Figure out a way to
    # disable it during training.
    self.embed = nn.Embedding(vocab_size, embed_dims)

  def __call__(self, x: mx.array) -> mx.array:
    raise NotImplementedError
