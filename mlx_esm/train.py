import random
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_esm.data import BatchTokenizer, Tokenizer, load_uniparc_dbs
from mlx_esm.model import MLP, Base


def set_seed(seed):
  random.seed(seed)
  mx.random.seed(seed)


@dataclass
class Config(object):
  seed: int = 42
  dbs: list[int] = field(default_factory=lambda: [1])
  max_iters: int = 100_000
  batch_size: int = 16
  learning_rate: float = 0.01
  mask_rate: float = 0.15
  context_size = 64


class TrainingLoader(object):
  def __init__(
    self,
    dbs: list[int],
    batch_size: int,
    context_size: int,
    mask_rate: float,
    dynamic_padding: bool,
  ):
    self.dbs = sorted(dbs)
    self.batch_size = batch_size
    self.context_size = context_size
    self.mask_rate = mask_rate
    self.batch_tokenizer = BatchTokenizer(
      context_size=context_size,
      dynamic_padding=dynamic_padding,
    )
    self.data: Optional[list[str]] = None
    self.dynamic_padding = dynamic_padding

  def load(self):
    sequences = load_uniparc_dbs(self.dbs)
    # We filter out sequences that are too long to fit in the context size.
    # Subtracting 2 from the context size to account for the CLS and EOS tokens.
    data = [s for s in sequences if len(s) <= self.context_size - 2]
    if self.dynamic_padding:
      data = sorted(data, key=lambda s: len(s))
    self.data = data

  @property
  def tokenizer(self) -> Tokenizer:
    return self.batch_tokenizer.tokenizer

  def next_batch(self) -> Tuple[mx.array, mx.array]:
    if self.data is None:
      raise Exception("data has not been loaded yet")

    batch: list[str] = random.sample(self.data, self.batch_size)

    encoded = self.batch_tokenizer.encode(batch)
    shape: list[int] = list(encoded.shape)

    tokenizer = self.batch_tokenizer.tokenizer
    pad_idx, mask_idx = tokenizer.pad_idx, tokenizer.mask_idx

    # We should not mask padding tokens because they do not form part of the
    # underlying protein sequence. We do include CLS & EOS tokens because they
    # are part of the sequence in so far that they do inform us about the structure
    # of the protein.
    #
    # TODO: should we mask CLS and EOS tokens?
    can_mask = encoded != pad_idx

    # We should mask tokens with a probability of `mask_rate`. We will use a
    # uniform distribution to determine which tokens to mask. By multiplying
    # the result of the uniform distribution by `can_mask`, we ensure that we
    # do not mask tokens that are padding tokens.
    should_mask = (mx.random.uniform(0, 1, shape, dtype=mx.float32) < self.mask_rate) * can_mask
    should_not_mask = 1 - should_mask

    # BERT differs from ESM-1 in how it does masking. In ESM-1, we mask tokens
    # with the mask token only, while in BERT the masking strategy is a bit more
    # complex. We will implement the ESM-1 masking strategy here.
    masked = (encoded * should_not_mask) + (should_mask * mask_idx)

    return (masked, encoded)


class Trainer(object):
  def __init__(self, model: Optional[Base] = None, config: Optional[Config] = None):
    config = config or Config()

    self.config = config
    self.loader = TrainingLoader(
      config.dbs,
      config.batch_size,
      config.context_size,
      config.mask_rate,
      False,
    )
    self.model = model or MLP(
      tokenzier=self.loader.tokenizer,
      context_size=config.context_size,
    )

    # variables that will be assigned to trainer class later for logging and etc
    self.iter_num = 0
    self.last_log_time = 0.0
    self.losses: list[float] = []

  def load(self):
    self.loader.load()

  def run(self, max_iters: Optional[int] = None):
    model = self.model
    config = self.config
    loader = self.loader

    model.train()
    mx.eval(model.parameters())

    def loss_fn(model: Base, x: mx.array, targets: mx.array) -> mx.array:
      return mx.mean(nn.losses.cross_entropy(model(x), targets))

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=config.learning_rate)

    set_seed(config.seed)

    self.iter_num = 0
    self.last_log_time = time.time()

    while self.iter_num < (max_iters or config.max_iters):
      x, y = loader.next_batch()

      # forward the model
      loss, grads = loss_and_grad_fn(model, x, y)

      # backprop and update the parameters
      # Update the optimizer state and model parameters
      # in a single call
      optimizer.update(model, grads)

      # Force a graph evaluation
      mx.eval(model.parameters(), optimizer.state)

      self.log_metrics(loss.item())

  def log_metrics(self, loss: float, bucket_size: int = 1000):
    self.iter_num += 1
    self.losses.append(loss)

    if self.iter_num == 0 or self.iter_num % bucket_size != 0:
      return

    now = time.time()
    duration = now - self.last_log_time
    self.last_log_time = now

    # Compute the average loss over the last batch sized window.
    losses = self.losses[-bucket_size:]
    avg_loss = round(sum(losses) / len(losses), 4)

    print(f"🚂 iter={self.iter_num} duration={round(duration, 1)} loss={avg_loss}")

  def plot_loss(self, bucket_size: int = 1000):
    xs = range(self.iter_num)
    ys = self.losses

    # We will bucket the losses to make the plot more readable.
    bucketed_xs = range(0, len(xs), bucket_size)
    bucketed_ys = [
      sum(ys[i : i + bucket_size]) / bucket_size for i in range(0, len(ys), bucket_size)
    ]

    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("Training Loss")
    plt.plot(bucketed_xs, bucketed_ys)
    plt.show()
