import random
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tqdm.auto import tqdm

from mlx_esm.data import DataSplit, Loader
from mlx_esm.model import ESM1
from mlx_esm.tokenizer import Tokenizer


def set_seed(seed: int):
  random.seed(seed)
  mx.random.seed(seed)


@dataclass
class Config(object):
  # The answer to the ultimate question of life, the universe, and everything.
  seed: int = 42

  dbs: list[int] = field(default_factory=lambda: [1, 2, 3])
  max_iters: int = 100_000
  batch_size: int = 16
  learning_rate: float = 0.01
  mask_rate: float = 0.15

  # The maximum sequence length for proteins to train on. This effectively
  # limits the "context size" of the model. Larger contexts are slower to
  # train on my GPU-poor MacBook Air.
  max_seq_len: int = 126


class Trainer(object):
  def __init__(self, model: Optional[ESM1] = None, config: Optional[Config] = None):
    self.config = config or Config()
    self.model = model or ESM1(Tokenizer())

    self.loader = Loader(
      self.model.tokenizer,
      self.config.dbs,
      self.config.batch_size,
      self.config.max_seq_len,
      self.config.mask_rate,
    )

    self.losses: dict[DataSplit, list[float]] = {
      "train": [],
      "validate": [],
    }

    set_seed(self.config.seed)

  def load(self):
    print("📥 loading data")
    self.loader.load()

  def train(self, max_iters: Optional[int] = None):
    return self.run("train", max_iters or self.config.max_iters)

  def validate(self, max_iters: Optional[int] = None):
    return self.run("train", max_iters or int(self.config.max_iters * 0.1))

  def run(self, split: DataSplit, max_iters: int):
    model = self.model
    config = self.config
    loader = self.loader

    if split == "train":
      model.train()
      desc = "🚂 training"
    else:
      model.eval()
      desc = "🔍 validating"

    mx.eval(model.parameters())

    def loss_fn(model: ESM1, x: mx.array, targets: mx.array) -> mx.array:
      return mx.mean(nn.losses.cross_entropy(model(x), targets))

    optimizer = optim.SGD(learning_rate=config.learning_rate, momentum=0.8)

    # https://ml-explore.github.io/mlx/build/html/usage/compile.html#compiling-training-graphs
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x: mx.array, y: mx.array):
      # forward the model
      loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
      loss, grads = loss_and_grad_fn(model, x, y)

      # backprop and update the parameters
      # Update the optimizer state and model parameters
      # in a single call
      if split == "train":
        optimizer.update(model, grads)

      return loss

    self.iter_num = 0
    self.last_log_time = time.time()

    loop = tqdm(
      range(max_iters or config.max_iters),
      ncols=120,
      desc=desc,
      postfix={"loss": "NaN"},
    )
    for _ in loop:
      x, y = loader.next_batch("train")
      loss = step(x, y)
      mx.eval(state)

      avg_loss = self.avg_loss(split, loss.item())
      loop.set_postfix({"loss": f"{avg_loss:.4f}"})

  def avg_loss(self, split: DataSplit, loss: float, bucket_size: int = 1000):
    losses = self.losses[split]
    losses.append(loss)

    window = losses[-bucket_size:]
    avg_loss = sum(window) / len(window)
    return avg_loss

  def plot_loss(self, split: DataSplit, bucket_size: int = 1000):
    losses = self.losses[split]

    xs = range(len(losses))
    ys = losses

    # We will bucket the losses to make the plot more readable.
    bucketed_xs = range(0, len(xs), bucket_size)
    bucketed_ys = [
      sum(ys[i : i + bucket_size]) / bucket_size for i in range(0, len(ys), bucket_size)
    ]

    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.plot(bucketed_xs, bucketed_ys)
    plt.show()
