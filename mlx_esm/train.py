import random
import time
from typing import Optional, Tuple

import mlx.core as mx

from mlx_esm.data import Sequence, Tokenizer
from mlx_esm.model import ESM2


def set_seed(seed):
  random.seed(seed)
  mx.random.seed(seed)


class Config(object):
  def __init__(
    self,
    seed: int = 42,
    max_iters: int = 100_000,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    mask_rate: float = 0.15,
  ):
    self.seed = seed
    self.max_iters = max_iters
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.mask_rate = mask_rate


class TrainingLoader(object):
  def __init__(self, data: list[Sequence], mask_rate: float):
    self.data = data
    self.mask_rate = mask_rate
    self.tokenizer = Tokenizer()

  def next_batch(self, size: int) -> Tuple[mx.array, mx.array]:
    tokenizer = self.tokenizer
    batch: list[Sequence] = random.sample(self.data, size)
    tokenized_batch = [
      mx.array(tokenizer.encode(tokenizer.tokenize(seq)), dtype=mx.uint32) for seq in batch
    ]
    return (
      mx.stack(tokenized_batch),
      mx.stack(tokenized_batch),
    )


class Trainer(object):
  def __init__(
    self, data: list[Sequence], model: Optional[ESM2] = None, config: Optional[Config] = None
  ):
    self.config = config if config is not None else Config()
    self.model = model if model is not None else ESM2()
    self.data = data
    self.loader = TrainingLoader(data, self.config.mask_rate)

    # variables that will be assigned to trainer class later for logging and etc
    self.iter_num = 0
    self.iter_time = 0.0
    self.iter_dt = 0.0

  def run(self):
    model = self.model
    config = self.config
    loader = self.loader

    set_seed(config.seed)

    self.iter_num = 0
    self.iter_time = time.time()

    while self.iter_num < config.max_iters:
      batch = loader.next_batch(config.batch_size)
      x, y = batch

      # forward the model
      logits, self.loss = model(x, y)

      # backprop and update the parameters
      model.zero_grad(set_to_none=True)
      self.loss.backward()

      self.update_and_log_metrics()

  def update_and_log_metrics(self):
    self.iter_num += 1
    now = time.time()
    self.iter_dt = now - self.iter_time
    self.iter_time = now

    if self.iter_num == 0 or self.iter_num % 1000 != 0:
      return
