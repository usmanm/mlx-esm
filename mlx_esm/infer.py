import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_esm.data import Tokenizer
from mlx_esm.model import Base


def unmask(
  model: Base,
  masked_seq: str,
  max_iters: int = 100,
) -> str:
  if len(masked_seq) > model.max_seq_len:
    raise ValueError("sequence exceeds context size")

  tokenizer = Tokenizer()

  model.eval()
  mx.eval(model.parameters())

  def eval_fn(model: Base, x: mx.array) -> mx.array:
    return model(x)

  logits_and_grad_fn = nn.value_and_grad(model, eval_fn)

  iter_num = 0
  start_seq = f"^{masked_seq}$"
  toks = tokenizer.encode(start_seq)
  x = mx.array([mx.array(toks)], dtype=mx.int32)

  print_sequence(tokenizer, toks, "🌱")

  while iter_num < max_iters:
    toks = x[0]

    iter_num += 1
    if tokenizer.mask_idx not in toks.tolist():
      break

    # forward the model
    logits, _ = logits_and_grad_fn(model, x)
    x = logits_to_next_x(logits)
    print_sequence(tokenizer, toks, "🕐")

  emoji = "🌳" if tokenizer.mask_idx not in toks.tolist() else "🤷‍♂️"
  return print_sequence(tokenizer, toks, emoji)


def logits_to_next_x(logits: mx.array) -> mx.array:
  probs = np.array(mx.softmax(logits, axis=-1))

  # This is equivalent to multinomial in PyTorch.
  samples = np.array(
    [
      np.random.choice(range(probs.shape[-1]), p=prob)
      for prob in probs.reshape(-1, probs.shape[-1])
    ]
  )
  sample = samples.reshape(probs.shape[0], probs.shape[1])

  return mx.array(sample)


def print_sequence(tokenizer: Tokenizer, toks: mx.array, emoji: str) -> str:
  s = "".join(tokenizer.decode(toks)).strip().replace("^", "").replace("$", "")
  print(emoji + " " + s)
  return s
