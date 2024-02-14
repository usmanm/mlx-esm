import mlx.core as mx
import numpy as np

from mlx_esm.data import Tokenizer
from mlx_esm.model import Base


def generate(model: Base, length: int = 32, max_iters: int = 256) -> str:
  # And then god said, let there be more proteins.
  mask_len = min(model.context_size - 1, length)

  pad_len = model.context_size - mask_len - 2
  start_seq = "^" + "*" * min(model.context_size - 1, length) + "$" + "%" * pad_len
  return impl(model, start_seq, max_iters)


def unmask(model: Base, masked_seq: str, max_iters: int = 256) -> str:
  if len(masked_seq) > model.max_seq_len:
    raise ValueError("sequence exceeds context size")
  return impl(model, f"^{masked_seq}$", max_iters)


def impl(model: Base, input: str, max_iters: int) -> str:
  if len(input) > model.context_size:
    raise ValueError("input exceeds context size")

  tokenizer = Tokenizer()

  model.eval()
  mx.eval(model.parameters())

  toks = tokenizer.encode(input).tolist()
  toks = mx.array(toks + [tokenizer.pad_idx] * (model.context_size - len(toks)))
  x = mx.array([toks], dtype=mx.int32)

  print_sequence(tokenizer, toks, "🌱")

  for i in range(max_iters):
    toks = x[0]

    if is_sequence_legit(tokenizer, toks):
      break

    if i > 0:
      print_sequence(tokenizer, toks, "🕐")

    # forward the model
    logits = model(x)
    x = compute_next_x(tokenizer, x, logits)

  emoji = "🌳" if is_sequence_legit(tokenizer, toks) else "🍂"
  return print_sequence(tokenizer, toks, emoji)


def compute_next_x(tokenizer: Tokenizer, x: mx.array, logits: mx.array) -> mx.array:
  probs = np.array(mx.softmax(logits, axis=-1))

  # This is equivalent to multinomial in PyTorch.
  samples = np.array(
    [
      np.random.choice(range(probs.shape[-1]), p=prob)
      for prob in probs.reshape(-1, probs.shape[-1])
    ]
  )
  sample = mx.array(samples.reshape(probs.shape[0], probs.shape[1]))

  # We only swap out the first mask token to generate proetins using
  # an autoregressive style. My theory is that this will lead to
  # more realistic sequences because the model sees it grow rather
  # than guessing the entire sequence in a single shot.
  mask_all = x == tokenizer.mask_idx
  mask_first = mx.zeros_like(mask_all)
  mask_first[mx.arange(mask_all.shape[0]), mask_all.argmax(axis=1)] = 1

  sample = sample * mask_first
  x = x * (1 - mask_first)

  return x + sample


def print_sequence(tokenizer: Tokenizer, toks: mx.array, emoji: str) -> str:
  s = "".join(tokenizer.decode(toks)).strip().rstrip("%").rstrip("$").lstrip("^")
  print(emoji + " " + s)
  return s


def is_sequence_legit(tokenizer: Tokenizer, toks: mx.array) -> bool:
  tokens: list[int] = toks.tolist()

  # Any invalid tokens in the sequence?
  invalid_toks = [tokenizer.mask_idx, tokenizer.unk_idx]
  if any(invalid_tok in tokens for invalid_tok in invalid_toks):
    return False

  # Remove any padding
  while tokens and tokens[-1] == tokenizer.pad_idx:
    tokens.pop()

  # Does the sequence start and end with the correct tokens?
  if tokens[0] != tokenizer.cls_idx or tokens[-1] != tokenizer.eos_idx:
    return False

  # Are only protein tokens in middle of the sequence?
  protein_toks: list[int] = tokenizer.encode("".join(tokenizer.protein_toks)).tolist()
  return all(tok in protein_toks for tok in tokens[1:-1])
